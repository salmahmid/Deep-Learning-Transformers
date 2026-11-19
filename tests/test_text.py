"""Tests du masque causal, du décalage des cibles et du masque de padding."""

from __future__ import annotations

import torch

from src.data import TRAIN_SENTENCES, build_minibert_corpus, build_word_vocab, encode_sentences
from src.text_models import (
    MASK,
    MiniBERT,
    MiniGPT,
    accuracy_on_selected,
    causal_mask,
    corrupt_for_mlm,
)


def test_causal_mask_blocks_the_future_only():
    mask = causal_mask(6)
    assert mask.dtype == torch.bool
    assert mask.shape == (6, 6)
    assert torch.equal(mask, torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1))
    assert not bool(mask[3, 3])
    assert not bool(mask[3, 2])
    assert bool(mask[3, 4])


def test_minigpt_past_logits_ignore_a_future_edit():
    torch.manual_seed(42)
    model = MiniGPT(vocab_size=30, context=32, dropout=0.0)
    model.eval()
    tokens = torch.randint(0, 30, (3, 32))
    edited = tokens.clone()
    position = 12
    edited[:, position] = (edited[:, position] + 3) % 30
    with torch.no_grad():
        original = model(tokens)
        changed = model(edited)
    torch.testing.assert_close(original[:, :position], changed[:, :position], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(original[:, position], changed[:, position])


def test_targets_are_shifted_by_one_character():
    tokens = torch.arange(80)
    context = 64
    inputs = tokens[:context]
    targets = tokens[1 : context + 1]
    assert targets.shape == inputs.shape
    assert torch.equal(targets[:-1], inputs[1:])
    assert int(targets[-1]) == context


def test_mlm_corruption_rules():
    generator = torch.Generator().manual_seed(123)
    sentences = list(TRAIN_SENTENCES) * 4
    vocabulary, _ = build_word_vocab(TRAIN_SENTENCES)
    ids = encode_sentences(sentences, vocabulary, max_length=32)
    corrupted, labels, selected = corrupt_for_mlm(ids, vocab_size=len(vocabulary), generator=generator)
    ordinary = ids >= 5
    assert torch.all(selected <= ordinary)
    assert torch.all(labels[~selected] == -100)
    assert torch.all(labels[selected] == ids[selected])
    for row in range(ids.size(0)):
        if bool(ordinary[row].any()):
            assert int(selected[row].sum()) >= 1
    # Les positions non sélectionnées ne sont pas modifiées.
    assert torch.equal(corrupted[~selected], ids[~selected])
    changed = corrupted.ne(ids) & selected
    assert torch.all((corrupted[changed] == MASK) | (corrupted[changed] >= 5))
    rate = float(selected[ordinary].float().mean())
    assert 0.05 < rate < 0.40


def test_padding_values_do_not_change_real_token_logits():
    torch.manual_seed(4)
    model = MiniBERT(vocab_size=40, max_length=16, dropout=0.1)
    model.eval()
    ids = torch.tensor([[2, 8, 9, 10, 11, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    pad_mask = torch.zeros_like(ids, dtype=torch.bool)
    pad_mask[:, 6:] = True
    edited = ids.clone()
    edited[:, 6:] = torch.arange(12, 22)
    with torch.no_grad():
        original = model(ids, pad_mask)
        changed = model(edited, pad_mask)
    torch.testing.assert_close(original[:, :6], changed[:, :6], atol=1e-5, rtol=1e-5)


def test_loss_ignores_unselected_positions():
    vocabulary, _ = build_word_vocab(TRAIN_SENTENCES)
    train, _ = build_minibert_corpus(copies_train=1, copies_val=1)
    ids = encode_sentences(train[:2], vocabulary)
    _, labels, selected = corrupt_for_mlm(ids, len(vocabulary), torch.Generator().manual_seed(1))
    logits = torch.zeros(ids.shape[0], ids.shape[1], len(vocabulary))
    correct, total = accuracy_on_selected(logits, labels)
    assert total == int(selected.sum())
    assert correct == 0
