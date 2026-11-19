"""Tests des fenêtres, de la normalisation et du fichier UCI."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import (
    RAW_DIR,
    TEXT_NAME,
    TRAIN_SENTENCES,
    VAL_SENTENCES,
    ZIP_NAME,
    _read_power_frame,
    build_word_vocab,
    encode_sentences,
    hourly_mean_power,
    make_sine_dataset,
    tokenize_words,
    window_split,
)


def test_sine_shapes_and_targets():
    data = make_sine_dataset()
    assert data["periode"].shape == (200,)
    assert data["serie"].shape == (200,)
    assert data["X"].shape == (190, 10, 1)
    assert data["y"].shape == (190, 1)
    assert data["X_train"].shape == (150, 10, 1)
    assert data["y_train"].shape == (150, 1)
    assert data["X_test"].shape == (40, 10, 1)
    assert data["y_test"].shape == (40, 1)
    assert torch.allclose(data["X"][0, :, 0], data["serie"][:10])
    assert torch.allclose(data["y"][0], data["serie"][10])
    assert torch.allclose(data["X"][5, :, 0], data["serie"][5:15])
    assert torch.equal(data["X_train"], data["X"][:150])
    assert torch.equal(data["X_test"], data["X"][150:])


def test_sine_matches_formula():
    data = make_sine_dataset()
    expected = torch.sin(torch.linspace(0, 20, 200))
    assert torch.allclose(data["serie"], expected)


def _hourly_series(size: int = 400, seed: int = 0) -> pd.Series:
    generator = np.random.default_rng(seed)
    values = np.sin(np.linspace(0, 30, size)) + generator.normal(0, 0.05, size)
    index = pd.date_range("2020-01-01", periods=size, freq="h")
    return pd.Series(values, index=index, name="global_active_power_kw")


def test_split_ratios_and_order():
    prepared = window_split(_hourly_series())
    assert prepared["fin_train"] == int(0.70 * prepared["n"])
    assert prepared["fin_val"] == int(0.85 * prepared["n"])
    assert np.all(prepared["positions"][prepared["train_mask"]] < prepared["fin_train"])
    assert np.all(prepared["positions"][prepared["val_mask"]] >= prepared["fin_train"])
    assert np.all(prepared["positions"][prepared["val_mask"]] < prepared["fin_val"])
    assert np.all(prepared["positions_test"] >= prepared["fin_val"])
    assert prepared["X_train"].shape[1:] == (24, 1)
    assert len(prepared["X_train"]) + len(prepared["X_val"]) + len(prepared["X_test"]) == len(prepared["X"])


def test_windows_use_only_the_past():
    serie = _hourly_series()
    prepared = window_split(serie)
    raw = serie.to_numpy()
    mean, std = prepared["mean"], prepared["std"]
    for row, target, position in zip(prepared["X"][:30], prepared["y"][:30], prepared["positions"][:30]):
        expected_window = (raw[position - 24 : position] - mean) / std
        expected_target = (raw[position] - mean) / std
        assert np.allclose(row[:, 0].numpy(), expected_window)
        assert np.allclose(target.numpy(), expected_target)
        assert position - 24 >= 0


def test_normalization_does_not_use_the_future():
    serie = _hourly_series(500)
    original = window_split(serie)
    changed = serie.copy()
    changed.iloc[original["fin_val"] :] = 50_000.0
    altered = window_split(changed)
    assert altered["mean"] == pytest.approx(original["mean"])
    assert altered["std"] == pytest.approx(original["std"])
    train_direct = float(serie.iloc[: original["fin_train"]].mean())
    assert original["mean"] == pytest.approx(train_direct)


def test_incomplete_windows_are_removed():
    serie = _hourly_series()
    serie.iloc[100] = np.nan
    prepared = window_split(serie)
    assert 100 not in set(prepared["positions"].tolist())
    leaked = [position for position in prepared["positions"] if 100 < position <= 124]
    assert leaked == []


def test_hourly_count_threshold_and_missing_marks():
    start = pd.Timestamp("2024-01-01 00:00:00")
    rows = []
    for minute in range(50):
        rows.append((start + pd.Timedelta(minutes=minute), 2.0))
    for minute in range(40):
        rows.append((start + pd.Timedelta(hours=1, minutes=minute), 4.0))
    for minute in range(60):
        value = np.nan if minute < 20 else 1.0
        rows.append((start + pd.Timedelta(hours=2, minutes=minute), value))
    frame = pd.DataFrame(rows, columns=["datetime", "Global_active_power"])
    serie = hourly_mean_power(frame, min_count=45)
    assert serie.loc[pd.Timestamp("2024-01-01 00:00:00")] == pytest.approx(2.0)
    assert np.isnan(serie.loc[pd.Timestamp("2024-01-01 01:00:00")])
    assert np.isnan(serie.loc[pd.Timestamp("2024-01-01 02:00:00")])


def test_validation_words_belong_to_the_training_vocabulary():
    vocabulary, _ = build_word_vocab(TRAIN_SENTENCES)
    missing = [
        word
        for sentence in VAL_SENTENCES
        for word in tokenize_words(sentence)
        if word not in vocabulary
    ]
    assert missing == []


def test_sentence_encoding_uses_special_tokens():
    vocabulary, inverse = build_word_vocab(TRAIN_SENTENCES)
    encoded = encode_sentences(["le chat dort sur le canapé"], vocabulary, max_length=32)
    assert encoded.shape == (1, 32)
    assert int(encoded[0, 0]) == vocabulary["[CLS]"]
    assert inverse[int(encoded[0, 1])] == "le"
    assert inverse[int(encoded[0, 6])] == "canapé"
    assert inverse[int(encoded[0, 7])] == "[SEP]"
    assert torch.all(encoded[0, 8:] == vocabulary["[PAD]"])


def test_real_file_header_and_question_marks():
    archive = RAW_DIR / ZIP_NAME
    if not archive.exists():
        pytest.skip("Archive UCI absente.")
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open(TEXT_NAME) as handle:
            preview = handle.read(120).decode("ascii")
            handle.seek(0)
            frame = _read_power_frame(handle, nrows=1500)
    assert preview.startswith("Date;Time;Global_active_power;")
    assert frame["datetime"].iloc[0] == pd.Timestamp("2006-12-16 17:24:00")
    assert frame["Global_active_power"].dtype.kind == "f"
    serie = hourly_mean_power(frame)
    assert len(serie) > 0
