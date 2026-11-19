"""Tests de formes, d'entraînement court et de sauvegarde."""

from __future__ import annotations

import torch

from src.data import make_sine_dataset
from src.models import EnergyTransformer, MiniTransformer, SimpleLSTM, SimpleRNN
from src.training import get_device, load_state_dict, predict, save_state_dict, set_seed, train_full_batch


def test_sequence_model_output_shapes():
    batch = torch.randn(8, 10, 1)
    for model in (SimpleRNN(), SimpleLSTM(), MiniTransformer()):
        output = model(batch)
        assert output.shape == (8, 1)


def test_energy_transformer_output_shape():
    batch = torch.randn(4, 24, 1)
    output = EnergyTransformer()(batch)
    assert output.shape == (4, 1)


def test_short_training_decreases_the_loss():
    set_seed(42)
    data = make_sine_dataset()
    model = SimpleRNN()
    losses = train_full_batch(
        model,
        data["X_train"],
        data["y_train"],
        epochs=40,
        lr=0.01,
        print_every=0,
        device=torch.device("cpu"),
    )
    assert losses[-1] < losses[0]


def test_checkpoint_roundtrip(tmp_path):
    set_seed(7)
    model = SimpleLSTM()
    features = torch.randn(5, 10, 1)
    model.eval()
    before = predict(model, features, device=torch.device("cpu"))
    path = save_state_dict(model, tmp_path / "lstm.pt")
    restored = SimpleLSTM()
    load_state_dict(restored, path, device=torch.device("cpu"))
    restored.eval()
    after = predict(restored, features, device=torch.device("cpu"))
    torch.testing.assert_close(before, after)


def test_device_detection_returns_a_torch_device():
    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}
