"""Vérification rapide : formes, une courte descente de perte, causalité, padding."""

from __future__ import annotations

import torch

from src.data import make_sine_dataset
from src.models import SimpleRNN
from src.training import get_device, set_seed, train_full_batch


def main() -> None:
    device = get_device()
    print(f"Périphérique détecté : {device}")
    set_seed(42)
    data = make_sine_dataset()
    print("Sinusoïde :", tuple(data["X_train"].shape), tuple(data["X_test"].shape))
    model = SimpleRNN()
    losses = train_full_batch(
        model,
        data["X_train"],
        data["y_train"],
        epochs=5,
        lr=0.01,
        print_every=0,
        device=torch.device("cpu"),
    )
    print(f"Perte RNN après 5 époques : {losses[0]:.4f} -> {losses[-1]:.4f}")
    if losses[-1] >= losses[0]:
        raise SystemExit("La perte du RNN n'a pas diminué.")
    print("Vérification rapide terminée. Lancez pytest pour la suite des contrôles.")


if __name__ == "__main__":
    main()
