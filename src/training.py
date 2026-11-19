"""Entraînement, reproductibilité et choix du processeur."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Fixe les graines Python, NumPy et PyTorch.

    Deux exécutions sur le même ordinateur et le même matériel se
    ressemblent davantage. Un GPU, surtout MPS sur Mac, peut encore
    produire de petites différences numériques.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        manual_seed = getattr(torch.mps, "manual_seed", None)
        if manual_seed is not None:
            manual_seed(seed)


def get_device() -> torch.device:
    """Choisit cuda, puis MPS, puis le CPU.

    La variable d'environnement DLT_DEVICE peut forcer ``cpu``, ``cuda``,
    ``mps`` ou ``auto``.
    """
    import os

    choice = os.environ.get("DLT_DEVICE", "auto").strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DLT_DEVICE=cuda, mais CUDA n'est pas disponible.")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("DLT_DEVICE=mps, mais MPS n'est pas disponible.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Compte les paramètres entraînables."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_full_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.01,
    print_every: int = 50,
    device: torch.device | None = None,
) -> list[float]:
    """Entraîne un petit modèle de régression sur tout le lot à chaque époque.

    C'est le protocole du TP pour la sinusoïde : Adam, MSE, et un affichage
    de la perte toutes les 50 époques.
    """
    device = device or get_device()
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for epoch in range(epochs):
        predictions = model(x)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if print_every and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1:3d} | Erreur : {loss.item():.6f}")
    return losses


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Calcule les prédictions sans gradient et les ramène sur le CPU."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    return model(x.to(device)).cpu()


@torch.no_grad()
def mean_squared_error_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """MSE moyenne sur un DataLoader. Le modèle passe en mode eval."""
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        predictions = model(features)
        total += float(criterion(predictions, targets).item())
        count += targets.numel()
    if count == 0:
        raise RuntimeError("Le DataLoader est vide : impossible de calculer la MSE.")
    return total / count


def train_with_validation(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 20,
    lr: float = 0.001,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Entraîne avec Adam et MSE, puis recharge la meilleure époque de validation.

    Toutes les époques demandées sont exécutées. Le meilleur état, au sens
    de la perte de validation, est ensuite restauré.
    """
    device = device or get_device()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * targets.numel()
            seen += targets.numel()
        train_loss = running / seen
        val_loss = mean_squared_error_loader(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            marker = "  <- meilleur"
        print(
            f"Epoch {epoch + 1:2d} | train {train_loss:.6f} | "
            f"val {val_loss:.6f}{marker}"
        )

    if best_state is None:
        raise RuntimeError("Aucun état de modèle n'a été enregistré.")
    model.load_state_dict(best_state)
    print(f"Meilleur modèle restauré : époque {best_epoch}, MSE validation {best_val:.6f}")
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, path)
        print(f"Checkpoint enregistré : {path}")
    return history


def save_state_dict(model: nn.Module, path: str | Path) -> Path:
    """Enregistre les poids du modèle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_state_dict(model: nn.Module, path: str | Path, device: torch.device | None = None) -> nn.Module:
    """Recharge des poids enregistrés par save_state_dict."""
    device = device or get_device()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    return model


def _batch_starts(num_tokens: int, context: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    if num_tokens <= context:
        raise ValueError("Le corpus est trop court pour la longueur de contexte demandée.")
    return torch.randint(0, num_tokens - context, (batch_size,), generator=generator)


def _windows_from_starts(tokens: torch.Tensor, starts: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([tokens[int(start) : int(start) + context] for start in starts])
    targets = torch.stack([tokens[int(start) + 1 : int(start) + context + 1] for start in starts])
    return inputs, targets


def language_model_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    context: int,
    device: torch.device,
    batch_size: int = 32,
    max_batches: int | None = None,
    generator: torch.Generator | None = None,
) -> float:
    """Perte d'entropie croisée moyenne, sans gradient."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    usable = len(tokens) - context
    if usable <= 0:
        raise ValueError("Pas assez de tokens pour évaluer le modèle de langue.")
    starts = torch.arange(0, usable, context)
    if max_batches is not None:
        starts = starts[: max_batches * batch_size]
    total = 0.0
    count = 0
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            inputs, targets = _windows_from_starts(tokens, batch_starts, context)
            logits = model(inputs.to(device))
            vocab = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab), targets.to(device).reshape(-1))
            total += float(loss.item()) * targets.numel()
            count += targets.numel()
    return total / count


def train_language_model(
    model: nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    context: int = 64,
    epochs: int = 40,
    steps_per_epoch: int = 60,
    batch_size: int = 32,
    lr: float = 3e-3,
    device: torch.device | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Entraîne un mini-GPT caractère par caractère.

    Chaque exemple est une fenêtre de ``context`` caractères. La cible est
    cette même fenêtre décalée d'un caractère. La perte est calculée sur
    toutes les positions.
    """
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for _ in range(steps_per_epoch):
            starts = _batch_starts(len(train_tokens), context, batch_size, generator)
            inputs, targets = _windows_from_starts(train_tokens, starts, context)
            logits = model(inputs.to(device))
            vocab = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab), targets.to(device).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * targets.numel()
            seen += targets.numel()
        train_loss = running / seen
        val_loss = language_model_loss(model, val_tokens, context, device, batch_size=batch_size)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            marker = "  <- meilleur"
        print(f"Epoch {epoch + 1:3d} | train {train_loss:.4f} | val {val_loss:.4f}{marker}")
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Meilleur mini-GPT restauré, perte de validation {best_val:.4f}")
    return history


def train_masked_language_model(
    model: nn.Module,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    vocab_size: int,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Entraîne le mini-BERT. La perte ignore les positions non sélectionnées."""
    from src.text_models import accuracy_on_selected, corrupt_for_mlm

    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    generator = torch.Generator()
    generator.manual_seed(seed)
    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")

    def batches(ids: torch.Tensor):
        order = torch.randperm(len(ids), generator=generator)
        for offset in range(0, len(ids), batch_size):
            yield ids[order[offset : offset + batch_size]]

    @torch.no_grad()
    def evaluate(ids: torch.Tensor) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_count = 0
        correct = 0
        selected_count = 0
        for batch in batches(ids):
            corrupted, labels, _ = corrupt_for_mlm(batch, vocab_size, generator)
            pad_mask = corrupted.eq(0)
            logits = model(corrupted.to(device), pad_mask.to(device))
            loss = criterion(logits.reshape(-1, vocab_size), labels.to(device).reshape(-1))
            active = labels.ne(-100)
            number = int(active.sum().item())
            total_loss += float(loss.item()) * max(number, 1)
            total_count += max(number, 1)
            batch_correct, batch_total = accuracy_on_selected(logits.cpu(), labels)
            correct += batch_correct
            selected_count += batch_total
        accuracy = correct / selected_count if selected_count else 0.0
        return total_loss / total_count, accuracy

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in batches(train_ids):
            corrupted, labels, _ = corrupt_for_mlm(batch, vocab_size, generator)
            pad_mask = corrupted.eq(0)
            logits = model(corrupted.to(device), pad_mask.to(device))
            loss = criterion(logits.reshape(-1, vocab_size), labels.to(device).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            number = int(labels.ne(-100).sum().item())
            running += float(loss.item()) * max(number, 1)
            seen += max(number, 1)
        train_loss = running / seen
        val_loss, val_acc = evaluate(val_ids)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }
        )
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            marker = "  <- meilleur"
        print(
            f"Epoch {epoch + 1:3d} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | exactitude val {val_acc:.2%}{marker}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Meilleur mini-BERT restauré, perte de validation {best_val:.4f}")
    return history


@torch.no_grad()
def evaluate_masked_language_model(
    model: nn.Module,
    token_ids: torch.Tensor,
    vocab_size: int,
    batch_size: int = 32,
    device: torch.device | None = None,
    seed: int = 123,
) -> dict[str, float]:
    """Mesure la perte et l'exactitude du mini-BERT déjà entraîné."""
    from src.text_models import accuracy_on_selected, corrupt_for_mlm

    device = device or next(model.parameters()).device
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    total_count = 0
    correct = 0
    selected_count = 0
    for offset in range(0, len(token_ids), batch_size):
        batch = token_ids[offset : offset + batch_size]
        corrupted, labels, _ = corrupt_for_mlm(batch, vocab_size, generator)
        pad_mask = corrupted.eq(0)
        logits = model(corrupted.to(device), pad_mask.to(device))
        loss = criterion(logits.reshape(-1, vocab_size), labels.to(device).reshape(-1))
        number = int(labels.ne(-100).sum().item())
        total_loss += float(loss.item()) * max(number, 1)
        total_count += max(number, 1)
        batch_correct, batch_total = accuracy_on_selected(logits.cpu(), labels)
        correct += batch_correct
        selected_count += batch_total
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": correct / selected_count if selected_count else 0.0,
        "selected": float(selected_count),
    }
