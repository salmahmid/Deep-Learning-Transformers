"""Métriques, comparaisons et graphiques."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def as_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convertit un tenseur ou un tableau en vecteur NumPy."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def regression_scores(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray) -> dict[str, float]:
    """Calcule MSE, MAE et RMSE sur les valeurs fournies."""
    truth = as_numpy(y_true)
    prediction = as_numpy(y_pred)
    mse = float(mean_squared_error(truth, prediction))
    mae = float(mean_absolute_error(truth, prediction))
    rmse = float(np.sqrt(mse))
    return {"mse": mse, "mae": mae, "rmse": rmse}


def denormalize(values: torch.Tensor | np.ndarray, mean: float, std: float) -> np.ndarray:
    """Repasse des valeurs normalisées vers les kilowatts d'origine."""
    return as_numpy(values) * std + mean


def baseline_predictions(features: torch.Tensor, mean: float, std: float) -> dict[str, np.ndarray]:
    """Persistance (dernière heure) et référence de la veille (24 heures avant).

    ``features`` a la forme ``(N, 24, 1)`` et contient des valeurs normalisées.
    L'indice 0 est l'heure ``t-24`` et l'indice -1 est l'heure ``t-1``.
    """
    normalized = features.detach().cpu().numpy()
    persistence = denormalize(normalized[:, -1, 0], mean, std)
    daily = denormalize(normalized[:, 0, 0], mean, std)
    return {"persistance": persistence, "veille": daily}


def save_json(path: str | Path, payload: dict) -> Path:
    """Enregistre un dictionnaire JSON lisible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_figure(figure: plt.Figure, path: str | Path) -> Path:
    """Enregistre une figure sans couper les légendes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    return path


def plot_series(x_values: np.ndarray, y_values: np.ndarray, title: str, xlabel: str, ylabel: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(x_values, y_values, color="#1f4e79", linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    xlabel: str = "Indice de l'échantillon de test",
    ylabel: str = "Valeur",
    pred_label: str = "Prédictions",
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(as_numpy(y_true), label="Valeurs réelles", color="#1f4e79", linewidth=2)
    axis.plot(as_numpy(y_pred), label=pred_label, color="#b22222", linestyle="--", linewidth=2)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_loss_curves(curves: dict[str, list[float]], title: str, ylabel: str = "MSE d'entraînement") -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12, 5))
    for name, losses in curves.items():
        axis.plot(np.arange(1, len(losses) + 1), losses, label=name, linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("Époque")
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_metric_bars(table: dict[str, dict[str, float]], title: str) -> plt.Figure:
    """Diagramme en barres groupées pour MAE et RMSE."""
    names = list(table)
    mae = [table[name]["mae"] for name in names]
    rmse = [table[name]["rmse"] for name in names]
    positions = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions - width / 2, mae, width=width, label="MAE", color="#1f4e79")
    axis.bar(positions + width / 2, rmse, width=width, label="RMSE", color="#d4760a")
    axis.set_xticks(positions, names)
    axis.set_ylabel("Erreur")
    axis.set_title(title)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def longest_consecutive(positions: np.ndarray) -> tuple[int, int]:
    """Retourne le début et la longueur de la plus longue plage horaire continue."""
    positions = np.asarray(positions)
    if len(positions) == 0:
        return 0, 0
    best_start = 0
    best_length = 1
    start = 0
    for index in range(1, len(positions)):
        if positions[index] != positions[index - 1] + 1:
            length = index - start
            if length > best_length:
                best_start, best_length = start, length
            start = index
    tail = len(positions) - start
    if tail > best_length:
        best_start, best_length = start, tail
    return best_start, best_length


def consecutive_run(positions: np.ndarray, length: int) -> int | None:
    """Indice de début de la première plage d'indices horaires consécutifs."""
    positions = np.asarray(positions)
    if len(positions) < length:
        return None
    for start in range(len(positions) - length + 1):
        window = positions[start : start + length]
        if int(window[-1] - window[0]) == length - 1 and np.all(np.diff(window) == 1):
            return start
    return None


def plot_week(
    timestamps,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence: np.ndarray | None = None,
) -> plt.Figure:
    """Observations et prédictions sur une plage horaire continue."""
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(timestamps, y_true, label="Observations", color="#1f4e79", linewidth=2)
    axis.plot(timestamps, y_pred, label="EnergyTransformer", color="#b22222", linestyle="--", linewidth=2)
    if persistence is not None:
        axis.plot(timestamps, persistence, label="Persistance", color="#6b7280", linestyle=":", linewidth=1.5)
    axis.set_title("Consommation électrique : sept jours du jeu de test")
    axis.set_xlabel("Heure")
    axis.set_ylabel("Puissance moyenne (kW)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure
