"""Préparation des séries temporelles et des corpus textuels."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ZIP_NAME = "individual+household+electric+power+consumption.zip"
TEXT_NAME = "household_power_consumption.txt"
HOURLY_CACHE = PROCESSED_DIR / "hourly_global_active_power.csv"

# Phrases du mini-BERT. Le vocabulaire de validation est volontairement
# couvert par l'entraînement : un test automatique le vérifie.
TRAIN_SENTENCES: tuple[str, ...] = (
    "le chat dort sur le canapé",
    "le chien dort sur le tapis",
    "le chat regarde la porte",
    "le chien regarde le jardin",
    "paris est la capitale de la france",
    "lyon est une ville de la france",
    "le modèle prédit la valeur suivante",
    "le réseau lit la séquence",
    "le cours explique le transformeur",
    "le professeur présente le principe du transformeur",
    "un élève écoute le cours",
    "la famille allume la lampe",
    "la lampe éclaire le salon",
    "le four chauffe la cuisine",
    "le réfrigérateur fonctionne pendant la nuit",
    "la consommation augmente le soir",
    "la consommation diminue le matin",
    "la puissance indique la consommation",
    "la maison contient une cuisine",
    "le salon contient une lampe",
    "le transformeur calcule le score",
    "le transformeur utilise le contexte",
    "le masque cache le mot",
    "la phrase contient un mot",
    "le caractère occupe une position",
    "la couche contient quatre têtes",
    "la mémoire conserve un état",
    "la fenêtre contient vingt quatre heures",
    "le livre contient un texte",
    "la famille mange dans la cuisine",
    "un enfant lit un livre",
    "le modèle utilise la mémoire",
    "le réseau utilise quatre têtes",
    "le cours présente la couche",
    "paris reste la capitale",
    "le chat reste sur le canapé",
    "le chien reste sur le tapis",
    "la lampe reste dans le salon",
    "le four reste dans la cuisine",
    "la consommation reste faible le matin",
    "le soir la famille allume la lampe",
    "le matin la consommation diminue",
    "le transformeur prédit la puissance",
    "le modèle lit vingt quatre heures",
    "un élève comprend le masque",
    "le professeur explique le contexte",
    "la phrase commence par un mot",
    "le jardin contient une porte",
)

VAL_SENTENCES: tuple[str, ...] = (
    "le chat dort sur le tapis",
    "le chien dort sur le canapé",
    "le chat regarde le jardin",
    "lyon est la ville",
    "le modèle lit la séquence",
    "le réseau prédit la valeur suivante",
    "un élève comprend le contexte",
    "la lampe éclaire la cuisine",
    "la consommation augmente le matin",
    "la famille allume le four",
    "le transformeur calcule le contexte",
    "la mémoire conserve une séquence",
)


def make_sine_dataset(
    n_points: int = 200,
    x_min: float = 0.0,
    x_max: float = 20.0,
    length: int = 10,
    n_train: int = 150,
) -> dict[str, torch.Tensor]:
    """Crée la sinusoïde du TP et les fenêtres glissantes.

    ``y = sin(x)`` pour 200 valeurs de x entre 0 et 20. Chaque entrée
    contient 10 valeurs et la cible est la 11e. Les 150 premières fenêtres
    forment l'entraînement, les 40 suivantes le test.
    """
    periode = torch.linspace(x_min, x_max, n_points)
    serie = torch.sin(periode)
    windows = len(serie) - length
    if n_train > windows:
        raise ValueError("Le nombre de séquences d'entraînement dépasse les fenêtres disponibles.")
    features = torch.stack([serie[i : i + length] for i in range(windows)])
    targets = torch.stack([serie[i + length] for i in range(windows)])
    features = features.unsqueeze(-1)
    targets = targets.unsqueeze(-1)
    return {
        "periode": periode,
        "serie": serie,
        "X": features,
        "y": targets,
        "X_train": features[:n_train],
        "y_train": targets[:n_train],
        "X_test": features[n_train:],
        "y_test": targets[n_train:],
    }


def _read_power_frame(source, nrows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(
        source,
        sep=";",
        na_values=["?"],
        low_memory=False,
        nrows=nrows,
    )
    frame["datetime"] = pd.to_datetime(
        frame["Date"] + " " + frame["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    frame["Global_active_power"] = pd.to_numeric(frame["Global_active_power"], errors="coerce")
    return frame


def hourly_mean_power(frame: pd.DataFrame, min_count: int = 45) -> pd.Series:
    """Moyenne horaire de Global_active_power, en kW.

    Une heure est conservée seulement si elle contient au moins
    ``min_count`` mesures numériques. Les autres heures deviennent NaN.
    ``count`` ignore déjà les valeurs manquantes.
    """
    power = frame["Global_active_power"].copy()
    power.index = pd.DatetimeIndex(frame["datetime"])
    hourly = power.resample("h").agg(["mean", "count"])
    serie = hourly["mean"].where(hourly["count"] >= min_count)
    serie.name = "global_active_power_kw"
    return serie


def ensure_power_text(zip_path: Path | None = None, text_path: Path | None = None) -> Path:
    """Retrouve le fichier texte, en le décompressant si nécessaire."""
    text_path = text_path or (RAW_DIR / TEXT_NAME)
    zip_path = zip_path or (RAW_DIR / ZIP_NAME)
    if text_path.exists():
        return text_path
    if not zip_path.exists():
        raise FileNotFoundError(
            "Fichier de consommation introuvable. Placez "
            f"{ZIP_NAME} dans data/raw/."
        )
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(TEXT_NAME, path=text_path.parent)
    if not text_path.exists():
        raise FileNotFoundError(f"L'archive ne contient pas {TEXT_NAME}.")
    return text_path


def load_hourly_power(use_cache: bool = True, nrows: int | None = None) -> pd.Series:
    """Charge la série horaire réelle. Le cache évite de relire 127 Mo."""
    if use_cache and nrows is None and HOURLY_CACHE.exists():
        loaded = pd.read_csv(HOURLY_CACHE, index_col=0, parse_dates=True)
        serie = loaded.iloc[:, 0] if isinstance(loaded, pd.DataFrame) else loaded
        serie.index = pd.DatetimeIndex(serie.index)
        serie.name = "global_active_power_kw"
        return serie
    text_path = ensure_power_text()
    frame = _read_power_frame(text_path, nrows=nrows)
    serie = hourly_mean_power(frame)
    if use_cache and nrows is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        serie.to_csv(HOURLY_CACHE, header=True)
    return serie


def window_split(
    serie: pd.Series,
    window: int = 24,
    train_ratio: float = 0.70,
    val_end_ratio: float = 0.85,
) -> dict:
    """Découpe chronologiquement, normalise sur le train, puis forme les fenêtres.

    La moyenne et l'écart-type sont calculés uniquement sur
    ``serie.iloc[:fin_train]``. Une fenêtre est rejetée dès qu'elle contient
    un NaN. Le lot d'une fenêtre est décidé par l'indice de sa cible.
    """
    values = serie.astype("float64")
    n = len(values)
    fin_train = int(train_ratio * n)
    fin_val = int(val_end_ratio * n)
    if not (0 < fin_train < fin_val < n):
        raise ValueError("Le découpage chronologique ne produit pas trois parties non vides.")
    train_slice = values.iloc[:fin_train]
    mean = float(train_slice.mean())
    std = float(train_slice.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std == 0.0:
        raise ValueError("La normalisation est impossible : moyenne ou écart-type invalide.")
    normalized = ((values - mean) / std).to_numpy(dtype=np.float64)
    features: list[np.ndarray] = []
    targets: list[float] = []
    target_positions: list[int] = []
    for index in range(window, n):
        sequence = normalized[index - window : index]
        target = normalized[index]
        if np.isfinite(sequence).all() and np.isfinite(target):
            features.append(sequence)
            targets.append(float(target))
            target_positions.append(index)
    if not features:
        raise ValueError("Aucune fenêtre complète après suppression des valeurs manquantes.")
    x = torch.tensor(np.asarray(features), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32).unsqueeze(-1)
    positions = np.asarray(target_positions)
    train_mask = positions < fin_train
    val_mask = (positions >= fin_train) & (positions < fin_val)
    test_mask = positions >= fin_val

    def take(tensor: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        return tensor[torch.from_numpy(np.asarray(mask, dtype=bool))]

    return {
        "X": x,
        "y": y,
        "positions": positions,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "X_train": take(x, train_mask),
        "y_train": take(y, train_mask),
        "X_val": take(x, val_mask),
        "y_val": take(y, val_mask),
        "X_test": take(x, test_mask),
        "y_test": take(y, test_mask),
        "positions_test": positions[test_mask],
        "mean": mean,
        "std": std,
        "n": n,
        "fin_train": fin_train,
        "fin_val": fin_val,
        "index": values.index,
        "window": window,
    }


def load_char_corpus(path: str | Path | None = None) -> tuple[str, dict[str, int], dict[int, str], torch.Tensor]:
    """Lit corpus.txt et construit la tokenisation caractère du TP."""
    path = Path(path) if path is not None else ROOT / "corpus.txt"
    text = path.read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("Le corpus est trop court pour un entraînement caractère.")
    characters = sorted(set(text))
    stoi = {character: index for index, character in enumerate(characters)}
    itos = {index: character for character, index in stoi.items()}
    tokens = torch.tensor([stoi[character] for character in text], dtype=torch.long)
    return text, stoi, itos, tokens


def split_token_stream(tokens: torch.Tensor, train_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """Sépare le flux de tokens sans mélanger le futur avec le passé."""
    cut = int(len(tokens) * train_ratio)
    if cut < 2 or cut >= len(tokens):
        raise ValueError("Découpage du corpus impossible.")
    return tokens[:cut].contiguous(), tokens[cut:].contiguous()


def tokenize_words(sentence: str) -> list[str]:
    """Découpe une phrase en mots, sans ponctuation."""
    cleaned = sentence.lower()
    for character in ",.;:!?()[]«»\"'":
        cleaned = cleaned.replace(character, " ")
    return [word for word in cleaned.split() if word]


def build_word_vocab(sentences: list[str] | tuple[str, ...]) -> tuple[dict[str, int], dict[int, str]]:
    """Vocabulaire du mini-BERT. Les identifiants 0 à 4 sont réservés."""
    stoi = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for sentence in sentences:
        for word in tokenize_words(sentence):
            if word not in stoi:
                stoi[word] = len(stoi)
    itos = {index: token for token, index in stoi.items()}
    return stoi, itos


def encode_sentences(
    sentences: list[str] | tuple[str, ...],
    stoi: dict[str, int],
    max_length: int = 32,
) -> torch.Tensor:
    """Encode ``[CLS] mots [SEP] [PAD]...`` jusqu'à ``max_length`` tokens."""
    if max_length < 3:
        raise ValueError("La longueur maximale doit contenir CLS, un mot et SEP.")
    rows: list[list[int]] = []
    unk = stoi["[UNK]"]
    for sentence in sentences:
        words = tokenize_words(sentence)[: max_length - 2]
        ids = [stoi["[CLS]"]] + [stoi.get(word, unk) for word in words] + [stoi["[SEP]"]]
        ids.extend([stoi["[PAD]"]] * (max_length - len(ids)))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def build_minibert_corpus(copies_train: int = 12, copies_val: int = 4, seed: int = 42) -> tuple[list[str], list[str]]:
    """Répète un petit corpus didactique, puis le mélange de façon reproductible."""
    import random

    train = list(TRAIN_SENTENCES) * copies_train
    validation = list(VAL_SENTENCES) * copies_val
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    return train, validation
