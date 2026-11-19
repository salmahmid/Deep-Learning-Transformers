"""Modèles de séries temporelles : RNN, LSTM et Transformers encodeurs.

Les noms de classes et les dimensions suivent le code du TP :
SimpleRNN, SimpleLSTM, MiniTransformer et EnergyTransformer.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def transformer_encoder_layer(**kwargs) -> nn.TransformerEncoderLayer:
    """Crée une couche d'encodeur.

    Le calcul reste celui de ``TransformerEncoderLayer``. Le chemin
    « nested tensor » est désactivé plus haut, sur l'encodeur complet.
    """
    if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoderLayer).parameters:
        kwargs.setdefault("enable_nested_tensor", False)
    return nn.TransformerEncoderLayer(**kwargs)


def transformer_encoder(layer: nn.TransformerEncoderLayer, num_layers: int) -> nn.TransformerEncoder:
    """Empile des couches d'encodeur sans le prototype nested tensor."""
    kwargs = {}
    if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
        kwargs["enable_nested_tensor"] = False
    return nn.TransformerEncoder(layer, num_layers=num_layers, **kwargs)


class SimpleRNN(nn.Module):
    """RNN à une couche : 10 valeurs de sinus, puis une prédiction.

    Dimensions, avec ``batch_first=True`` :

    - entrée : ``(B, 10, 1)``
    - états du RNN : ``(B, 10, 16)``
    - dernier état : ``(B, 16)``
    - sortie : ``(B, 1)``
    """

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=16,
            batch_first=True,
        )
        self.sortie = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sans état initial, PyTorch commence avec un état nul.
        etats, _ = self.rnn(x)
        dernier_etat = etats[:, -1, :]
        return self.sortie(dernier_etat)


class SimpleLSTM(nn.Module):
    """LSTM à une couche. La prédiction utilise le dernier état caché.

    La cellule transmet aussi un état de mémoire ``c_t``, mais la couche
    linéaire ne lit que ``h_t`` au dernier instant, comme dans le TP.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
        )
        self.sortie = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        etats, _ = self.lstm(x)
        dernier_etat = etats[:, -1, :]
        return self.sortie(dernier_etat)


class MiniTransformer(nn.Module):
    """Encodeur Transformer minimal pour la sinusoïde.

    Projection ``1 -> 16``, positions apprises, une couche d'encodeur,
    puis la représentation de la dernière position projetée vers une valeur.
    Aucun masque causal : les 10 entrées sont toutes connues.
    """

    def __init__(self, longueur_max: int = 10) -> None:
        super().__init__()
        dimension = 16
        self.projection = nn.Linear(1, dimension)
        self.positions = nn.Embedding(longueur_max, dimension)
        couche = transformer_encoder_layer(
            d_model=dimension,
            nhead=2,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )
        self.encodeur = transformer_encoder(couche, num_layers=1)
        self.sortie = nn.Linear(dimension, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        longueur = x.size(1)
        x = self.projection(x)
        indices = torch.arange(longueur, device=x.device)
        x = x + self.positions(indices).unsqueeze(0)
        representations = self.encodeur(x)
        return self.sortie(representations[:, -1, :])


class EnergyTransformer(nn.Module):
    """Transformer encodeur pour la puissance moyenne de l'heure suivante.

    - projection ``Linear(1, 32)``
    - embedding positionnel des 24 heures
    - deux ``TransformerEncoderLayer``
    - ``d_model=32``, ``nhead=4``, ``dim_feedforward=128``, ``dropout=0.1``
    - ``Linear(32, 1)`` sur la dernière heure
    """

    def __init__(self, longueur_max: int = 24) -> None:
        super().__init__()
        dimension = 32
        self.projection = nn.Linear(1, dimension)
        self.positions = nn.Embedding(longueur_max, dimension)
        couche = transformer_encoder_layer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encodeur = transformer_encoder(couche, num_layers=2)
        self.sortie = nn.Linear(dimension, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        longueur = x.size(1)
        x = self.projection(x)
        indices = torch.arange(longueur, device=x.device)
        x = x + self.positions(indices).unsqueeze(0)
        representations = self.encodeur(x)
        return self.sortie(representations[:, -1, :])
