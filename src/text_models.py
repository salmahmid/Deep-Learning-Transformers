"""Mini-GPT causal et mini-BERT bidirectionnel, entraînés depuis zéro."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models import transformer_encoder, transformer_encoder_layer

PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4


def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """Masque booléen demandé par le TP.

    ``True`` signifie que la relation d'attention est interdite.
    Une position peut lire le passé et elle-même, pas le futur.
    """
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class GPTBlock(nn.Module):
    """Bloc pré-normalisé : attention causale, résidu, MLP, résidu."""

    def __init__(self, d_model: int = 64, n_head: int = 4, d_ff: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.ln1(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attended
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Petit décodeur caractère, sans poids préentraînés.

    - identifiants : ``(batch, 64)``
    - représentations : ``(batch, 64, 64)``
    - logits : ``(batch, 64, vocabulaire)``
    """

    def __init__(
        self,
        vocab_size: int,
        context: int = 64,
        d_model: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        d_ff: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model=d_model, n_head=n_head, d_ff=d_ff, dropout=dropout) for _ in range(n_layer)]
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        if length > self.context:
            raise ValueError(f"La séquence ({length}) dépasse le contexte ({self.context}).")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions).unsqueeze(0)
        mask = causal_mask(length, token_ids.device)
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.output(hidden)


class MiniBERT(nn.Module):
    """Encodeur bidirectionnel pour des mots masqués.

    Le réseau feed-forward interne fait ``64 -> 256 -> 64``. Le TP ne fixe
    pas cette largeur : 256 reprend le facteur quatre utilisé pour le mini-GPT
    et reste adapté à un petit modèle. Le dropout vaut 0,1.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int = 32,
        d_model: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_length, d_model)
        layer = transformer_encoder_layer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = transformer_encoder(layer, num_layers=n_layer)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        _, length = token_ids.shape
        if length > self.max_length:
            raise ValueError(f"La séquence ({length}) dépasse la longueur maximale ({self.max_length}).")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions).unsqueeze(0)
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        return self.output(hidden)


def _rand(shape: tuple[int, ...], generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.rand(shape)
    return torch.rand(shape, generator=generator)


def _randint(low: int, high: int, shape: tuple[int, ...], generator: torch.Generator | None) -> torch.Tensor:
    if generator is None:
        return torch.randint(low, high, shape)
    return torch.randint(low, high, shape, generator=generator)


def corrupt_for_mlm(
    input_ids: torch.Tensor,
    vocab_size: int,
    generator: torch.Generator | None = None,
    selection_probability: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrompt 15 % des tokens ordinaires : 80 % MASK, 10 % aléatoire, 10 % inchangé.

    Les cibles valent ``-100`` partout sauf aux positions sélectionnées, où
    elles conservent l'identifiant d'origine. Toute phrase qui contient un
    token ordinaire garde au moins une position à prédire.
    """
    if vocab_size <= MASK + 1:
        raise ValueError("Le vocabulaire ne contient aucun token ordinaire.")
    ordinary = input_ids >= (MASK + 1)
    selected = ordinary & (_rand(input_ids.shape, generator) < selection_probability)
    for row in range(input_ids.size(0)):
        if bool(ordinary[row].any()) and not bool(selected[row].any()):
            candidates = ordinary[row].nonzero(as_tuple=False).view(-1)
            pick = int(_randint(0, candidates.numel(), (1,), generator).item())
            selected[row, candidates[pick]] = True

    chance = _rand(input_ids.shape, generator)
    masked = selected & (chance < 0.8)
    replaced = selected & (chance >= 0.8) & (chance < 0.9)
    random_tokens = _randint(MASK + 1, vocab_size, input_ids.shape, generator)
    corrupted = input_ids.clone()
    corrupted[masked] = MASK
    corrupted[replaced] = random_tokens[replaced]
    labels = input_ids.clone()
    labels[~selected] = -100
    return corrupted, labels, selected


def accuracy_on_selected(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Retourne le nombre de bonnes prédictions et le nombre de positions évaluées."""
    predictions = logits.argmax(dim=-1)
    active = labels.ne(-100)
    total = int(active.sum().item())
    if total == 0:
        return 0, 0
    correct = int((predictions.eq(labels) & active).sum().item())
    return correct, total


@torch.no_grad()
def generate_characters(
    model: MiniGPT,
    prompt_ids: torch.Tensor,
    n_new: int,
    temperature: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Ajoute ``n_new`` caractères en échantillonnant le suivant."""
    if temperature <= 0:
        raise ValueError("La température doit être strictement positive.")
    model.eval()
    sequence = prompt_ids.to(device)
    if sequence.dim() == 1:
        sequence = sequence.unsqueeze(0)
    for _ in range(n_new):
        window = sequence[:, -model.context :]
        logits = model(window)[:, -1, :] / temperature
        # Le tirage se fait sur CPU : le générateur aléatoire y est portable.
        probabilities = torch.softmax(logits, dim=-1).cpu()
        if generator is None:
            next_id = torch.multinomial(probabilities, num_samples=1)
        else:
            next_id = torch.multinomial(probabilities, num_samples=1, generator=generator)
        sequence = torch.cat([sequence, next_id.to(device)], dim=1)
    return sequence


def decode_ids(token_ids: torch.Tensor, itos: dict[int, str]) -> str:
    """Reconstitue une chaîne à partir d'identifiants de caractères."""
    values = token_ids.view(-1).tolist()
    return "".join(itos[int(value)] for value in values)


def topk_words(
    logits: torch.Tensor,
    position: int,
    itos: dict[int, str],
    k: int = 3,
) -> list[tuple[str, float]]:
    """Retourne les k tokens les plus probables à une position."""
    probabilities = torch.softmax(logits[position], dim=-1)
    scores, indices = torch.topk(probabilities, k=k)
    return [(itos[int(index)], float(score)) for score, index in zip(scores, indices)]
