"""Construit les cinq notebooks du projet. Exécuter depuis la racine."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def notebook(name: str, cells: list) -> None:
    book = nbf.v4.new_notebook()
    book["cells"] = cells
    book["metadata"] = {
        "kernelspec": {"display_name": "Python DLT", "language": "python", "name": "dlt"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path = OUT / name
    path.write_text(nbf.writes(book), encoding="utf-8")
    print("écrit", path)


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str):
    return nbf.v4.new_code_cell(source.strip("\n"))


SETUP = '''
import sys
from pathlib import Path

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import count_parameters, get_device, set_seed

set_seed(42)
device = get_device()
print("Dossier du projet :", ROOT)
print("Périphérique :", device)
'''


def build_sine() -> None:
    cells = [
        md("""
# Sinusoïde : RNN, LSTM et Transformer

Ce notebook reprend la première partie du TP « Architecture Transformer ».
On apprend à prédire la valeur suivante d'une sinusoïde avec trois modèles,
puis on compare leurs erreurs **calculées** sur le même jeu de test.

La sinusoïde est un problème volontairement simple. Elle sert à voir les
dimensions et le rôle de chaque bloc avant d'aborder l'électricité et le texte.
        """),
        md("""
## Comment lire ce notebook

Chaque modèle reçoit **10 valeurs** et doit annoncer **la 11e**.
Les poids sont initialisés avec la graine 42, puis réinitialisés avant chaque
architecture pour que les trois entraînements partent du même hasard.

Le code réutilisable est dans `src/`. Les cellules ci-dessous montrent les
appels et les résultats. Les classes exactes s'appellent `SimpleRNN`,
`SimpleLSTM` et `MiniTransformer`, comme dans le code du professeur.
        """),
        code(SETUP),
        md("""
## 1. La série `y = sin(x)`

`torch.linspace(0, 20, 200)` place 200 abscisses régulières entre 0 et 20.
Le sinus de ces abscisses donne la série. Une fenêtre glissante de longueur 10
produit 190 exemples, car `200 - 10 = 190`.

| Exemple | Entrée | Cible |
|---|---|---|
| 0 | `s0 ... s9` | `s10` |
| 1 | `s1 ... s10` | `s11` |
| 149 | dernières valeurs d'entraînement | valeur suivante |
| 150 à 189 | 40 fenêtres de test | valeurs jamais utilisées pour régler les poids |

Avec `unsqueeze(-1)`, une série devient un tenseur `(nombre, longueur, 1)` :
une seule variable est observée à chaque instant.
        """),
        code("""
from src.data import make_sine_dataset

data = make_sine_dataset()
periode, serie = data["periode"], data["serie"]
X, y = data["X"], data["y"]
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

print("période :", tuple(periode.shape))
print("série   :", tuple(serie.shape))
print("X       :", tuple(X.shape), "  y :", tuple(y.shape))
print("train   :", tuple(X_train.shape), tuple(y_train.shape))
print("test    :", tuple(X_test.shape), tuple(y_test.shape))
print()
for index in (0, 1, 2):
    entree = X[index, :, 0].numpy()
    print(f"exemple {index} | entrée {np.array2string(entree, precision=3)} | cible {y[index].item(): .4f}")
        """),
        code("""
from src.evaluation import plot_series, save_figure

figure = plot_series(
    periode.numpy(),
    serie.numpy(),
    title="Série y = sin(x), 200 points entre 0 et 20",
    xlabel="x",
    ylabel="y",
)
save_figure(figure, ROOT / "outputs" / "figures" / "01_sinusoide.png")
plt.show()
        """),
        md("""
## 2. RNN

Un RNN lit les 10 valeurs **dans l'ordre**. Il met à jour un état caché :

$$
h_t = \\tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)
$$

Ici `input_size=1` et `hidden_size=16`. La couche `Linear(16, 1)` ne voit que
le dernier état. Les dimensions sont donc :

$$
(B, 10, 1) \\rightarrow (B, 10, 16) \\rightarrow (B, 16) \\rightarrow (B, 1)
$$

On entraîne 300 époques avec Adam, un pas de 0,01 et la MSE. Tout le lot
d'entraînement passe à chaque époque : il ne contient que 150 séquences.
        """),
        code("""
from src.models import SimpleRNN
from src.training import predict, train_full_batch

set_seed(42)
rnn = SimpleRNN()
print("Paramètres du RNN :", count_parameters(rnn))
losses_rnn = train_full_batch(rnn, X_train, y_train, epochs=300, lr=0.01, device=device)
pred_rnn = predict(rnn, X_test)
erreur_rnn = torch.nn.functional.mse_loss(pred_rnn, y_test)
print(f"\\nErreur de test : {erreur_rnn.item():.6f}")
print(f"Valeur réelle  : {y_test[0].item():.4f}")
print(f"Valeur prédite : {pred_rnn[0].item():.4f}")
        """),
        code("""
from src.evaluation import plot_forecast

figure = plot_forecast(
    y_test, pred_rnn,
    title="RNN : valeurs réelles et prédictions",
    pred_label="Prédictions du RNN",
)
save_figure(figure, ROOT / "outputs" / "figures" / "01_rnn_predictions.png")
plt.show()
        """),
        md("""
## 3. LSTM

Le LSTM garde deux états. L'état caché `h_t` est la sortie visible.
L'état de cellule `c_t` est une mémoire interne contrôlée par trois portes.

$$
c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t
$$

$$
h_t = o_t \\odot \\tanh(c_t)
$$

La configuration du TP est `input_size=1`, `hidden_size=16`, `num_layers=1`,
`batch_first=True`, puis `Linear(16, 1)`. L'entraînement est le même que pour le RNN.
        """),
        code("""
from src.models import SimpleLSTM

set_seed(42)
lstm = SimpleLSTM()
print("Paramètres du LSTM :", count_parameters(lstm))
losses_lstm = train_full_batch(lstm, X_train, y_train, epochs=300, lr=0.01, device=device)
pred_lstm = predict(lstm, X_test)
erreur_lstm = torch.nn.functional.mse_loss(pred_lstm, y_test)
print(f"\\nErreur de test : {erreur_lstm.item():.6f}")
print(f"Valeur réelle  : {y_test[0].item():.4f}")
print(f"Valeur prédite : {pred_lstm[0].item():.4f}")
        """),
        code("""
figure = plot_forecast(
    y_test, pred_lstm,
    title="LSTM : valeurs réelles et prédictions",
    pred_label="Prédictions du LSTM",
)
save_figure(figure, ROOT / "outputs" / "figures" / "01_lstm_predictions.png")
plt.show()
        """),
        md("""
## 4. Mini-Transformer

Ce modèle ne relit pas la séquence pas à pas. Il projette chaque valeur de 1 vers 16,
ajoute une position apprise, puis applique **une** couche d'encodeur :

- `d_model=16`
- `nhead=2` (donc 8 composantes par tête)
- `dim_feedforward=32`
- `dropout=0`
- activation ReLU, qui est le réglage par défaut de `TransformerEncoderLayer`

La prédiction part de la dernière position. Aucun masque causal n'est posé :
les 10 entrées sont connues, et la 11e n'est jamais donnée au modèle.
        """),
        code("""
from src.models import MiniTransformer

set_seed(42)
transformer = MiniTransformer(longueur_max=10)
print("Paramètres du Transformer :", count_parameters(transformer))
losses_transformer = train_full_batch(
    transformer, X_train, y_train, epochs=300, lr=0.01, device=device
)
pred_transformer = predict(transformer, X_test)
erreur_transformer = torch.nn.functional.mse_loss(pred_transformer, y_test)
print(f"\\nErreur de test : {erreur_transformer.item():.6f}")
print(f"Valeur réelle  : {y_test[0].item():.4f}")
print(f"Valeur prédite : {pred_transformer[0].item():.4f}")
        """),
        code("""
figure = plot_forecast(
    y_test, pred_transformer,
    title="Transformer : valeurs réelles et prédictions",
    pred_label="Prédictions du Transformer",
)
save_figure(figure, ROOT / "outputs" / "figures" / "01_transformer_predictions.png")
plt.show()
        """),
        md("""
## 5. Comparaison sur le test

La MSE moyenne les carrés des erreurs. La MAE moyenne les valeurs absolues.
La RMSE est la racine de la MSE : elle s'exprime dans la même unité que `y`,
ici sans unité, puisque le sinus est déjà entre -1 et 1.

Ces trois nombres sont recalculés par scikit-learn sur les 40 prédictions de test.
Ils ne sont pas recopiés à la main.
        """),
        code("""
from src.evaluation import plot_loss_curves, plot_metric_bars, regression_scores, save_json

resultats = {
    "RNN": regression_scores(y_test, pred_rnn),
    "LSTM": regression_scores(y_test, pred_lstm),
    "Transformer": regression_scores(y_test, pred_transformer),
}
tableau = pd.DataFrame(resultats).T
tableau.index.name = "modèle"
print(tableau.to_string(float_format=lambda valeur: f"{valeur:.6f}"))
save_json(ROOT / "outputs" / "metrics" / "01_sinusoide.json", {
    "device": str(device),
    "metriques": resultats,
    "perte_finale_train": {
        "RNN": losses_rnn[-1],
        "LSTM": losses_lstm[-1],
        "Transformer": losses_transformer[-1],
    },
})

figure = plot_metric_bars(resultats, "Erreurs de test sur la sinusoïde")
save_figure(figure, ROOT / "outputs" / "figures" / "01_comparaison_barres.png")
plt.show()

figure = plot_loss_curves(
    {"RNN": losses_rnn, "LSTM": losses_lstm, "Transformer": losses_transformer},
    "Perte d'entraînement sur la sinusoïde",
)
save_figure(figure, ROOT / "outputs" / "figures" / "01_pertes.png")
plt.show()

figure = plot_forecast(y_test, pred_rnn, title="Les trois modèles sur le test", pred_label="RNN")
axis = figure.axes[0]
axis.plot(pred_lstm.numpy().reshape(-1), label="LSTM", linewidth=1.6)
axis.plot(pred_transformer.numpy().reshape(-1), label="Transformer", linewidth=1.6)
axis.legend()
save_figure(figure, ROOT / "outputs" / "figures" / "01_comparaison_courbes.png")
plt.show()

meilleur = min(resultats, key=lambda nom: resultats[nom]["rmse"])
print(f"RMSE la plus basse sur ce test : {meilleur} ({resultats[meilleur]['rmse']:.6f}).")
print("Cette série est courte et très régulière : une bonne RMSE n'annonce pas")
print("le même classement sur la consommation électrique.")
        """),
        md("""
## Ce qu'il faut retenir pour l'oral

- Le RNN résume la fenêtre dans un état qui avance dans le temps.
- Le LSTM ajoute une mémoire de cellule et des portes, donc plus de paramètres.
- Le Transformer relie les positions par l'attention et a besoin d'une information de position.
- Sur cette sinusoïde, les trois courbes de test doivent être lues avec le tableau,
  pas avec une impression visuelle seule.
- Le test est chronologique : ce sont les 40 dernières fenêtres, pas un tirage aléatoire.
        """),
    ]
    notebook("01_sinusoide_rnn_lstm_transformer.ipynb", cells)


def build_pretrained() -> None:
    cells = [
        md("""
# GPT-2 et BERT préentraînés

Ces deux modèles viennent de Hugging Face. **On ne les entraîne pas.**
Leurs poids ont déjà été appris sur de très grands corpus. Ici, on les interroge
seulement pour voir deux usages du Transformer.

| | GPT-2 | BERT multilingue |
|---|---|---|
| Classe | décodeur seul | encodeur seul |
| Attention | causale : pas de regard vers le futur | bidirectionnelle |
| Tâche de cette démo | continuer un texte | retrouver un mot masqué |
| Modèle | `gpt2` | `bert-base-multilingual-cased` |

GPT-2 a surtout vu de l'anglais, d'où la phrase d'amorce en anglais du TP.
BERT `bert-base-multilingual-cased` couvre le français et distingue les majuscules.
        """),
        code(SETUP + """
import os
os.environ.setdefault("HF_HOME", str(ROOT / "outputs" / "huggingface"))
print("Cache Hugging Face :", os.environ["HF_HOME"])
        """),
        md("""
## 1. GPT-2, décodeur seul

Le texte de départ est exactement :

`Artificial intelligence can help people`

On demande 50 **nouveaux** tokens, avec un tirage aléatoire (`do_sample=True`),
une température de 0,8 et un `top_p` de 0,9. Le `top_p` garde le plus petit
ensemble de tokens dont la probabilité cumulée atteint au moins 90 %.

GPT-2 n'a pas de token de padding dédié. On réutilise le token de fin, comme
dans le notebook du cours. Avec une seule phrase, ce padding ne sert presque pas.
        """),
        code("""
import json
from pathlib import Path

statut = {"gpt2": None, "bert": None}
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    nom = "gpt2"
    tokenizer_gpt = AutoTokenizer.from_pretrained(nom)
    modele_gpt = AutoModelForCausalLM.from_pretrained(nom)
    tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
    modele_gpt.config.pad_token_id = tokenizer_gpt.pad_token_id
    modele_gpt.eval()

    texte = "Artificial intelligence can help people"
    entrees = tokenizer_gpt(texte, return_tensors="pt")
    print("Dimensions de l'entrée :", tuple(entrees["input_ids"].shape))
    set_seed(42)
    with torch.no_grad():
        sortie = modele_gpt.generate(
            **entrees,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer_gpt.pad_token_id,
        )
    texte_genere = tokenizer_gpt.decode(sortie[0], skip_special_tokens=True)
    print("\\nTexte généré :")
    print(texte_genere)
    statut["gpt2"] = {
        "ok": True,
        "forme_entree": list(entrees["input_ids"].shape),
        "texte": texte_genere,
    }
except Exception as erreur:
    statut["gpt2"] = {"ok": False, "erreur": f"{type(erreur).__name__}: {erreur}"}
    print("GPT-2 n'a pas pu être exécuté.")
    print(statut["gpt2"]["erreur"])
    print("Aucun texte n'est inventé à la place de cette génération.")
        """),
        md("""
### Pourquoi l'attention est causale

GPT-2 produit le token suivant à partir des tokens déjà écrits. Dans chaque bloc,
une position peut lire son passé et elle-même, pas l'avenir. C'est nécessaire :
au moment où l'on écrit un mot, la suite n'existe pas encore.

La génération est autorégressive. Le token choisi est ajouté à la séquence,
puis le modèle recalcule le suivant. Une température inférieure à 1 concentre
le tirage. Elle ne rend pas le texte vrai : elle le rend seulement moins dispersé.
        """),
        md("""
## 2. BERT, encodeur seul

La phrase demandée est :

`Paris est la [MASK] de la France.`

Le token de masque n'est pas tapé au hasard : on utilise `tokenizer.mask_token`,
afin d'obtenir l'identifiant que le modèle a vraiment appris. Le tokenizer ajoute
aussi `[CLS]` et `[SEP]`.

On lit les logits à la position du masque, on applique un softmax sur tout le
vocabulaire, puis on affiche les cinq tokens les plus probables.
        """),
        code("""
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    nom = "bert-base-multilingual-cased"
    tokenizer_bert = AutoTokenizer.from_pretrained(nom)
    modele_bert = AutoModelForMaskedLM.from_pretrained(nom)
    modele_bert.eval()

    phrase = f"Paris est la {tokenizer_bert.mask_token} de la France."
    entrees = tokenizer_bert(phrase, return_tensors="pt")
    print("Texte :", phrase)
    print("Token de masque :", tokenizer_bert.mask_token)
    print("Dimensions de l'entrée :", tuple(entrees["input_ids"].shape))
    position = (entrees["input_ids"][0] == tokenizer_bert.mask_token_id).nonzero(as_tuple=True)[0].item()
    with torch.no_grad():
        logits = modele_bert(**entrees).logits[0, position, :]
    probabilites = torch.softmax(logits, dim=-1)
    top_probabilites, top_indices = torch.topk(probabilites, k=5)
    propositions = []
    print("\\nPropositions de BERT :")
    for probabilite, indice in zip(top_probabilites, top_indices):
        token = tokenizer_bert.decode([indice.item()]).strip()
        phrase_complete = phrase.replace(tokenizer_bert.mask_token, token)
        print(f"{probabilite.item():.2%} | {phrase_complete}")
        propositions.append({"probabilite": float(probabilite.item()), "token": token, "phrase": phrase_complete})

    # La cellule de code du notebook d'origine utilise aussi cette variante.
    phrase_rabat = f"Rabat est la {tokenizer_bert.mask_token} Maroc."
    entrees_rabat = tokenizer_bert(phrase_rabat, return_tensors="pt")
    position_rabat = (entrees_rabat["input_ids"][0] == tokenizer_bert.mask_token_id).nonzero(as_tuple=True)[0].item()
    with torch.no_grad():
        logits_rabat = modele_bert(**entrees_rabat).logits[0, position_rabat, :]
    probs_rabat, ids_rabat = torch.topk(torch.softmax(logits_rabat, dim=-1), k=5)
    print("\\nVariante présente dans le code d'origine :")
    rabat = []
    for probabilite, indice in zip(probs_rabat, ids_rabat):
        token = tokenizer_bert.decode([indice.item()]).strip()
        complete = phrase_rabat.replace(tokenizer_bert.mask_token, token)
        print(f"{probabilite.item():.2%} | {complete}")
        rabat.append({"probabilite": float(probabilite.item()), "token": token, "phrase": complete})
    statut["bert"] = {"ok": True, "paris": propositions, "rabat": rabat}
except Exception as erreur:
    statut["bert"] = {"ok": False, "erreur": f"{type(erreur).__name__}: {erreur}"}
    print("BERT n'a pas pu être exécuté.")
    print(statut["bert"]["erreur"])
    print("Aucune proposition n'est inventée.")

destination = ROOT / "outputs" / "metrics" / "02_modeles_preentraines.json"
destination.write_text(json.dumps(statut, ensure_ascii=False, indent=2), encoding="utf-8")
print("Compte rendu :", destination)
        """),
        md("""
### Pourquoi l'attention est bidirectionnelle

Le masque est au milieu de la phrase. Pour proposer un mot, BERT peut s'appuyer
sur « Paris est la » **et** sur « de la France ». Un décodeur causal n'aurait
pas le droit de voir la fin. Cette liberté correspond à l'entraînement masqué :
le modèle apprend à remplir un trou, pas à continuer un texte.

Les probabilités affichées sont celles du softmax sur le vocabulaire entier.
Elles décrivent la préférence du modèle, pas une certitude factuelle.
Un token proposé peut aussi être un fragment de mot.
        """),
    ]
    notebook("02_gpt2_bert_pretrained.ipynb", cells)


def build_energy() -> None:
    cells = [
        md("""
# Exercice 1 — EnergyTransformer

Objectif : prédire la **puissance active moyenne de l'heure suivante**, en kW,
à partir des **24 heures précédentes**.

Les mesures viennent du fichier UCI *Individual Household Electric Power Consumption*,
fourni dans l'archive du projet. On utilise seulement `Global_active_power`.
Les points d'interrogation sont des valeurs manquantes.

Chaque prédiction de test est faite à partir des **vraies** heures déjà observées.
On ne réinjecte pas les prédictions du modèle dans la fenêtre suivante.
        """),
        code(SETUP),
        md("""
## 1. Préparation, sans fuite

1. Lire le fichier avec `sep=';'` et `na_values=['?']`.
2. Construire l'horodatage `Date + Time`.
3. Moyenner par heure et écarter les heures de moins de 45 mesures valides.
4. Couper la série dans l'ordre : 70 % train, 15 % validation, 15 % test.
5. Calculer la moyenne et l'écart-type **uniquement** sur le train.
6. Former des fenêtres de 24 heures et supprimer celles qui contiennent un NaN.

Le lot d'une fenêtre dépend de l'indice de sa cible. Les entrées sont toujours
les 24 heures d'avant. Une fenêtre de validation peut donc commencer à la fin
du train : ce sont des observations passées, pas la cible de test.
        """),
        code("""
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_hourly_power, window_split

serie = load_hourly_power()
prepare = window_split(serie)
print("Heures de la série, y compris les trous :", prepare["n"])
print("Heures numériques :", int(serie.notna().sum()))
print("Début :", serie.index[0], "| fin :", serie.index[-1])
print("Fin du train :", prepare["fin_train"], "| fin de la validation :", prepare["fin_val"])
print(f"Moyenne train : {prepare['mean']:.4f} kW | écart-type train : {prepare['std']:.4f} kW")
print("Fenêtres train / val / test :",
      len(prepare["X_train"]), len(prepare["X_val"]), len(prepare["X_test"]))
print("Forme d'une entrée :", tuple(prepare["X_train"].shape[1:]))
print("Exemple normalisé, 24 heures :")
print(np.array2string(prepare["X_train"][0, :, 0].numpy(), precision=3))
        """),
        md("""
## 2. Le modèle

`EnergyTransformer` suit la consigne :

- `Linear(1, 32)`, puis des positions apprises ;
- deux `TransformerEncoderLayer` ;
- `d_model=32`, `nhead=4`, `dim_feedforward=128`, `dropout=0.1`, `batch_first=True` ;
- `Linear(32, 1)` appliqué à la dernière heure.

L'optimiseur est Adam avec un pas de 0,001. La perte est la MSE sur les valeurs
normalisées. On fait au plus 20 époques, par lots de 64, et on recharge l'époque
dont la perte de validation est la plus basse.
        """),
        code("""
from src.models import EnergyTransformer
from src.training import train_with_validation

set_seed(42)
generateur = torch.Generator().manual_seed(42)
train_loader = DataLoader(
    TensorDataset(prepare["X_train"], prepare["y_train"]),
    batch_size=64,
    shuffle=True,
    generator=generateur,
)
val_loader = DataLoader(TensorDataset(prepare["X_val"], prepare["y_val"]), batch_size=64)
modele = EnergyTransformer()
print("Paramètres :", count_parameters(modele))
historique = train_with_validation(
    modele,
    train_loader,
    val_loader,
    epochs=20,
    lr=0.001,
    device=device,
    checkpoint_path=ROOT / "outputs" / "checkpoints" / "energy_transformer.pt",
)
        """),
        md("""
## 3. Erreurs en kilowatts

On dénormalise avec la moyenne et l'écart-type **du train** :

$$
\\text{kW} = z \\times \\sigma_{train} + \\mu_{train}
$$

La persistance répète la dernière heure de la fenêtre, donc l'heure `t-1`.
La référence journalière répète la première heure de la fenêtre, donc l'heure `t-24`.
        """),
        code("""
from src.evaluation import (
    baseline_predictions,
    consecutive_run,
    denormalize,
    longest_consecutive,
    plot_loss_curves,
    plot_metric_bars,
    plot_week,
    regression_scores,
    save_figure,
    save_json,
)
from src.training import predict

prediction_norm = predict(modele, prepare["X_test"], device=device)
reel_kw = denormalize(prepare["y_test"], prepare["mean"], prepare["std"])
pred_kw = denormalize(prediction_norm, prepare["mean"], prepare["std"])
bases = baseline_predictions(prepare["X_test"], prepare["mean"], prepare["std"])

scores = {
    "EnergyTransformer": regression_scores(reel_kw, pred_kw),
    "Persistance": regression_scores(reel_kw, bases["persistance"]),
    "Veille (t-24)": regression_scores(reel_kw, bases["veille"]),
}
tableau = pd.DataFrame(scores).T
print(tableau.to_string(float_format=lambda valeur: f"{valeur:.4f}"))

figure = plot_metric_bars(scores, "Test électrique : MAE et RMSE en kW")
save_figure(figure, ROOT / "outputs" / "figures" / "03_energy_barres.png")
plt.show()

figure = plot_loss_curves(
    {
        "Entraînement": [ligne["train_loss"] for ligne in historique],
        "Validation": [ligne["val_loss"] for ligne in historique],
    },
    "MSE normalisée de l'EnergyTransformer",
)
save_figure(figure, ROOT / "outputs" / "figures" / "03_energy_pertes.png")
plt.show()
        """),
        code("""
positions = prepare["positions_test"]
debut = consecutive_run(positions, 168)
if debut is None:
    debut, longueur = longest_consecutive(positions)
    print(f"Aucune plage de 168 heures. Plus longue plage continue : {longueur} heures.")
else:
    longueur = 168
    print("Plage de sept jours consécutifs trouvée.")

instants = prepare["index"][positions[debut:debut + longueur]]
print("Du", instants[0], "au", instants[-1])
figure = plot_week(
    instants,
    reel_kw[debut:debut + longueur],
    pred_kw[debut:debut + longueur],
    bases["persistance"][debut:debut + longueur],
)
save_figure(figure, ROOT / "outputs" / "figures" / "03_energy_sept_jours.png")
plt.show()

save_json(ROOT / "outputs" / "metrics" / "03_energy.json", {
    "device": str(device),
    "mean_train_kw": prepare["mean"],
    "std_train_kw": prepare["std"],
    "n_train": int(len(prepare["X_train"])),
    "n_val": int(len(prepare["X_val"])),
    "n_test": int(len(prepare["X_test"])),
    "meilleure_epoque": int(min(historique, key=lambda ligne: ligne["val_loss"])["epoch"]),
    "scores_kw": scores,
    "fenetre_graphique": {
        "debut": str(instants[0]),
        "fin": str(instants[-1]),
        "heures": int(longueur),
    },
})

for nom, mesure in scores.items():
    print(
        f"{nom}: MAE {mesure['mae']:.4f} kW | RMSE {mesure['rmse']:.4f} kW"
    )
        """),
        md("""
## 4. Questions de réflexion

Les réponses distinguent ce qui a été **mesuré** dans les cellules précédentes
et ce qui reste une **hypothèse**. Les chiffres exacts sont ceux du tableau,
pas une valeur apprise par cœur avant l'exécution.

### Le Transformer améliore-t-il les deux références ?

Réponse mesurée : comparer la MAE et la RMSE du modèle à celles de la persistance
et de la veille. Une amélioration n'existe que si les deux erreurs du modèle
sont plus basses. Si la persistance reste meilleure, il faut le dire :
l'électricité d'une heure ressemble beaucoup à celle de l'heure précédente,
et 20 époques d'un petit encodeur ne garantissent pas de battre cette copie.

### Une fenêtre de 168 heures capturerait-elle mieux la semaine ?

Hypothèse non testée. Le graphique de sept jours montre un rythme jour / nuit,
et le corpus du cours rappelle qu'un même jour de la semaine précédente est hors
de portée d'une fenêtre de 24 heures. Une fenêtre de 168 heures pourrait voir
ce cycle. Elle n'a pas été entraînée ici. Son coût d'attention grandirait,
et il faudrait refaire le découpage sans fuite avant de conclure.

### À quoi serviraient l'heure et le jour ?

Hypothèse non testée. L'heure du jour et le jour de la semaine sont des indices
calendaires que la puissance seule ne nomme pas explicitement. Les positions
apprises distinguent « il y a 1 heure » de « il y a 23 heures », mais elles ne
disent pas « c'est un dimanche ». Ajouter ces variables pourrait aider les
horaires réguliers. Ce notebook ne contient pas cette expérience.

### Pourquoi normaliser seulement avec le train ?

La moyenne et l'écart-type décrivent le jeu qui sert à apprendre. Si on les
calcule avec le test, une information du futur entre dans le prétraitement :
c'est une fuite. Au moment d'une vraie prévision, le futur n'est pas connu.
Le test de `tests/test_data.py` vérifie qu'en changeant la fin de la série,
ces deux statistiques ne bougent pas. Le retour en kilowatts utilise les mêmes
statistiques de train.

### Comment prédire des kWh plutôt que des kW ?

`Global_active_power` est une puissance en kilowatts. La moyenne des minutes
d'une heure complète est numériquement égale à l'énergie de cette heure en kWh,
parce qu'une heure multipliée par la puissance moyenne redonne des kilowattheures.
Si les 60 minutes sont présentes, `somme(puissance) / 60` et la moyenne coïncident.
Comme on exige seulement 45 mesures, cette égalité n'est plus exacte pour les
heures incomplètes : l'énergie devrait alors sommer les minutes réellement
observées, en comptant chaque minute comme un soixantième d'heure, sans remplacer
les trous par la moyenne. Cette adaptation n'a pas été réentraînée.
        """),
    ]
    notebook("03_energy_transformer.ipynb", cells)


def build_minigpt() -> None:
    cells = [
        md("""
# Exercice 2 — Mini-GPT depuis zéro

Ce modèle n'utilise aucun poids Hugging Face. Il apprend à prédire le **caractère
suivant** dans `corpus.txt`, un texte original écrit pour l'exercice.

Le contexte vaut 64 caractères. Les embeddings de caractères et de positions
ont la dimension 64. Deux blocs pré-normalisés utilisent quatre têtes.
Le réseau interne fait `64 → 256 → 64` avec GELU. La projection finale couvre
le vocabulaire de caractères.

La cible est décalée d'un caractère. La perte d'entropie croisée est calculée
sur **toutes** les positions, pas seulement la dernière.
        """),
        code(SETUP),
        md("""
## 1. Vocabulaire de caractères

`stoi` donne un numéro à chaque caractère. `itos` fait le chemin inverse.
Pour une entrée qui se termine visuellement par `Bonjou`, la cible est le texte
décalé : chaque position apprend le caractère qui vient juste après.
        """),
        code("""
from src.data import load_char_corpus, split_token_stream

texte, stoi, itos, tokens = load_char_corpus(ROOT / "corpus.txt")
tokens_train, tokens_val = split_token_stream(tokens, train_ratio=0.9)
print("Caractères :", len(texte))
print("Taille du vocabulaire :", len(stoi))
print("Tokens train / validation :", len(tokens_train), len(tokens_val))
print("Extrait :", repr(texte[:80]))

exemple = "Bonjou"
print("Décalage pédagogique, si les caractères existent dans le vocabulaire :")
print("entrée :", exemple[:-1])
print("cible  :", exemple[1:])

contexte = 64
x_demo = tokens[:contexte]
y_demo = tokens[1:contexte + 1]
print("Forme d'une fenêtre réelle :", tuple(x_demo.shape), "->", tuple(y_demo.shape))
print("Entrée :", repr("".join(itos[int(i)] for i in x_demo[:40])), "...")
print("Cible  :", repr("".join(itos[int(i)] for i in y_demo[:40])), "...")
        """),
        md("""
## 2. Masque causal et entraînement

`True` dans le masque signifie : cette paire est interdite. Le triangle
supérieur, diagonale exclue, cache le futur.

L'entraînement dure 60 époques de 80 lots de 32 fenêtres. Ce budget n'est pas
imposé par le TP : il reste raisonnable sur CPU. Adam utilise un pas de 0,003,
choix de travail puisque le sujet ne fixe pas le pas. Le meilleur état selon
la perte de validation est rechargé avant la génération.
        """),
        code("""
from src.text_models import MiniGPT, causal_mask
from src.training import train_language_model

masque = causal_mask(8)
print("Masque 8 x 8, True = interdit :")
print(masque.int().numpy())

set_seed(42)
mini_gpt = MiniGPT(vocab_size=len(stoi), context=64, dropout=0.0)
print("Paramètres :", count_parameters(mini_gpt))
historique = train_language_model(
    mini_gpt,
    tokens_train,
    tokens_val,
    context=64,
    epochs=60,
    steps_per_epoch=80,
    batch_size=32,
    lr=3e-3,
    device=device,
    seed=42,
)
        """),
        code("""
from src.evaluation import plot_loss_curves, save_figure, save_json
from src.text_models import decode_ids, generate_characters

figure = plot_loss_curves(
    {
        "Entraînement": [ligne["train_loss"] for ligne in historique],
        "Validation": [ligne["val_loss"] for ligne in historique],
    },
    "Entropie croisée du mini-GPT",
    ylabel="Perte",
)
save_figure(figure, ROOT / "outputs" / "figures" / "04_minigpt_pertes.png")
plt.show()

amorce = "Le transformeur lit une séquence et prédit le "
identifiants = torch.tensor([[stoi[caractere] for caractere in amorce]], dtype=torch.long)
generations = {}
for temperature in (0.5, 1.0, 1.5):
    generateur = torch.Generator().manual_seed(42)
    suite = generate_characters(
        mini_gpt, identifiants, n_new=200, temperature=temperature,
        device=device, generator=generateur,
    )
    texte_complet = decode_ids(suite[0], itos)
    generations[str(temperature)] = texte_complet
    print("\\n" + "=" * 72)
    print(f"Température {temperature}")
    print(texte_complet)
        """),
        md("""
## 3. Test obligatoire de causalité

En mode `eval()`, on modifie un caractère situé après la position 20.
Les logits des positions 0 à 19 ne doivent pas changer. Si elles changeaient,
le bloc aurait lu le futur.
        """),
        code("""
mini_gpt.eval()
lot = tokens_train[:64].unsqueeze(0).clone()
modifie = lot.clone()
position = 20
modifie[0, position] = (modifie[0, position] + 1) % len(stoi)
with torch.no_grad():
    avant = mini_gpt(lot.to(device)).cpu()
    apres = mini_gpt(modifie.to(device)).cpu()
ecart = (avant[:, :position] - apres[:, :position]).abs().max().item()
print(f"Écart maximal sur les logits passés : {ecart:.3e}")
if ecart > 1e-5:
    raise RuntimeError("Le masque causal ne protège pas les positions passées.")
print("Les positions antérieures sont inchangées.")

save_json(ROOT / "outputs" / "metrics" / "04_minigpt.json", {
    "device": str(device),
    "vocabulaire": len(stoi),
    "caracteres": len(texte),
    "perte_finale": historique[-1],
    "meilleure_validation": min(ligne["val_loss"] for ligne in historique),
    "generations": generations,
    "ecart_causal": ecart,
})
        """),
        md("""
## Questions

### Que se passe-t-il si l'on retire le masque pendant l'entraînement ?

Le test ci-dessus montre le mécanisme, pas un second entraînement : avec le
masque, changer le futur ne change pas le passé. Sans masque, une position
pourrait lire la cible située plus loin dans la même fenêtre. La tâche deviendrait
artificiellement facile. À la génération, ce futur n'existe pas encore, donc
le modèle serait utilisé dans une situation qu'il n'a pas apprise. Cette
conséquence sur la qualité du texte n'a pas été mesurée par un entraînement
sans masque : elle suit directement du décalage entre entraînement et génération.

### Quel effet a la température ?

Les trois textes de 200 caractères sont produits avec la même amorce et la même
graine de tirage. Une température de 0,5 rend la distribution plus piquée :
le modèle répète davantage les enchaînements qu'il juge probables. Une température
de 1,0 laisse les probabilités telles quelles. Une température de 1,5 les aplatit
et laisse apparaître des caractères plus inattendus. Il faut comparer les textes
affichés, sans prétendre qu'un petit corpus produit une prose littéraire.

### Pourquoi décaler les cibles ?

Le modèle reçoit des caractères déjà connus et doit dire celui qui vient après.
Si la cible n'était pas décalée, il apprendrait à recopier l'entrée. L'exemple
`Bonjou` le montre : l'entrée `Bonjo` a pour cible `onjou`. Dans le tenseur réel,
`y` commence un caractère après `x` et se termine un caractère plus loin.
        """),
    ]
    notebook("04_mini_gpt.ipynb", cells)


def build_minibert() -> None:
    cells = [
        md("""
# Exercice 3 — Mini-BERT, en bonus

On entraîne depuis zéro un encodeur qui retrouve des mots masqués.
Ce n'est pas le BERT préentraîné du second notebook : le corpus est petit,
le vocabulaire est construit sur nos phrases, et les poids partent au hasard.

Les identifiants réservés sont `PAD=0`, `UNK=1`, `CLS=2`, `SEP=3`, `MASK=4`.
Chaque phrase est complétée jusqu'à 32 tokens :

`[CLS] le chat dort sur le canapé [SEP] [PAD] ...`

On sélectionne environ 15 % des tokens ordinaires. Parmi eux, 80 % deviennent
`[MASK]`, 10 % un mot ordinaire aléatoire, et 10 % restent visibles. La perte
ne concerne que ces positions, grâce à `ignore_index=-100`.
        """),
        code(SETUP),
        md("""
## 1. Phrases et vocabulaire

Les phrases de validation réutilisent uniquement des mots vus à l'entraînement,
mais ce ne sont pas les mêmes combinaisons. Le modèle peut donc être interrogé
sur des phrases connues pour la démonstration, et mesuré sur des phrases tenues
à l'écart pour l'exactitude.

Le réseau feed-forward interne fait `64 → 256 → 64`. Le sujet ne fixe pas cette
largeur : 256 reprend le même facteur quatre que le mini-GPT. Le dropout vaut 0,1.
L'attention n'a pas de masque causal. Elle a un masque de padding.
        """),
        code("""
from src.data import (
    build_minibert_corpus,
    build_word_vocab,
    encode_sentences,
    tokenize_words,
)

phrases_train, phrases_val = build_minibert_corpus()
stoi, itos = build_word_vocab(phrases_train)
ids_train = encode_sentences(phrases_train, stoi, max_length=32)
ids_val = encode_sentences(phrases_val, stoi, max_length=32)
print("Phrases train / validation :", len(phrases_train), len(phrases_val))
print("Vocabulaire :", len(stoi))
print("Forme :", tuple(ids_train.shape), tuple(ids_val.shape))
exemple = encode_sentences(["le chat dort sur le canapé"], stoi)[0]
print("Exemple :", " ".join(itos[int(i)] for i in exemple if int(i) != stoi["[PAD]"]))
        """),
        code("""
from src.text_models import corrupt_for_mlm

generateur = torch.Generator().manual_seed(42)
corrompu, labels, selection = corrupt_for_mlm(ids_train[:4], len(stoi), generateur)
for ligne in range(4):
    mots = [itos[int(indice)] for indice in corrompu[ligne] if int(indice) != 0]
    cibles = [
        itos[int(labels[ligne, colonne])]
        for colonne in range(labels.size(1))
        if int(labels[ligne, colonne]) != -100
    ]
    print("entrée :", " ".join(mots))
    print("à prédire :", cibles)
    print()
        """),
        md("""
## 2. Entraînement

Adam, pas de 0,001, 30 époques, lots de 32. Ce budget est un choix de travail :
le sujet demande un entraînement, pas un nombre d'époques. On conserve l'époque
de meilleure perte de validation. L'exactitude affichée pendant l'entraînement
dépend des masques tirés à cette époque. La mesure finale est refaite ensuite,
avec une graine fixe, sur le modèle retenu.
        """),
        code("""
from src.text_models import MiniBERT
from src.training import evaluate_masked_language_model, train_masked_language_model

set_seed(42)
mini_bert = MiniBERT(vocab_size=len(stoi), max_length=32, dropout=0.1)
print("Paramètres :", count_parameters(mini_bert))
historique = train_masked_language_model(
    mini_bert,
    ids_train,
    ids_val,
    vocab_size=len(stoi),
    epochs=30,
    batch_size=32,
    lr=1e-3,
    device=device,
    seed=42,
)
final = evaluate_masked_language_model(
    mini_bert, ids_val, vocab_size=len(stoi), device=device, seed=123
)
print(f"Exactitude de validation, masques figés par la graine 123 : {final['accuracy']:.2%}")
print(f"Perte de validation correspondante : {final['loss']:.4f}")
        """),
        code("""
from src.evaluation import plot_loss_curves, save_figure, save_json
from src.text_models import topk_words

figure = plot_loss_curves(
    {
        "Entraînement": [ligne["train_loss"] for ligne in historique],
        "Validation": [ligne["val_loss"] for ligne in historique],
    },
    "Perte masquée du mini-BERT",
    ylabel="Entropie croisée",
)
save_figure(figure, ROOT / "outputs" / "figures" / "05_minibert_pertes.png")
plt.show()

def montrer(phrase: str, mot: str) -> None:
    mots = tokenize_words(phrase)
    position = 1 + mots.index(mot)
    ids = encode_sentences([phrase], stoi)[0].clone()
    vrai = itos[int(ids[position])]
    ids[position] = stoi["[MASK]"]
    masque = ids.eq(stoi["[PAD]"]).unsqueeze(0)
    mini_bert.eval()
    with torch.no_grad():
        logits = mini_bert(ids.unsqueeze(0).to(device), masque.to(device)).cpu()[0]
    propositions = topk_words(logits, position, itos, k=3)
    print(phrase, "-> masque sur", mot)
    for rang, (token, proba) in enumerate(propositions, start=1):
        marque = "  <-- mot d'origine" if token == vrai else ""
        print(f"  {rang}. {token:20s} {proba:6.1%}{marque}")

montrer("paris est la capitale de la france", "capitale")
montrer("le chat dort sur le canapé", "dort")
montrer("le chien dort sur le tapis", "tapis")
montrer("la consommation augmente le soir", "augmente")
        """),
        md("""
## 3. Invariance au padding

On change les identifiants placés sur les positions de padding, sans changer
le masque. En mode `eval()`, les logits des tokens utiles doivent rester les mêmes.
        """),
        code("""
ids = encode_sentences(["le chat dort sur le canapé"], stoi)
masque_pad = ids.eq(stoi["[PAD]"])
autre = ids.clone()
autre[masque_pad] = torch.arange(5, 5 + int(masque_pad.sum()))
mini_bert.eval()
with torch.no_grad():
    logits_a = mini_bert(ids.to(device), masque_pad.to(device)).cpu()
    logits_b = mini_bert(autre.to(device), masque_pad.to(device)).cpu()
utiles = ~masque_pad
ecart = (logits_a[utiles] - logits_b[utiles]).abs().max().item()
print(f"Écart maximal hors padding : {ecart:.3e}")
if ecart > 1e-4:
    raise RuntimeError("Le masque de padding ne bloque pas les jetons de remplissage.")
print("Les prédictions utiles sont inchangées.")

save_json(ROOT / "outputs" / "metrics" / "05_minibert.json", {
    "device": str(device),
    "vocabulaire": len(stoi),
    "exactitude_validation": final["accuracy"],
    "perte_validation": final["loss"],
    "positions_evaluees": final["selected"],
    "ecart_padding": ecart,
    "historique_final": historique[-1],
})
        """),
        md("""
## Questions

### Pourquoi BERT peut-il utiliser le contexte à droite ?

Parce que sa tâche est de remplir un trou dans une phrase déjà écrite.
Rien n'oblige à cacher la partie droite : elle fait partie du contexte connu.
L'encodeur n'a donc pas de masque causal. Le mini-GPT, au contraire, doit
produire un caractère qui n'existe pas encore, donc il cache le futur.

### Pourquoi la perte ignore-t-elle les autres positions ?

À chaque exemple, seules quelques positions ont été choisies pour être prédites.
Les autres mots sont restés visibles, ou servent de contexte. Calculer la perte
partout pousserait surtout le modèle à recopier les mots déjà donnés. Le `-100`
dit à `CrossEntropyLoss` de ne pas compter ces positions. Les positions
sélectionnées mais laissées inchangées — les 10 % — restent dans la perte :
le modèle doit quand même les deviner.

### Pourquoi ce modèle n'écrit-il pas comme GPT ?

Il n'a pas appris à continuer une suite sous une contrainte causale. Il propose
une distribution de mots à une position masquée, en voyant les deux côtés.
Pour écrire de gauche à droite, il faudrait une autre procédure, et le résultat
ne serait pas celui d'un décodeur entraîné à cette tâche. Les trois propositions
affichées plus haut illustrent cette consultation ponctuelle, pas une rédaction.
        """),
    ]
    notebook("05_mini_bert_bonus.ipynb", cells)


if __name__ == "__main__":
    build_sine()
    build_pretrained()
    build_energy()
    build_minigpt()
    build_minibert()
