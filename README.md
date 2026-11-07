# gopu

Framework en **Rust pur** (bibliothèque standard uniquement, zéro dépendance externe) avec une API **Python** (`ctypes`, rien à installer côté Python). Il contient :

1. **gopu-lm** : un modèle de langage de type chat (Transformer, tokenizer BPE, attention, têtes de prédiction du futur, appels d'outils), entraînable sur CPU.
2. Un petit moteur de réseaux denses (MLP) avec rétropropagation, AdamW et lecture/écriture `.safetensors` / `.b8n`.
3. Un orchestrateur de sous-agents avec graphe de session, boîtes aux lettres et DAG de tâches.

> **Honnêteté d'abord.** `modele.b8n` est un modèle minuscule (environ 0,65 M de paramètres, dont 0,45 M de table de positions) entraîné sur des conversations générées à partir de modèles-types. Il reproduit très bien le format (réflexion → outil → réponse) et les tâches du corpus (salutations, calculs via `python`/`calc`, heure, capitales…), mais il **ne raisonne pas** et ne répond pas correctement à des questions libres. Pour lui apprendre autre chose, il faut un corpus plus large et plus varié, au même format.

---

## 1. Installation

Prérequis : **Rust** ([rustup.rs](https://rustup.rs) ; sous Termux : `pkg install rust`) et **Python ≥ 3.8**.

```sh
pip install .          # compile le Rust et installe tout
pip install -e .       # idem, en mode développement
```

Ou sans pip : `./build.sh` (tests + compilation + démos).

`setup.py` fait : `cargo build --release`, puis installe le paquet `gopu_lm` avec la bibliothèque native (`libgopu.so` / `.dylib` / `.dll`), le modèle `modele.b8n` par défaut et la commande `gopu-lm`. Le paquet `gopu` (MLP + orchestrateur) est installé aussi s'il existe dans `python/gopu/`.

Utilisation depuis les sources, sans installer : `cargo build --release`, puis `sys.path.insert(0, "python")` (la bibliothèque est trouvée dans `target/release/` ; sinon variable `GOPU_LIB=/chemin/libgopu.so`).

## 2. Démarrage rapide

```sh
python3 test_chat.py            # discuter avec le modèle dans le terminal
python3 train.py                # entraîner / continuer l'entraînement sur data/*.txt
```

```python
from gopu_lm import Model
M = Model.load_model("modele.b8n")                 # ou Model.load_model() : modèle installé avec le paquet
r = M.askInto(prompt="Combien font 17 fois 23 ?", temperature=0.2, max_tokens=160, seed=1)
print(r)              # réponse seule : "17 fois 23 font 391."
print(r.thinking)     # la réflexion du modèle
print(r.tool_calls)   # [{'tool': 'python', 'input': 'print(17 * 23)', 'output': '391'}]
```

## 3. Format de conversation

Le modèle est entraîné sur des conversations dans ce format (chacune se termine par `<|eos|>`) :

```
<|user|>
Combien font 17 fois 23 ?
</|user|>
<|thinking|>
L'utilisateur demande un calcul. Pour être sûr du résultat, j'utilise l'outil python.
</|thinking|>
<|assistant|>
<|tools="python">
print(17 * 23)
</|tools>
<|tool_result>
391
</|tool_result>
17 fois 23 font 391.
</|assistant|>
<|eos|>
```

| Balise | Rôle |
|---|---|
| `<\|system\|>…</\|system\|>` | consigne système (facultative) |
| `<\|user\|>…</\|user\|>` | message de l'utilisateur |
| `<\|thinking\|>…</\|thinking\|>` | réflexion du modèle avant de répondre |
| `<\|assistant\|>…</\|assistant\|>` | réponse du modèle |
| `<\|tools="nom">…</\|tools>` | appel d'outil (le nom est du texte libre : `python`, `calc`, ou le tien) |
| `<\|tool_result>…</\|tool_result>` | résultat de l'outil, **injecté par le programme**, jamais écrit par le modèle |
| `<\|eos\|>` | fin de conversation |

Pendant l'entraînement, la **perte n'est calculée que sur ce que le modèle doit produire** : réflexion, appel d'outil, réponse, fin de tour. Le message utilisateur et le résultat d'outil sont ignorés (option `user_weight` pour apprendre aussi le texte utilisateur).

Plusieurs tours peuvent s'enchaîner dans une même conversation (`<|user|>` … `<|assistant|>` … `<|user|>` …).

## 4. Entraînement : `train.py`

```sh
python3 train.py                                   # reprend modele.b8n (sinon en crée un), lit data/*.txt
python3 train.py --fresh                           # repart de zéro (nouveau tokenizer, nouveaux poids)
python3 train.py --epochs 5 --batch 8 --max-minutes 30
python3 train.py --data-dir mes_donnees --model autre.b8n --out sortie.b8n
python3 train.py --fresh --data-dir data/improved --model modele_ameliore.b8n --pack-len 256
```

- **Données** : tous les fichiers `data/*.txt`, au format de la section 3. `python3 tools/gen_data.py --n 6000 --out data/train.txt` fabrique un corpus d'essai (salutations, calculs avec outils, heure, capitales, code, dialogues à plusieurs tours). Remplace-le par tes propres conversations pour changer le comportement du modèle.
- Pour enrichir le corpus existant sans le remplacer : `python3 tools/expand_data.py --repeat 3 --seed 23`. Cette commande ajoute trois formulations tirées au sort pour chacun de 98 thèmes (Rust, Python, informatique, sciences, géographie, français, mathématiques et réponses prudentes). Elle ajoute 294 conversations ; `--repeat` contrôle le nombre d'exemples par thème.
- **Contexte : 7 024 jetons** (`CONTEXT_TOKENS`, option `--context`). Si le modèle repris avait un contexte plus court, sa table de positions est étendue (les positions existantes sont conservées).
- **Plafond : 1 000 000 de jetons** (`MAX_TRAIN_TOKENS`, option `--max-tokens`) : au-delà, le corpus est tronqué (après mélange).
- **Empaquetage** (`--pack-len`, défaut 1024) : les conversations sont mises bout à bout pour former des séquences de ~1024 jetons, afin que le modèle apprenne aussi les positions lointaines. **Les positions au-delà de `--pack-len` ne sont jamais apprises** par défaut : le contexte de 7 024 est une *capacité*, pas une garantie de qualité sur de longs textes. Pour vraiment exploiter un long contexte, il faut des données longues et un `--pack-len` plus grand (temps et mémoire croissent environ comme la longueur au carré).
- L'ancien modèle est sauvegardé en `modele.b8n.prev` avant d'être remplacé.
- Réglages d'un **nouveau** modèle : `--vocab 1024 --d 128 --layers 4 --heads 8 --future 2`. C'est un modèle plus grand ; utilise `--fresh --model modele_ameliore.b8n` pour le créer sans remplacer le modèle existant.
- Autres options : `--lr`, `--steps` (remplace `--epochs`), `--batch`, `--seed`, `--no-backup`.

Pendant l'entraînement : perte lissée, perte de validation (1 conversation sur 20 est mise de côté), taux d'apprentissage, norme du gradient. L'entraînement utilise tous les cœurs. L'optimiseur est AdamW avec montée linéaire puis décroissance en cosinus et écrêtage du gradient.

### Adaptation LoRA

`gopu.trl.lora` ajoute une adaptation faible rang sur la tête de sortie, sans modifier les poids du Transformer. L'adaptateur est enregistré séparément et ne peut être chargé que sur une architecture compatible.

```python
from gopu_lm import Model
from gopu.trl.lora import LoRAConfig, LoRATrainer

model = Model.load_model("modele_ameliore.b8n")
trainer = LoRATrainer(model, LoRAConfig(rank=8, steps=300))
trainer.train("mes_exemples.txt")
trainer.save("domaine.lora.b8n")
# Plus tard : model.load_lora("domaine.lora.b8n")
```

Le benchmark s'exécute avec `python3 tools/evaluate_lm.py modele_ameliore.b8n`. Son jeu de questions est stocké séparément dans `data/eval.jsonl`.

Équivalent en ligne de commande Rust :
```sh
./target/release/gopu-lm train --data data/train.txt --out modele.b8n --resume modele.b8n \
    --ctx 7024 --pack-len 1024 --max-corpus-tokens 1000000 --steps 500 --batch 8 --lr 2e-3
```

## 5. API Python (`gopu_lm`)

```python
M = Model.load_model("modele.b8n")        # .b8n ou .safetensors
M.save_model("copie.b8n")
M = Model.new(corpus_texte, vocab=640, d=64, layers=3, heads=4, ctx=7024, future=2)   # nouveau modèle
M.train(corpus_texte_ou_chemin, steps=500, batch=8, lr=2e-3, pack_len=1024, max_corpus_tokens=1_000_000)
```

### `askInto`

```python
r = M.askInto(
    prompt="Bonjour",           # ou un dict : M.askInto({"prompt": "Bonjour", "temperature": 0})
    system=None,                # consigne système
    history=None,               # [{"role": "user", "content": ...}, {"role": "assistant", "content": ..., "raw": r.raw}]
    max_tokens=160, temperature=0.7, top_k=40, top_p=0.95, repetition_penalty=1.0, seed=None,
    stop=None,                  # chaînes d'arrêt supplémentaires
    tools="auto",               # "auto" = python + calc ; False = aucun ; {nom: fonction} = ajoute/remplace
    max_tool_rounds=3,          # nombre maximal d'appels d'outils par réponse
    show_thinking=False,        # préfixe la réponse par la réflexion
    return_dict=False,          # renvoie un dict au lieu d'un Reply
)
```
Le résultat est un `Reply` (sous-classe de `str`) avec `.thinking`, `.tool_calls`, `.raw` (texte brut généré), `.finish`, `.n_tokens`. Alias : `ask_into`, `M(...)`. `temperature=0` donne une sortie déterministe.

### Outils

Quand le modèle écrit `<|tools="nom">code</|tools>`, la génération s'arrête, la fonction du même nom est appelée avec `code`, son résultat est injecté dans `<|tool_result>`, puis le modèle continue.

```python
def meteo(ville): return "ensoleillé, 24 °C"
M.askInto("Quel temps à Kinshasa ?", tools={"meteo": meteo})   # n'a de sens que si le corpus contient des exemples de l'outil « meteo »
```
Outils fournis : `python` (exécute le code dans un sous-processus isolé : `python -I`, dossier temporaire, délai 5 s, limite CPU, limite mémoire 512 Mo) et `calc` (calculatrice arithmétique sûre).
**Sécurité** : `python` exécute du code écrit par un modèle ; ce n'est **pas** un bac à sable complet. Utilise `tools=False` pour le désactiver.
**Android/Termux** : la limite mémoire est désactivée automatiquement (le chargeur d'Android réserve beaucoup de mémoire virtuelle et empêchait Python de démarrer). Réglable avec la variable `GOPU_PY_MEM_MB` (`0` = pas de limite).

### Tokenizer, embeddings, futur

```python
ids = M.tokenize("Bonjour")                # liste d'ids ; M.detokenize(ids) pour revenir au texte
vecteurs = M.embed("Bonjour")              # embeddings appris, un vecteur par jeton
etats = M.embed("Bonjour", "hidden")       # états contextuels (après les couches d'attention)
M.predict_future("Bonjour", top=3)         # jetons les plus probables à +1, +2, +3
M.info                                     # configuration et nombre de paramètres
```

## 6. Ligne de commande

```sh
gopu-lm info  --model modele.b8n
gopu-lm ask   --model modele.b8n --prompt "Bonjour" --temperature 0 --future 1
gopu-lm train --data data/train.txt --out modele.b8n [--resume modele.b8n] [--steps N] [--batch N] [--lr X]
              [--ctx N] [--pack-len N] [--max-corpus-tokens N] [--d 64 --layers 3 --heads 4 --future 2 --vocab 640]
```
(`gopu-lm ask` génère le texte brut, sans exécuter les outils : utilise Python pour cela. Sans installation : `./target/release/gopu-lm`.)

## 7. Architecture

- `src/tokenizer.rs` : BPE entraînable au niveau des octets (256 octets + 14 jetons spéciaux + fusions apprises). Chaque chiffre est un jeton à part (pour recopier les nombres fidèlement). Aller-retour texte ↔ jetons exact.
- `src/lm.rs` : embeddings de jetons et de positions (appris) → *N* blocs [LayerNorm → **attention causale multi-têtes** → résidu → LayerNorm → MLP GELU → résidu] → LayerNorm finale → sortie **partagée** avec les embeddings. **Têtes de prédiction du futur** (multi-token prediction) : la tête *j* prédit le jeton *t+2+j* ; elles servent d'objectif auxiliaire à l'entraînement et alimentent `predict_future`. Rétropropagation écrite à la main, **vérifiée par différences finies** dans `cargo test`.
- `src/ffi_lm.rs`, `src/ffi.rs` : interface C (cdylib) ; `python/gopu_lm/` : paquet Python.
- `src/bin/gopu-lm.rs` : CLI.
- `src/tensor.rs`, `src/nn.rs`, `src/io.rs`, `src/json.rs` : tenseurs f32, MLP + AdamW, formats de fichiers, JSON minimal.
- `src/graph.rs`, `src/bus.rs`, `src/orch.rs` : graphe de session, boîtes aux lettres par agent, DAG de tâches en vagues parallèles (réessais, annulation, budget).

Configuration par défaut : `d=64`, 3 couches, 4 têtes, `ff=256`, 2 têtes de futur, vocabulaire 640, contexte 7 024 → environ 0,65 M de paramètres dont ~0,45 M de positions.

### Formats de fichier

`.b8n` : `"B8N\0"` | version u16 LE | taille d'en-tête u64 LE | en-tête JSON (même schéma que safetensors, avec `__metadata__`) | données f32 LE | CRC32 u32 LE. `.safetensors` est aussi pris en charge (lecture/écriture). Le modèle de langage y stocke ses tenseurs nommés (`tok_emb`, `pos_emb`, `layers.N.attn.wq`, …) et, dans les métadonnées, sa configuration et les fusions du tokenizer. Les anciens modèles MLP (`arch = gopu-mlp`, p. ex. `modele_xor.b8n`) restent lisibles par `gopu::io::load_model`.

## 8. Orchestrateur de sous-agents (inchangé)

```python
import gopu
o = gopu.Orchestrator("ma-session")
o.remember("gopu.inc est ecrit en Rust.")
o.register_builtin("recherche", "recall")
o.register("majuscules", lambda prompt, ctx, deps: " ".join(deps).upper())
o.run([{"name": "a", "agent": "recherche", "prompt": "Rust", "deps": []},
       {"name": "b", "agent": "majuscules", "prompt": "", "deps": [0]}])
o.save_graph("session.graph")
```
(nécessite le paquet Python `gopu`, absent de ce dépôt : voir `src/ffi.rs` pour l'API C correspondante.)

## 9. Tests

```sh
cargo test --release           # gradients, tokenizer, entraînement, sauvegarde/rechargement, poids de perte
python3 python/test_lm.py modele.b8n   # test de bout en bout : outils, précision sur calculs inédits, embeddings, futur
```

## 10. Limites connues

- Modèle minuscule : pas de raisonnement, pas de culture générale hors corpus.
- Pas de cache clé/valeur : chaque jeton généré recalcule tout le contexte. Un prompt très long est lent (le coût de l'attention croît avec le carré de la longueur) et gourmand en mémoire temporaire (≈ 4·T²·4 octets par couche, soit ~0,8 Go pour T = 7 024).
- Positions apprises : au-delà de la longueur vue à l'entraînement (`--pack-len`), la qualité n'est pas garantie.
- Calcul f32 sur CPU, multiplication de matrices simple (pas de SIMD explicite ni de BLAS).
- L'optimiseur n'est pas sauvegardé : reprendre un entraînement redémarre AdamW (le modèle, lui, est repris tel quel).

## 11. Dépannage

| Symptôme | Cause / solution |
|---|---|
| `libgopu introuvable` | lance `cargo build --release` ou `pip install .`, ou `GOPU_LIB=/chemin/libgopu.so` |
| `bionic/linker … MapShadow … failed` dans le résultat de l'outil python | limite mémoire incompatible avec Android : corrigé (désactivée automatiquement) ; sinon `GOPU_PY_MEM_MB=0` |
| Réponse vide après un appel d'outil | le résultat de l'outil était une erreur ; regarde `r.tool_calls` |
| Réponses incohérentes en conversation longue | le modèle a été entraîné sur des contextes courts ; `/reset` dans `test_chat.py`, ou `--memory 1` |
| `Aucun fichier .txt dans 'data'` | mets des conversations au format de la section 3 dans `data/` |
| Perte qui ne baisse pas / `gradient non fini` | baisse `--lr` ; vérifie le format du corpus |
