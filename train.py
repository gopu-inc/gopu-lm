#!/usr/bin/env python3
"""Entraîne gopu-lm sur les fichiers texte du dossier data/.

    python3 train.py                       # reprend modele.b8n s'il existe (sinon crée un modèle), entraîne, sauvegarde
    python3 train.py --fresh               # repart de zéro (nouveau tokenizer + nouveaux poids)
    python3 train.py --epochs 5 --batch 8 --max-minutes 30
    python3 train.py --data-dir mes_donnees --model autre.b8n

Réglages demandés :
  CONTEXT_TOKENS = 7024       taille de contexte du modèle (en jetons ; la mémoire de position est étendue à 7024)
  MAX_TRAIN_TOKENS = 1_000_000  plafond de jetons de corpus utilisés pour l'entraînement

Les fichiers data/*.txt sont lus tels quels : des conversations au format de chat gopu (voir README.md),
chacune terminée par <|eos|>. Pour fabriquer un corpus d'essai : python3 tools/gen_data.py --n 6000 --out data/train.txt
"""
import argparse
import glob
import os
import shutil
import sys
import time

try:
    from gopu_lm import Model
except ImportError:  # depuis les sources, sans installation
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
    from gopu_lm import Model

CONTEXT_TOKENS = 7024
MAX_TRAIN_TOKENS = 1_000_000


def read_corpus(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        sys.exit("Aucun fichier .txt dans '%s'. Mets-y des conversations au format gopu "
                 "(ou génère-en : python3 tools/gen_data.py --out %s/train.txt)." % (data_dir, data_dir))
    parts = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            t = fh.read()
        if "<|eos|>" not in t:
            print("  ! %s : aucun <|eos|> trouvé, le fichier sera traité comme une seule conversation" % f)
        parts.append(t.rstrip("\n") + "\n")
        print("  %-40s %9d caractères" % (f, len(t)))
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Entraîne gopu-lm sur data/")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="modele.b8n", help="modèle à reprendre s'il existe")
    ap.add_argument("--out", default=None, help="fichier de sortie (défaut : --model)")
    ap.add_argument("--fresh", action="store_true", help="ignorer le modèle existant et en créer un nouveau")
    ap.add_argument("--context", type=int, default=CONTEXT_TOKENS, help="taille de contexte en jetons (défaut %d)" % CONTEXT_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TRAIN_TOKENS, help="plafond de jetons d'entraînement (défaut %d)" % MAX_TRAIN_TOKENS)
    ap.add_argument("--pack-len", type=int, default=1024,
                    help="longueur des séquences d'entraînement : plusieurs conversations mises bout à bout (défaut 1024). "
                         "Les positions au-delà ne sont jamais apprises ; plus grand = plus lent et plus de mémoire (~ longueur²)")
    ap.add_argument("--epochs", type=float, default=None, help="passages sur le corpus (défaut : 3 en reprise, 12 pour un nouveau modèle)")
    ap.add_argument("--steps", type=int, default=None, help="nombre d'étapes (remplace --epochs)")
    ap.add_argument("--batch", type=int, default=8, help="séquences par étape")
    ap.add_argument("--lr", type=float, default=None, help="taux d'apprentissage (défaut 2e-3 en reprise, 3e-3 sinon)")
    ap.add_argument("--max-minutes", type=float, default=0, help="arrêt après N minutes (0 = illimité)")
    ap.add_argument("--seed", type=int, default=1)
    # architecture d'un NOUVEAU modèle
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--future", type=int, default=2, help="têtes de prédiction du futur")
    ap.add_argument("--no-backup", action="store_true", help="ne pas sauvegarder l'ancien modèle en .prev")
    a = ap.parse_args()
    out = a.out or a.model

    print("Corpus :")
    corpus = read_corpus(a.data_dir)

    resume = os.path.exists(a.model) and not a.fresh
    if resume:
        M = Model.load_model(a.model)
        print("Reprise de %s (%d paramètres, contexte %d)" % (a.model, M.info["n_params"], M.info["ctx"]))
    else:
        M = Model.new(corpus, vocab=a.vocab, d=a.d, layers=a.layers, heads=a.heads, ctx=a.context, future=a.future, seed=a.seed)
        print("Nouveau modèle : %d paramètres" % M.info["n_params"])
    print("Contexte -> %d jetons | plafond d'entraînement : %d jetons" % (a.context, a.max_tokens))

    n_tok = len(M.tokenize(corpus))
    used = min(n_tok, a.max_tokens)
    print("Corpus : %d jetons%s" % (n_tok, " (plafonné à %d)" % a.max_tokens if n_tok > a.max_tokens else ""))
    tokens_per_step = a.batch * a.pack_len
    epochs = a.epochs if a.epochs is not None else (3 if resume else 12)
    steps = a.steps if a.steps else max(50, int(epochs * used * 0.95 / tokens_per_step))
    lr = a.lr if a.lr else (2e-3 if resume else 3e-3)
    print("Entraînement : %d étapes x %d séquences de ~%d jetons (≈ %.1f passages), lr %g\n" % (steps, a.batch, a.pack_len, steps * tokens_per_step / max(used, 1), lr))

    t0 = time.time()
    rep = M.train(corpus, steps=steps, batch=a.batch, lr=lr, ctx=a.context, pack_len=a.pack_len,
                  max_corpus_tokens=a.max_tokens, seed=a.seed, warmup=max(10, steps // 40),
                  log_every=max(1, steps // 30), eval_every=max(1, steps // 6), max_seconds=a.max_minutes * 60)

    if resume and not a.no_backup and os.path.abspath(out) == os.path.abspath(a.model):
        shutil.copy2(a.model, a.model + ".prev")
        print("\nAncien modèle conservé : %s.prev" % a.model)
    M.save_model(out)
    val = rep.get("val_loss")
    print("\nTerminé en %.1f min : %d étapes | perte %.3f | validation %s%s" % (
        (time.time() - t0) / 60, rep["steps"], rep["loss"], "%.3f" % val if val is not None else "-",
        " | corpus plafonné" if rep.get("corpus_truncated") else ""))
    print("Modèle enregistré : %s (contexte %d)\n" % (out, M.info["ctx"]))

    print("Essai rapide :")
    for q in ["Bonjour", "Combien font 17 fois 23 ?"]:
        r = M.askInto(q, temperature=0, seed=1)
        outils = " [%s]" % ", ".join("%s: %s -> %s" % (c["tool"], c["input"].replace("\n", " ; "), c["output"]) for c in r.tool_calls) if r.tool_calls else ""
        print("  > %s\n    %s%s" % (q, r or "(réponse vide)", outils))


if __name__ == "__main__":
    main()
