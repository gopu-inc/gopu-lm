#!/usr/bin/env python3
"""Lance TOUT l'entraînement d'airtel avec un fichier train.txt (bibliothèque standard Python + binaire Rust `airtel`).

    python3 lancer_entrainement.py train.txt                 # pipeline complet
    python3 lancer_entrainement.py train.txt --steps 800     # plus court
    python3 lancer_entrainement.py train.txt --name airtel-v2 --max-minutes 45

Étapes : conversion au format gopu -> modèle + tokenizer -> pré-entraînement -> LoRA poli -> échantillons + notes -> DPO.
Sorties (dossiers prêts pour Hugging Face) : <name>/, <name>-poli/, <name>-poli-dpo/ + adapters/.
Les dossiers existants du même nom NE sont PAS écrasés (le script s'arrête : choisis un autre --name).

Réglages demandés : contexte 20 000, plafond d'entraînement 1 000 000 de jetons.
Prérequis : binaire compilé :  cargo build --release --bin airtel   (dans le dossier du projet gopu)
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT_TOKENS = 20000
MAX_TRAIN_TOKENS = 1_000_000


def find_bin():
    for c in (os.environ.get("GOPU_AIRTEL_BIN"), os.path.join(HERE, "target", "release", "airtel"),
              os.path.expanduser("~/gopu/target/release/airtel")):
        if c and os.path.isfile(c):
            return c
    sys.exit("binaire 'airtel' introuvable. Compile-le :  cd %s && cargo build --release --bin airtel" % HERE)


def run(*a):
    print("\n$ " + " ".join(map(str, a)), flush=True)
    subprocess.check_call([str(x) for x in a], cwd=HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train", help="fichier brut <|user|>/<|assistant|>/<|eos|> (ex. train.txt)")
    ap.add_argument("--name", default="airtel-2.1m-v2", help="nom du dossier modèle de base (défaut airtel-2.1m-v2)")
    ap.add_argument("--steps", type=int, default=1500, help="étapes de pré-entraînement (1500 ≈ 4 passages sur 1M de jetons)")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--max-minutes", type=float, default=0, help="borne de temps du pré-entraînement (0 = aucune)")
    ap.add_argument("--no-french", action="store_true", help="ne pas mélanger data/train.txt (corpus français du projet)")
    ap.add_argument("--thinking", choices=["template", "none"], default="template",
                    help="template (défaut) : ajoute un court bloc <|thinking|> aux conversations ; none : user/assistant seulement")
    ap.add_argument("--skip-lora", action="store_true")
    ap.add_argument("--skip-dpo", action="store_true")
    a = ap.parse_args()

    for need in ("tools/convert_dataset.py", "tools/gen_polite.py", "tools/make_preferences.py"):
        if not os.path.isfile(os.path.join(HERE, need)):
            sys.exit("'%s' introuvable : ce script doit être placé et lancé dans le dossier du projet gopu (celui qui contient tools/, "
                     "src/, target/), pas dans le dossier du modèle. Exemple : cp lancer_entrainement.py ~/gopu/ && cd ~/gopu" % need)
    binp = find_bin()
    raw = os.path.abspath(a.train)
    if not os.path.isfile(raw):
        sys.exit("fichier introuvable : " + raw)
    base, poli, dpo = a.name, a.name + "-poli", a.name + "-poli-dpo"
    for d in (base, poli, dpo):
        if os.path.exists(os.path.join(HERE, d)):
            sys.exit("'%s' existe déjà : choisis un autre --name pour ne rien écraser." % d)
    for d in ("data", "finetune", "adapters", "work"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)

    conv = os.path.join("work", "en_converted.txt")
    run(sys.executable, "tools/convert_dataset.py", raw, "--out", conv, "--max-chars", 1500, "--max-convs", 5000, "--thinking", a.thinking)
    corpus = os.path.join("work", "corpus.txt")
    with open(os.path.join(HERE, corpus), "w", encoding="utf-8") as out:
        fr = os.path.join(HERE, "data", "train.txt")
        if not a.no_french and os.path.isfile(fr):
            out.write(open(fr, encoding="utf-8").read().rstrip("\n") + "\n")
            print("mélange avec data/train.txt (corpus français du projet)")
        out.write(open(os.path.join(HERE, conv), encoding="utf-8").read())

    run(binp, "init", "--data", corpus, "--out", base, "--vocab", a.vocab, "--ctx", CONTEXT_TOKENS)
    run(binp, "train", "--model", base, "--data", corpus, "--out", base, "--steps", a.steps, "--batch", 8, "--pack-len", 512,
        "--lr", "3e-3", "--warmup", 60, "--holdout-every", 40, "--eval-every", 500, "--log-every", 25,
        "--max-corpus-tokens", MAX_TRAIN_TOKENS, "--max-seconds", int(a.max_minutes * 60))
    cur = base
    if not a.skip_lora:
        run(sys.executable, "tools/gen_polite.py", "--n", 1500)
        run(binp, "lora", "--model", cur, "--data", "finetune/polite.txt", "--adapter-out", "adapters/" + poli, "--merge-out", poli,
            "--r", 16, "--alpha", 32, "--targets", "all", "--steps", 250, "--batch", 8, "--pack-len", 256, "--lr", "2e-3",
            "--holdout-every", 30)
        cur = poli
    if not a.skip_dpo and not a.skip_lora:
        run(binp, "sample", "--model", cur, "--prompts", "finetune/prompts.txt", "--n", 6, "--temperature", 1.0, "--top-k", 0,
            "--top-p", 1.0, "--repetition-penalty", 1.0, "--max-tokens", 110, "--out", "finetune/samples.jsonl")
        run(sys.executable, "tools/make_preferences.py", "finetune/samples.jsonl", "--out", "finetune/prefs_model.jsonl")
        with open(os.path.join(HERE, "finetune", "prefs_mix.jsonl"), "w", encoding="utf-8") as o:
            o.write(open(os.path.join(HERE, "finetune", "prefs_model.jsonl"), encoding="utf-8").read())
            o.writelines(open(os.path.join(HERE, "finetune", "prefs.jsonl"), encoding="utf-8").readlines()[:300])
        run(binp, "dpo", "--model", cur, "--pairs", "finetune/prefs_mix.jsonl", "--adapter-out", "adapters/" + dpo,
            "--merge-out", dpo, "--targets", "all", "--r", 16)
        cur = dpo
    print("\n=== Terminé ===\nModèle final : %s/  (model.safetensors, model.b8n, config.json, tokenizer…)" % cur)
    print("Essai : cd %s && python3 chat.py   (ou : %s ask --model %s --prompt Bonjour)" % (cur, binp, cur))


if __name__ == "__main__":
    main()

