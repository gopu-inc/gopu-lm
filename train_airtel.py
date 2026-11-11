#!/usr/bin/env python3
"""Pipeline complet airtel-2.1m (Rust pur + bibliotheque standard Python) :

  1. convertit un corpus brut en format gopu      (tools/convert_dataset.py)
  2. cree le modele + tokenizer BPE                (airtel init)
  3. pre-entraine (contexte 20 000, plafond 1 000 000 jetons)  (airtel train)
  4. fine-tuning LoRA "poli"                       (airtel lora)
  5. genere plusieurs reponses, les note, DPO      (airtel sample + tools/make_preferences.py + airtel dpo)

    python3 train_airtel.py --raw train.txt                # tout
    python3 train_airtel.py --skip-pretrain --steps 300    # reprend a partir de airtel-2.1m/
Sur telephone (Termux) compter ~1 h pour 1500 etapes ; --max-seconds borne chaque etape.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("GOPU_AIRTEL_BIN") or os.path.join(HERE, "target", "release", "airtel")
CONTEXT_TOKENS = 20000
MAX_TRAIN_TOKENS = 1_000_000


def run(*a):
    print("\n$", " ".join(map(str, a)), flush=True)
    subprocess.check_call([str(x) for x in a], cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="corpus brut <|user|>/<|assistant|>/<|eos|> a convertir (facultatif)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--skip-pretrain", action="store_true")
    ap.add_argument("--skip-lora", action="store_true")
    ap.add_argument("--skip-dpo", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit("binaire absent : cargo build --release --bin airtel")
    os.makedirs("data", exist_ok=True)
    if a.raw:
        run(sys.executable, "tools/convert_dataset.py", a.raw, "--out", "data/en_converted.txt", "--max-chars", 1500, "--max-convs", 5000)
    corpus = "build_corpus.txt"
    with open(os.path.join(HERE, corpus), "w", encoding="utf-8") as out:
        for f in sorted(os.listdir(os.path.join(HERE, "data"))):
            if f.endswith(".txt"):
                out.write(open(os.path.join(HERE, "data", f), encoding="utf-8").read().rstrip("\n") + "\n")
    if not a.skip_pretrain:
        run(BIN, "init", "--data", corpus, "--out", "airtel-2.1m", "--vocab", 4096, "--ctx", CONTEXT_TOKENS)
        run(BIN, "train", "--model", "airtel-2.1m", "--data", corpus, "--out", "airtel-2.1m", "--steps", a.steps, "--batch", 8,
            "--pack-len", 512, "--lr", "3e-3", "--warmup", 60, "--holdout-every", 40, "--eval-every", 500,
            "--max-corpus-tokens", MAX_TRAIN_TOKENS, "--max-seconds", a.max_seconds)
    cur = "airtel-2.1m"
    if not a.skip_lora:
        run(sys.executable, "tools/gen_polite.py", "--n", 1500)
        run(BIN, "lora", "--model", cur, "--data", "finetune/polite.txt", "--adapter-out", "adapters/poli", "--merge-out", "airtel-2.1m-poli",
            "--r", 16, "--alpha", 32, "--targets", "all", "--steps", 250, "--batch", 8, "--pack-len", 256, "--lr", "2e-3", "--holdout-every", 30)
        cur = "airtel-2.1m-poli"
    if not a.skip_dpo:
        run(BIN, "sample", "--model", cur, "--prompts", "finetune/prompts.txt", "--n", 6, "--temperature", 1.0, "--top-k", 0, "--top-p", 1.0,
            "--repetition-penalty", 1.0, "--max-tokens", 110, "--out", "finetune/samples.jsonl")
        run(sys.executable, "tools/make_preferences.py", "finetune/samples.jsonl", "--out", "finetune/prefs_model.jsonl")
        with open(os.path.join(HERE, "finetune", "prefs_mix.jsonl"), "w", encoding="utf-8") as o:
            o.write(open(os.path.join(HERE, "finetune", "prefs_model.jsonl"), encoding="utf-8").read())
            o.writelines(open(os.path.join(HERE, "finetune", "prefs.jsonl"), encoding="utf-8").readlines()[:300])
        run(BIN, "dpo", "--model", cur, "--pairs", "finetune/prefs_mix.jsonl", "--adapter-out", "adapters/dpo",
            "--merge-out", "airtel-2.1m-poli-dpo", "--targets", "all", "--r", 16)
    print("\nTermine. Essai : python3 examples/chat_poli.py --compare")


if __name__ == "__main__":
    main()
