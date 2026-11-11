#!/usr/bin/env python3
"""Exemple de chat poli avec airtel (apres fine-tuning LoRA et/ou DPO).

    python3 examples/chat_poli.py                            # utilise airtel-2.1m-poli-dpo (sinon -poli, sinon airtel-2.1m)
    python3 examples/chat_poli.py --model airtel-2.1m --adapter adapters/poli
    python3 examples/chat_poli.py --compare                   # meme question : modele de base vs modele poli
"""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from gopu_lm.airtel import Airtel

QUESTIONS = ["Bonjour", "Peux-tu m'aider ?", "Combien font 17 fois 23 ?", "Je ne comprends pas", "Merci beaucoup", "Au revoir"]


def clean(t):
    import re
    t = re.sub(r"<\|thinking\|>.*?</\|thinking\|>", "", t, flags=re.S)
    return t.replace("<|assistant|>", "").replace("</|assistant|>", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--adapter")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    model = a.model or next((m for m in ("airtel-2.1m-poli-dpo", "airtel-2.1m-poli", "airtel-2.1m") if os.path.isdir(m)), "airtel-2.1m")
    m = Airtel(model, adapter=a.adapter)
    base = Airtel("airtel-2.1m") if a.compare else None
    print("Modele : %s%s\n(Ctrl-D pour quitter ; sans entree, une démonstration est jouée)\n" % (model, " + " + a.adapter if a.adapter else ""))
    qs = QUESTIONS if sys.stdin.isatty() is False or a.compare else None
    def turn(q):
        if base:
            print("  base  : %s" % clean(base.ask(q, temperature=0)))
        print("  poli  : %s" % clean(m.ask(q, temperature=0)))
    if qs:
        for q in qs:
            print("> " + q); turn(q); print()
        return
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if q:
            turn(q)


if __name__ == "__main__":
    main()
