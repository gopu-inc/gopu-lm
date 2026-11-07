#!/usr/bin/env python3
"""Construit un corpus d'apprentissage distinct sans toucher aux donnees existantes.

Il reprend le corpus de base et ajoute toutes les formulations du curriculum
general de ``expand_data.py``. Les reponses de l'evaluation restent separees.
"""
import argparse
from pathlib import Path

from expand_data import FACTS, conversation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/train.txt")
    p.add_argument("--out", default="data/improved/train.txt")
    args = p.parse_args()
    source = Path(args.source).read_text(encoding="utf-8")
    additions = []
    for _, answer, questions in FACTS:
        for question in questions:
            item = conversation(question, answer)
            if item not in source:
                additions.append(item)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source.rstrip() + "\n\n" + "".join(additions), encoding="utf-8")
    print(f"{len(additions)} nouvelles conversations ; corpus : {out}")


if __name__ == "__main__":
    main()
