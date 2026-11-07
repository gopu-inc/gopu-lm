#!/usr/bin/env python3
"""Benchmark reproductible, separe du corpus d'entrainement.

Les mots attendus sont volontairement simples : ce test mesure les capacites
utiles du petit modele, sans pretendre remplacer une evaluation humaine.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from gopu_lm import Model


CASES = [
    ("Bonjour, peux-tu me guider ?", ["bonjour"]),
    ("Peux-tu te présenter brièvement ?", ["gopu"]),
    ("Quelle ville est la capitale japonaise ?", ["tokyo"]),
    ("Explique simplement la différence entre météo et climat.", ["court", "long"]),
    ("À quoi sert un trait dans Rust ?", ["comportement"]),
    ("Quelle est l'idée de Option<T> en Rust ?", ["none"]),
    ("Comment calcule-t-on une moyenne arithmétique ?", ["somme", "nombre"]),
    ("Comment calculer l'aire d'un rectangle ?", ["longueur", "largeur"]),
    ("Pourquoi faut-il vérifier les unités avant un calcul ?", ["compatibles"]),
    ("Que faire lorsque tu n'es pas certain d'une information récente ?", ["limite"]),
    ("Explique en une phrase ce qu'est une liste Python.", ["liste"]),
    ("Pourquoi le ciel est-il bleu ?", ["diffusion"]),
    ("Résume le principe de la photosynthèse.", ["lumière"]),
    ("Quelle était l'idée principale de la Révolution française ?", ["révolution"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="modele_ameliore.b8n")
    ap.add_argument("--out", default="data/eval.jsonl")
    ap.add_argument("--write-only", action="store_true")
    args = ap.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"prompt": q, "keywords": words}, ensure_ascii=False) + "\n" for q, words in CASES), encoding="utf-8")
    if args.write_only:
        print(f"{len(CASES)} cas écrits dans {path}")
        return
    model = Model.load_model(args.model)
    passed = 0
    for i, (question, words) in enumerate(CASES, 1):
        answer = str(model.askInto(question, temperature=0, seed=17, max_tokens=100, tools="auto"))
        ok = all(word.casefold() in answer.casefold() for word in words)
        passed += ok
        print(f"{'OK' if ok else 'ECHEC'} {i:>2}/{len(CASES)} | {question}\n  {answer}")
    print(f"\nScore : {passed}/{len(CASES)} ({100 * passed / len(CASES):.1f} %)")


if __name__ == "__main__":
    main()
