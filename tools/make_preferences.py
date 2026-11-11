#!/usr/bin/env python3
"""Notation des reponses et construction des paires de preferences (etape "humain ou autre modele note").

Trois modes de notation :
  --judge auto   (defaut) juge a regles en Python pur : politesse, exactitude des calculs, format, repetitions
  --judge human  tu choisis toi-meme la meilleure et la pire reponse au clavier
  --ratings f    un fichier JSONL externe (humain ou autre modele) : {"prompt":..., "scores":[s1,s2,...]}

Entree : sortie de `airtel sample` (JSONL {"prompt","answers":[...]}). Sortie : JSONL {"prompt","chosen","rejected"}.
    airtel sample --model airtel-2.1m-poli --prompts finetune/prompts.txt --n 4 --out samples.jsonl
    python3 tools/make_preferences.py samples.jsonl --out finetune/prefs_model.jsonl
"""
import argparse, json, re, sys

POLITE = ["bonjour", "bonsoir", "salut", "merci", "s'il vous plaît", "s'il te plaît", "avec plaisir", "je vous en prie",
          "bien sûr", "ravi", "enchanté", "n'hésitez pas", "hello", "thank", "welcome", "please", "happy to help", "of course", "pas de souci", "désolé"]
RUDE = ["quoi ?", "ça ne te regarde", "fais-le toi-même", "tchao", "whatever", "not my problem", "et alors", "je m'en fiche", "ferme", "idiot", "stupide"]

LEX = set()

def load_lexicon(path):
    """Mots "attendus" (fluidite) : tous les mots du corpus de dialogues de reference."""
    try:
        LEX.update(re.findall(r"\w+", open(path, encoding="utf-8").read().lower()))
    except OSError:
        pass

def visible(ans):
    """Texte de la reponse sans bloc de reflexion ni balises."""
    t = re.sub(r"<\|thinking\|>.*?</\|thinking\|>", "", ans, flags=re.S)
    t = re.sub(r"<\|tools=\".*?</\|tools>", " ", t, flags=re.S)
    return re.sub(r"</?\|[a-z_=\"]+\|?>", " ", t)

def score(prompt, ans):
    a = ans.lower()
    s = 0.0
    s += 1.0 * min(2, sum(p in a for p in POLITE))
    if LEX:
        ws = re.findall(r"\w+", (visible(ans) + " " + re.sub(r"<\|thinking\|>|</\|thinking\|>", " ", "".join(re.findall(r"<\|thinking\|>(.*?)</\|thinking\|>", ans, flags=re.S)))).lower())
        if ws:
            s -= 6.0 * sum(w not in LEX for w in ws) / len(ws)
    if ans.count("</|tools>") > 1 or ans.count("<|tools=\"") > 1:
        s -= 2
    if "</|assistant|>" not in ans:
        s -= 1
    s -= 2.0 * sum(r in a for r in RUDE)
    if "</|assistant|>" in ans or "<|assistant|>" in ans:
        s += 0.5  # format complet
    words = re.findall(r"\w+", a)
    if len(words) < 3:
        s -= 2
    if words and len(set(words)) / len(words) < 0.5:
        s -= 2  # repetitions
    # exactitude : "Combien font A op B" -> le resultat doit apparaitre
    m = re.search(r"(\d+)\s*(plus|moins|fois|\+|-|\*|x)\s*(\d+)", prompt.lower())
    if m:
        x, op, y = int(m.group(1)), m.group(2), int(m.group(3))
        res = x + y if op in ("plus", "+") else x - y if op in ("moins", "-") else x * y
        good_call = ("print(%d %s %d)" % (x, {"plus": "+", "moins": "-", "fois": "*", "x": "*"}.get(op, op), y)) in ans
        said = re.search(r"font %d\b|^%d$|\b%d\.\s" % (res, res, res), visible(ans) + " " + ans, flags=re.M) is not None
        s += (3.0 if said else 0.0) + (1.5 if good_call else -1.5)
        nums = set(int(n) for n in re.findall(r"(?<![\d-])\d+(?!\d)", visible(ans))) - {x, y}
        if nums and res not in nums:
            s -= 3.0  # annonce un autre resultat
    return s

def human(prompt, answers):
    print("\n=== %s" % prompt)
    for i, a in enumerate(answers):
        print("[%d] %s\n" % (i, a))
    try:
        b = int(input("meilleure (numero) : ")); w = int(input("pire (numero) : "))
    except (ValueError, EOFError):
        return None
    return (b, w) if b != w and 0 <= b < len(answers) and 0 <= w < len(answers) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples"); ap.add_argument("--out", required=True)
    ap.add_argument("--judge", choices=["auto", "human"], default="auto")
    ap.add_argument("--ratings", help="JSONL externe {prompt, scores:[...]} (prioritaire)")
    ap.add_argument("--lexicon", default="finetune/polite.txt")
    ap.add_argument("--min-gap", type=float, default=1.0, help="ecart minimal de score entre la meilleure et la pire")
    a = ap.parse_args()
    load_lexicon(a.lexicon)
    ext = {}
    if a.ratings:
        for l in open(a.ratings, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); ext[r["prompt"]] = r["scores"]
    n = 0
    with open(a.samples, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as out:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l); p, ans = r["prompt"], r["answers"]
            if len(set(ans)) < 2:
                continue
            if p in ext and len(ext[p]) == len(ans):
                sc = ext[p]
            elif a.judge == "human":
                hw = human(p, ans)
                if not hw:
                    continue
                sc = [0] * len(ans); sc[hw[0]] = 1; sc[hw[1]] = -1
            else:
                sc = [score(p, x) for x in ans]
            b, w = max(range(len(ans)), key=lambda i: sc[i]), min(range(len(ans)), key=lambda i: sc[i])
            if sc[b] - sc[w] < a.min_gap:
                continue
            fmt = lambda t: t if t.rstrip().endswith("<|eos|>") else t.rstrip() + "\n<|eos|>"
            out.write(json.dumps({"prompt": p, "chosen": fmt(ans[b]), "rejected": fmt(ans[w])}, ensure_ascii=False) + "\n")
            n += 1
    print("%d paires ecrites -> %s" % (n, a.out))

if __name__ == "__main__":
    main()
