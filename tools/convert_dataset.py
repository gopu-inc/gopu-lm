#!/usr/bin/env python3
"""Convertit un corpus <|user|>/<|assistant|>/<|eos|> au format de chat gopu (avec bloc <|thinking|>).

    python3 tools/convert_dataset.py brut.txt --out data/en_converted.txt [--max-chars 1800] [--max-convs 6000]

- coupe sur <|eos|>, ignore les conversations vides, trop longues ou mal formees
- ajoute un court bloc <|thinking|> (gabarit selon le type de demande) entre </|user|> et <|assistant|>
- garantit que chaque reponse se termine par </|assistant|> puis <|eos|>
Aucune dependance : bibliotheque standard seulement.
"""
import argparse, hashlib, random, re

OPEN, CLOSE = "<|assistant|>", "</|assistant|>"

def kind(q):
    s = q.lower()
    if re.search(r"```|\b(code|python|function|rust|javascript|java|c\+\+|sql|script|program|bug|compile)\b", s):
        return "code"
    if re.search(r"\b(translate|traduis|in french|in spanish|in german)\b", s):
        return "translate"
    if re.search(r"\b(write|compose|story|poem|essay|email|letter|draft)\b", s):
        return "write"
    if re.search(r"\b(list|give me|suggest|ideas|examples|recommend)\b", s):
        return "list"
    if re.search(r"\b(explain|why|how does|how do|what is|what are|difference|describe|meaning)\b", s) or "?" in s:
        return "explain"
    return "other"

T = {
 "code": ["The user wants help with code. I will give a clear, correct solution and explain the key points.",
          "This is a programming request. I should answer precisely and keep the code readable."],
 "translate": ["The user wants a translation. I will keep the meaning and the tone of the original."],
 "write": ["The user asks me to write something. I will follow the requested form and keep it well organised."],
 "list": ["The user wants a list of items. I will give relevant, distinct points."],
 "explain": ["The user asks for an explanation. I will answer directly, then add the useful details.",
             "This is a question. I will answer it clearly and politely, without inventing facts."],
 "other": ["I will read the request carefully and answer it helpfully and politely."],
}

def think(q, i):
    opts = T[kind(q)]
    h = int(hashlib.md5((q[:80] + str(i)).encode()).hexdigest(), 16)
    return opts[h % len(opts)]

def convert(conv, max_chars, i):
    c = conv.strip()
    if not c.startswith("<|user|>") or c.count("<|user|>") != 1 or OPEN not in c:
        return None
    if len(c) > max_chars or "<|thinking|>" in c or "<|tools" in c:
        return None
    u_end = c.find("</|user|>")
    a_start = c.find(OPEN)
    if u_end < 0 or a_start < u_end:
        return None
    q = c[len("<|user|>"):u_end].strip()
    a = c[a_start + len(OPEN):].replace(CLOSE, "").strip()
    if len(q) < 2 or len(a) < 2:
        return None
    return "<|user|>\n%s\n</|user|>\n<|thinking|>\n%s\n</|thinking|>\n<|assistant|>\n%s\n</|assistant|>\n<|eos|>\n" % (q, think(q, i), a)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--max-convs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    raw = open(a.src, encoding="utf-8").read().split("<|eos|>")
    out = [x for x in (convert(c, a.max_chars, i) for i, c in enumerate(raw)) if x]
    random.Random(a.seed).shuffle(out)
    if a.max_convs:
        out = out[:a.max_convs]
    open(a.out, "w", encoding="utf-8").write("".join(out))
    print("%d conversations lues -> %d converties (%d caracteres) : %s" % (len(raw), len(out), sum(map(len, out)), a.out))

if __name__ == "__main__":
    main()
