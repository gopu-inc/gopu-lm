#!/usr/bin/env python3
"""Genere (1) un corpus de dialogues polis pour le fine-tuning LoRA, (2) des paires de preferences {prompt, chosen, rejected}.

    python3 tools/gen_polite.py --out-sft data/polite.txt --out-prefs data/prefs.jsonl --n 1500

Bibliotheque standard uniquement. Reponses "chosen" : polies, utiles, honnetes ; "rejected" : seches, desinvoltes ou fausses.
"""
import argparse, json, random

NAMES = ["Alice", "Moussa", "Chantal", "Paul", "Sofia", "Kevin"]
GREET = [("Bonjour", "Bonjour ! Comment puis-je vous aider aujourd'hui ?", "Quoi ?"),
         ("Salut", "Salut ! Ravi de vous voir. Que puis-je faire pour vous ?", "Ouais, quoi."),
         ("Hello", "Hello ! How can I help you today?", "What."),
         ("Bonsoir", "Bonsoir ! Avec plaisir, de quoi avez-vous besoin ?", "Parle."),
         ("Merci beaucoup", "Avec plaisir ! N'hésitez pas si vous avez d'autres questions.", "Ouais."),
         ("Merci !", "Je vous en prie, c'est un plaisir de vous aider.", "De rien, passe ton chemin."),
         ("Thank you", "You're welcome! Feel free to ask if you need anything else.", "Yeah, whatever."),
         ("Au revoir", "Au revoir et bonne journée ! À bientôt.", "Tchao."),
         ("Comment ça va ?", "Je vais bien, merci de demander ! Et vous, comment puis-je vous aider ?", "Ça ne te regarde pas."),
         ("How are you?", "I'm doing well, thank you for asking! How can I help you?", "Why do you care?")]
HELP = [("Peux-tu m'aider ?", "Bien sûr, avec plaisir ! Dites-moi de quoi il s'agit.", "Non."),
        ("J'ai besoin d'aide pour mon devoir", "Avec plaisir ! Expliquez-moi le sujet et ce qui vous pose problème, et nous le verrons ensemble.", "Fais-le toi-même."),
        ("Can you help me?", "Of course, I'd be happy to help. What do you need?", "No."),
        ("Je suis fatigué aujourd'hui", "Je suis désolé de l'entendre. Prenez le temps de vous reposer ; puis-je faire quelque chose pour vous ?", "Et alors ?"),
        ("Je ne comprends pas", "Pas de souci, c'est normal. Pouvez-vous me dire quelle partie n'est pas claire ? Je vais réexpliquer plus simplement.", "C'est pas mon problème."),
        ("I don't understand", "No problem at all. Tell me which part is unclear and I'll explain it in a simpler way.", "Not my problem.")]
OPS = [("plus", "+"), ("moins", "-"), ("fois", "*")]

def calc(r):
    a, b = r.randint(2, 99), r.randint(2, 99)
    w, sym = r.choice(OPS)
    res = eval("%d %s %d" % (a, sym, b))
    q = r.choice(["Combien font %d %s %d ?", "Calcule %d %s %d, s'il te plaît", "Peux-tu calculer %d %s %d ?"]) % (a, w, b)
    code = "print(%d %s %d)" % (a, sym, b)
    good = ("<|thinking|>\nL'utilisateur veut un calcul. J'utilise l'outil python pour être exact, puis je réponds poliment.\n</|thinking|>\n"
            "<|assistant|>\n<|tools=\"python\">\n%s\n</|tools>\n<|tool_result>\n%d\n</|tool_result>\n"
            "Avec plaisir ! %d %s %d font %d.\n</|assistant|>" % (code, res, a, w, b, res))
    wrong = res + r.choice([-10, -1, 1, 2, 10, 100])
    bad = "<|thinking|>\nCalcul rapide.\n</|thinking|>\n<|assistant|>\nC'est %d, évidemment.\n</|assistant|>" % wrong
    return q, good, bad

def simple(q, good, bad):
    g = "<|thinking|>\nJe réponds de façon polie, claire et utile.\n</|thinking|>\n<|assistant|>\n%s\n</|assistant|>" % good
    b = "<|thinking|>\nPeu importe.\n</|thinking|>\n<|assistant|>\n%s\n</|assistant|>" % bad
    return q, g, b

def sample(r):
    k = r.random()
    if k < 0.35: return simple(*r.choice(GREET))
    if k < 0.55: return simple(*r.choice(HELP))
    if k < 0.7:
        n = r.choice(NAMES)
        return simple("Je m'appelle %s" % n, "Enchanté, %s ! Ravi de faire votre connaissance. Comment puis-je vous aider ?" % n, "Et moi je m'en fiche.")
    return calc(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-sft", default="finetune/polite.txt")
    ap.add_argument("--out-prefs", default="finetune/prefs.jsonl")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    r = random.Random(a.seed)
    sft, prefs = [], []
    for _ in range(a.n):
        q, good, bad = sample(r)
        sft.append("<|user|>\n%s\n</|user|>\n%s\n<|eos|>\n" % (q, good))
        c = good.split("<|assistant|>\n", 1)[1] if False else good
        prefs.append({"prompt": q, "chosen": good + "\n<|eos|>", "rejected": bad + "\n<|eos|>"})
    open(a.out_sft, "w", encoding="utf-8").write("".join(sft))
    with open(a.out_prefs, "w", encoding="utf-8") as f:
        for p in prefs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("SFT : %d dialogues -> %s | preferences : %d paires -> %s" % (len(sft), a.out_sft, len(prefs), a.out_prefs))

if __name__ == "__main__":
    main()
