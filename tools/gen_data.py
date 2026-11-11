#!/usr/bin/env python3
"""Genere un corpus d'entrainement au format de chat gopu (francais) : data/train.txt

Format d'une conversation (chaque conversation se termine par <|eos|>) :

    <|user|>
    Bonjour
    </|user|>
    <|thinking|>
    ... reflexion ...
    </|thinking|>
    <|assistant|>
    Bonjour, comment puis-je vous aider aujourd'hui ?
    </|assistant|>
    <|eos|>

Appel d'outil dans la reponse (le resultat est injecte par le programme, pas par le modele) :

    <|assistant|>
    <|tools="python">
    print(17 * 23)
    </|tools>
    <|tool_result>
    391
    </|tool_result>
    17 fois 23 font 391.
    </|assistant|>

Usage : python3 tools/gen_data.py [--n 1400] [--seed 7] [--out data/train.txt]
"""
import argparse
import random

R = random.Random()


def conv(user, think, answer, tool=None):
    """tool = (nom, code, resultat) ou None."""
    s = f"<|user|>\n{user}\n</|user|>\n<|thinking|>\n{think}\n</|thinking|>\n<|assistant|>\n"
    if tool:
        name, code, res = tool
        s += f'<|tools="{name}">\n{code}\n</|tools>\n<|tool_result>\n{res}\n</|tool_result>\n'
    s += f"{answer}\n</|assistant|>\n<|eos|>\n"
    return s


def two_turns(u1, t1, a1, u2, t2, a2):
    return (
        f"<|user|>\n{u1}\n</|user|>\n<|thinking|>\n{t1}\n</|thinking|>\n<|assistant|>\n{a1}\n</|assistant|>\n"
        f"<|user|>\n{u2}\n</|user|>\n<|thinking|>\n{t2}\n</|thinking|>\n<|assistant|>\n{a2}\n</|assistant|>\n<|eos|>\n"
    )


# ---------------------------------------------------------------- salutations

GREETINGS = ["Bonjour", "Salut", "Bonsoir", "Coucou", "Hello", "Hey", "Bonjour !", "Salut !", "Bonsoir !", "Yo"]
GREET_ANS = [
    "Bonjour, comment puis-je vous aider aujourd'hui ?",
    "Bonjour ! Comment puis-je vous aider ?",
    "Salut ! Que puis-je faire pour vous ?",
    "Bonjour, que puis-je faire pour vous aujourd'hui ?",
]


def greetings():
    g = R.choice(GREETINGS)
    word = g.rstrip(" !")
    lang = "anglais" if word in ("Hello", "Hey") else "français"
    if word == "Bonsoir":
        think = f"L'utilisateur dit \"{word}\", un mot français qui salue le soir. Je dois répondre de manière polie et gentille, puis proposer mon aide."
        ans = R.choice(["Bonsoir, comment puis-je vous aider ce soir ?", "Bonsoir ! Que puis-je faire pour vous ?"])
    elif lang == "anglais":
        think = f"L'utilisateur dit \"{word}\", un salut en anglais. Je réponds poliment en français et je propose mon aide."
        ans = R.choice(GREET_ANS)
    else:
        think = f"L'utilisateur dit \"{word}\", un mot français pour saluer. Je dois répondre de manière polie et gentille, puis proposer mon aide."
        ans = R.choice(GREET_ANS)
    return conv(g, think, ans)


def how_are_you():
    q = R.choice(["Comment ça va ?", "Ça va ?", "Comment vas-tu ?", "Tu vas bien ?", "Comment allez-vous ?"])
    think = "L'utilisateur me demande comment je vais. Je suis un programme, mais je réponds gentiment et je lui demande ce dont il a besoin."
    ans = R.choice([
        "Je vais bien, merci ! Et vous, comment puis-je vous aider ?",
        "Tout va bien, merci de demander. Que puis-je faire pour vous ?",
    ])
    return conv(q, think, ans)


def identity():
    q = R.choice(["Qui es-tu ?", "Tu es qui ?", "Comment tu t'appelles ?", "Présente-toi.", "Quel est ton nom ?", "Tu es quoi ?"])
    think = "L'utilisateur veut savoir qui je suis. Je me présente comme un petit assistant basé sur le modèle gopu, écrit en Rust."
    ans = R.choice([
        "Je suis un petit assistant basé sur le modèle gopu, écrit en Rust.",
        "Je m'appelle gopu, un petit modèle de langage écrit en Rust. Je peux discuter et utiliser des outils.",
    ])
    return conv(q, think, ans)


def abilities():
    q = R.choice(["Que sais-tu faire ?", "Tu peux m'aider à quoi ?", "À quoi tu sers ?", "Quelles sont tes capacités ?"])
    think = "L'utilisateur demande ce que je sais faire. Je décris simplement mes capacités : discuter, calculer avec un outil et donner l'heure."
    ans = "Je peux discuter en français, faire des calculs avec l'outil python et donner l'heure. Je reste un petit modèle, donc mes connaissances sont limitées."
    return conv(q, think, ans)


def thanks():
    q = R.choice(["Merci", "Merci beaucoup !", "Merci !", "Super, merci", "Thanks"])
    think = "L'utilisateur me remercie. Je réponds poliment que c'est avec plaisir."
    ans = R.choice(["Avec plaisir ! N'hésitez pas si vous avez d'autres questions.", "De rien ! Je reste disponible si vous avez besoin d'aide."])
    return conv(q, think, ans)


def bye():
    q = R.choice(["Au revoir", "À bientôt", "Salut, à plus", "Bonne journée", "Bonne soirée", "Ciao"])
    think = "L'utilisateur prend congé. Je lui dis au revoir poliment."
    ans = R.choice(["Au revoir, à bientôt !", "À bientôt ! Passez une excellente journée.", "Au revoir ! Revenez quand vous voulez."])
    return conv(q, think, ans)


# ---------------------------------------------------------------- calculs avec outils

def num(lo=2, hi=999):
    return R.randint(lo, hi)


def math_python():
    kind = R.choice(["mul", "mul", "add", "sub", "div"])
    if kind == "mul":
        a, b = num(2, 99), num(2, 99)
        q = R.choice([f"Combien font {a} fois {b} ?", f"{a} x {b} = ?", f"Calcule {a} * {b}", f"Peux-tu multiplier {a} par {b} ?", f"Combien fait {a} multiplié par {b} ?"])
        code, res, ans = f"print({a} * {b})", a * b, f"{a} fois {b} font {a * b}."
    elif kind == "add":
        a, b = num(), num()
        q = R.choice([f"Combien font {a} plus {b} ?", f"{a} + {b} = ?", f"Calcule {a} + {b}", f"Additionne {a} et {b}", f"Quelle est la somme de {a} et {b} ?"])
        code, res, ans = f"print({a} + {b})", a + b, f"{a} plus {b} font {a + b}."
    elif kind == "sub":
        a, b = num(10, 999), num(2, 99)
        q = R.choice([f"Combien font {a} moins {b} ?", f"{a} - {b} = ?", f"Calcule {a} - {b}", f"Soustrais {b} de {a}"])
        code, res, ans = f"print({a} - {b})", a - b, f"{a} moins {b} font {a - b}."
    else:
        b = num(2, 20)
        res = num(2, 60)
        a = b * res
        q = R.choice([f"Combien font {a} divisé par {b} ?", f"{a} / {b} = ?", f"Calcule {a} / {b}", f"Divise {a} par {b}"])
        code, ans = f"print({a} // {b})", f"{a} divisé par {b} font {res}."
    think = R.choice([
        "L'utilisateur demande un calcul. Pour être sûr du résultat, j'utilise l'outil python.",
        "C'est une opération arithmétique. Je vais utiliser l'outil python pour obtenir le bon résultat.",
        "L'utilisateur veut le résultat d'un calcul. J'appelle l'outil python plutôt que de deviner.",
    ])
    return conv(q, think, ans, ("python", code, str(res)))


def math_calc():
    a, b = num(2, 99), num(2, 99)
    op, word = R.choice([("+", "plus"), ("-", "moins"), ("*", "fois")])
    expr = f"{a} {op} {b}"
    res = eval(expr)
    q = R.choice([f"Utilise la calculatrice : {expr}", f"Calculatrice : {expr}", f"Avec la calculatrice, combien font {a} {word} {b} ?"])
    think = "L'utilisateur demande la calculatrice. J'appelle l'outil calc avec l'expression."
    return conv(q, think, f"Le résultat est {res}.", ("calc", expr, str(res)))


def count_to():
    n = R.randint(3, 12)
    q = R.choice([f"Compte de 1 à {n}", f"Compte jusqu'à {n}", f"Écris les nombres de 1 à {n}"])
    think = "L'utilisateur veut une suite de nombres. J'utilise l'outil python pour afficher la liste."
    res = " ".join(str(i) for i in range(1, n + 1))
    return conv(q, think, f"Voici la liste : {res}.", ("python", f"print(*range(1, {n + 1}))", res))


def sum_to():
    n = R.randint(3, 60)
    q = R.choice([f"Quelle est la somme des nombres de 1 à {n} ?", f"Additionne tous les entiers de 1 à {n}"])
    think = "L'utilisateur veut la somme d'une suite d'entiers. Je laisse l'outil python la calculer."
    res = sum(range(1, n + 1))
    return conv(q, think, f"La somme de 1 à {n} vaut {res}.", ("python", f"print(sum(range(1, {n + 1})))", str(res)))


def power():
    a, b = R.randint(2, 9), R.randint(2, 6)
    q = R.choice([f"Combien font {a} puissance {b} ?", f"Calcule {a} ** {b}", f"{a} à la puissance {b} ?"])
    think = "L'utilisateur demande une puissance. Je passe par l'outil python pour éviter une erreur."
    res = a ** b
    return conv(q, think, f"{a} puissance {b} font {res}.", ("python", f"print({a} ** {b})", str(res)))


def time_now():
    h, m = R.randint(0, 23), R.randint(0, 59)
    q = R.choice(["Quelle heure est-il ?", "Il est quelle heure ?", "Donne-moi l'heure", "Tu peux me dire l'heure ?"])
    think = "L'utilisateur veut connaître l'heure. Je ne la connais pas moi-même, donc j'appelle l'outil python."
    code = 'from datetime import datetime\nprint(datetime.now().strftime("%H:%M"))'
    return conv(q, think, f"Il est {h:02d}:{m:02d}.", ("python", code, f"{h:02d}:{m:02d}"))


DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def date_today():
    q = R.choice(["Quel jour sommes-nous ?", "On est quel jour ?", "Quel jour de la semaine est-on ?"])
    think = "L'utilisateur demande le jour de la semaine. J'utilise l'outil python pour le connaître."
    d = R.choice(DAYS)
    code = 'from datetime import datetime\njours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]\nprint(jours[datetime.now().weekday()])'
    return conv(q, think, f"Nous sommes {d}.", ("python", code, d))


def reverse_word():
    w = R.choice(["bonjour", "ordinateur", "rust", "gopu", "modèle", "soleil", "maison", "langage", "tokenizer", "attention"])
    q = R.choice([f"Écris le mot {w} à l'envers", f"Inverse le mot {w}", f"Comment s'écrit {w} à l'envers ?"])
    think = "L'utilisateur veut un mot inversé. Je laisse l'outil python inverser les lettres."
    return conv(q, think, f"À l'envers, « {w} » s'écrit « {w[::-1]} ».", ("python", f'print("{w}"[::-1])', w[::-1]))


def length_word():
    w = R.choice(["bonjour", "ordinateur", "rust", "gopu", "modèle", "soleil", "maison", "langage", "tokenizer", "attention", "transformer"])
    q = R.choice([f"Combien de lettres dans le mot {w} ?", f"Quelle est la longueur du mot {w} ?"])
    think = "L'utilisateur demande le nombre de lettres d'un mot. J'utilise l'outil python pour compter."
    return conv(q, think, f"Le mot « {w} » contient {len(w)} lettres.", ("python", f'print(len("{w}"))', str(len(w))))


# ---------------------------------------------------------------- connaissances (memorisees)

CAPITALS = [
    ("la France", "Paris"), ("la Belgique", "Bruxelles"), ("l'Italie", "Rome"), ("l'Espagne", "Madrid"),
    ("l'Allemagne", "Berlin"), ("le Portugal", "Lisbonne"), ("la Suisse", "Berne"), ("le Canada", "Ottawa"),
    ("le Japon", "Tokyo"), ("le Maroc", "Rabat"), ("l'Algérie", "Alger"), ("le Sénégal", "Dakar"),
    ("la Côte d'Ivoire", "Yamoussoukro"), ("le Cameroun", "Yaoundé"), ("la République démocratique du Congo", "Kinshasa"),
    ("le Congo", "Brazzaville"), ("le Rwanda", "Kigali"), ("le Brésil", "Brasilia"), ("l'Égypte", "Le Caire"),
    ("le Kenya", "Nairobi"),
]


def capital():
    c, v = R.choice(CAPITALS)
    de = "du " + c[3:] if c.startswith("le ") else "de " + c
    q = R.choice([f"Quelle est la capitale {de} ?", f"Capitale {de} ?", f"Dis-moi la capitale {de}"])
    think = f"L'utilisateur demande la capitale {de}. C'est une question de culture générale que je connais."
    return conv(q, think, f"La capitale {de} est {v}.")


def code_rust():
    items = [
        ("Écris une fonction Rust qui additionne deux nombres", "fn addition(a: i32, b: i32) -> i32 {\n    a + b\n}"),
        ("Comment afficher du texte en Rust ?", 'fn main() {\n    println!("Bonjour");\n}'),
        ("Écris une fonction Rust qui double un nombre", "fn double(x: i32) -> i32 {\n    x * 2\n}"),
    ]
    q, code = R.choice(items)
    think = "L'utilisateur demande un exemple de code Rust. Je réponds avec un court exemple."
    return conv(q, think, f"Voici un exemple :\n{code}")


def code_python():
    items = [
        ("Écris une fonction Python qui additionne deux nombres", "def addition(a, b):\n    return a + b"),
        ("Comment afficher du texte en Python ?", 'print("Bonjour")'),
        ("Écris une boucle Python de 1 à 5", "for i in range(1, 6):\n    print(i)"),
    ]
    q, code = R.choice(items)
    think = "L'utilisateur demande un exemple de code Python. Je réponds avec un court exemple."
    return conv(q, think, f"Voici un exemple :\n{code}")


def unknown():
    q = R.choice([
        "Quel temps fera-t-il demain ?", "Qui va gagner le prochain match ?", "Quel est le cours de l'action Apple aujourd'hui ?",
        "Quelle est la dernière actualité ?", "Quel est le numéro de téléphone de mon voisin ?",
    ])
    think = "L'utilisateur demande une information que je ne peux pas connaître : elle est récente ou privée. Je dois le dire honnêtement."
    ans = R.choice([
        "Je ne sais pas, je n'ai pas accès à cette information.",
        "Désolé, je ne peux pas le savoir : je n'ai pas accès à des informations en direct.",
    ])
    return conv(q, think, ans)


def multi_turn():
    a, b = num(2, 99), num(2, 99)
    return two_turns(
        R.choice(["Bonjour", "Salut"]),
        "L'utilisateur me salue. Je réponds poliment et je propose mon aide.",
        "Bonjour, comment puis-je vous aider aujourd'hui ?",
        f"Combien font {a} fois {b} ?",
        "L'utilisateur demande maintenant un calcul. J'utilise l'outil python pour être sûr du résultat.",
        f'<|tools="python">\nprint({a} * {b})\n</|tools>\n<|tool_result>\n{a * b}\n</|tool_result>\n{a} fois {b} font {a * b}.',
    )


WEIGHTS = [
    (greetings, 7), (how_are_you, 2), (identity, 3), (abilities, 2), (thanks, 2), (bye, 2),
    (math_python, 22), (math_calc, 5), (count_to, 4), (sum_to, 3), (power, 3), (time_now, 4), (date_today, 2),
    (reverse_word, 4), (length_word, 3), (capital, 6), (code_rust, 2), (code_python, 2), (unknown, 3),
]


def chain(fns):
    """2 ou 3 conversations a la suite dans le meme contexte (un seul <|eos|> a la fin) : apprend a continuer un dialogue."""
    parts = [R.choice(fns)() for _ in range(R.choice([2, 2, 3]))]
    return "".join(p.removesuffix("<|eos|>\n") for p in parts[:-1]) + parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/train.txt")
    a = ap.parse_args()
    R.seed(a.seed)
    fns = [f for f, w in WEIGHTS for _ in range(w)]
    out = []
    for i in range(a.n):
        out.append(multi_turn() if i % 40 == 39 else chain(fns) if i % 10 < 4 else R.choice(fns)())
    # le premier exemple est exactement celui donne dans la demande (avec la grammaire corrigee)
    out.insert(
        0,
        "<|user|>\nBonjour\n</|user|>\n<|thinking|>\nL'utilisateur nous dit \"bonjour\", qui est un mot français. "
        "Nous devons répondre d'une manière polie et gentille, par exemple : bonjour, comment puis-je t'aider aujourd'hui ?\n"
        "</|thinking|>\n<|assistant|>\nBonjour, comment puis-je vous aider aujourd'hui ?\n</|assistant|>\n<|eos|>\n",
    )
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("".join(out))
    print(f"{len(out)} conversations -> {a.out}")


if __name__ == "__main__":
    main()
