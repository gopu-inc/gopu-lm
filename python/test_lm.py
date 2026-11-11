"""Test de bout en bout : charge un modele .b8n, interroge, execute les outils.  usage : python3 python/test_lm.py modele.b8n"""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gopu_lm import Model

M = Model.load_model(sys.argv[1] if len(sys.argv) > 1 else "modele.b8n")
print("info :", M.info)
for p in ["Bonjour", "Qui es-tu ?", "Combien font 17 fois 23 ?", "Quelle heure est-il ?", "Quelle est la capitale du Japon ?", "Calculatrice : 12 + 30"]:
    r = M.askInto(prompt=p, temperature=0, seed=1)
    print(f"\n> {p}\n  réflexion : {r.thinking}\n  outils    : {list(r.tool_calls)}\n  réponse   : {r}")

# precision sur des calculs inedits (graine differente de celle des donnees d'entrainement)
rnd = random.Random(12345)
ok = called = 0
N = 60
for i in range(N):
    a, b = rnd.randint(2, 99), rnd.randint(2, 99)
    r = M.askInto(f"Combien font {a} fois {b} ?", temperature=0)
    called += bool(r.tool_calls)
    ok += str(a * b) in str(r) and bool(r.tool_calls)
print(f"\ncalculs inédits : outil appelé {called}/{N}, réponse juste {ok}/{N}")

ids = M.tokenize("Bonjour <|user|>")
print("tokenize :", ids, "->", repr(M.detokenize(ids)))
print("embedding tokens :", len(M.embed("Bonjour")), "x", len(M.embed("Bonjour")[0]), "| hidden :", len(M.embed("Bonjour", "hidden")))
print("futur :", [[(t["token"], round(t["p"], 2)) for t in h[:2]] for h in M.predict_future("Bonjour")])
