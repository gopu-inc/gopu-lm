#!/usr/bin/env python3
"""Discute avec gopu-lm dans le terminal.

    python3 test_chat.py                      # modele.b8n (dossier courant ou paquet installe)
    python3 test_chat.py chemin/modele.b8n --temperature 0.3 --show-thinking

Commandes : /quit  /reset  /think (affiche/masque la reflexion)  /raw (texte brut du modele)
            /temp 0.5  /tools (active/desactive les outils)  /future (jetons probables suivants)  /help
"""
import argparse
import os
import sys

try:
    from gopu_lm import Model
except ImportError:  # depuis les sources, sans installation
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
    from gopu_lm import Model

HELP = __doc__.split("Commandes :")[1].strip()


def main():
    ap = argparse.ArgumentParser(description="Chat avec gopu-lm")
    ap.add_argument("model", nargs="?", default=None, help="fichier .b8n (defaut : modele.b8n)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--memory", type=int, default=1, help="nombre de tours precedents gardes (le contexte du modele est court : 128 jetons)")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--show-thinking", action="store_true")
    a = ap.parse_args()

    try:
        M = Model.load_model(a.model)
    except Exception as e:
        sys.exit("Impossible de charger le modele : %s" % e)
    info = M.info
    print("gopu-lm : %d parametres, contexte %d jetons. Tape /help pour les commandes, /quit pour sortir.\n" % (info["n_params"], info["ctx"]))

    history, temp, tools, think, raw = [], a.temperature, not a.no_tools, a.show_thinking, False
    while True:
        try:
            line = input("vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/reset":
                history = []
                print("(conversation effacee)")
            elif cmd == "/think":
                think = not think
                print("(reflexion %s)" % ("affichee" if think else "masquee"))
            elif cmd == "/raw":
                raw = not raw
                print("(texte brut %s)" % ("active" if raw else "desactive"))
            elif cmd == "/tools":
                tools = not tools
                print("(outils %s)" % ("actives" if tools else "desactives"))
            elif cmd == "/temp":
                try:
                    temp = float(arg)
                    print("(temperature = %s)" % temp)
                except ValueError:
                    print("usage : /temp 0.5")
            elif cmd == "/future":
                prompt = arg or (history[-2]["content"] if history else "Bonjour")
                for i, h in enumerate(M.predict_future(prompt, top=3), 1):
                    print("  +%d : %s" % (i, "  ".join("%r %.2f" % (t["token"], t["p"]) for t in h)))
            else:
                print(HELP)
            continue
        try:
            r = M.askInto(line, history=history[-2 * a.memory:] if a.memory > 0 else None, temperature=temp, top_k=a.top_k,
                          top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed, tools="auto" if tools else False)
        except Exception as e:
            print("erreur : %s" % e)
            continue
        if think and r.thinking:
            print("  [reflexion] %s" % r.thinking)
        for c in r.tool_calls:
            print("  [outil %s] %s  ->  %s" % (c["tool"], c["input"].replace("\n", " ; "), c["output"]))
        print("gopu > %s" % (r.raw if raw else (str(r) or "(reponse vide)")))
        history += [{"role": "user", "content": line}, {"role": "assistant", "content": str(r), "raw": r.raw}]


if __name__ == "__main__":
    main()
