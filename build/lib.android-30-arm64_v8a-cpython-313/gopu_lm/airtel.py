"""Interface Python (bibliotheque standard seulement) du modele airtel : pilote le binaire Rust `airtel`.

    from gopu_lm.airtel import Airtel
    m = Airtel("airtel-2.1m")                       # dossier du modele (ou .b8n)
    print(m.ask("Bonjour", temperature=0))
    m.lora("finetune/polite.txt", adapter_out="adapters/poli", merge_out="airtel-2.1m-poli", steps=300)
    answers = m.sample(["Bonjour", "Merci"], n=4)   # plusieurs reponses par prompt
    m.dpo("finetune/prefs.jsonl", merge_out="airtel-2.1m-poli-dpo")

Le binaire est cherche dans $GOPU_AIRTEL_BIN, le PATH, puis target/release/airtel.
"""
import json
import os
import shutil
import subprocess
import tempfile


def _find_bin():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.environ.get("GOPU_AIRTEL_BIN"), os.path.join(here, "airtel"), shutil.which("airtel"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "release", "airtel")):
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("binaire 'airtel' introuvable : lance `cargo build --release --bin airtel` (ou fixe GOPU_AIRTEL_BIN)")


def _run(args, capture=True):
    p = subprocess.run([_find_bin()] + [str(a) for a in args], text=True,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "").strip() or "airtel a echoue")
    return p.stdout or ""


def _opts(**kw):
    out = []
    for k, v in kw.items():
        if v is not None:
            out += ["--" + k.replace("_", "-"), v]
    return out


class Airtel:
    def __init__(self, model, adapter=None, system=None):
        self.model, self.adapter, self.system = model, adapter, system

    def _base(self):
        return _opts(model=self.model, adapter=self.adapter)

    def info(self):
        return _run(["info"] + self._base())

    def ask(self, prompt, max_tokens=160, temperature=0.7, top_k=40, top_p=0.95, seed=1, repetition_penalty=1.1):
        return _run(["ask"] + self._base() + _opts(prompt=prompt, system=self.system, max_tokens=max_tokens, temperature=temperature,
                                                   top_k=top_k, top_p=top_p, seed=seed, repetition_penalty=repetition_penalty)).strip()

    def sample(self, prompts, n=4, temperature=0.9, max_tokens=160, seed=1):
        with tempfile.TemporaryDirectory() as d:
            pf, of = os.path.join(d, "p.txt"), os.path.join(d, "o.jsonl")
            with open(pf, "w", encoding="utf-8") as f:
                f.write("\n".join(p.replace("\n", " ") for p in prompts) + "\n")
            _run(["sample"] + self._base() + _opts(prompts=pf, out=of, n=n, temperature=temperature, max_tokens=max_tokens, seed=seed, system=self.system))
            with open(of, encoding="utf-8") as f:
                return [json.loads(l) for l in f if l.strip()]

    def train(self, data, out=None, **kw):
        _run(["train"] + self._base() + _opts(data=data, out=out, **kw), capture=False)
        return Airtel(out or self.model)

    def lora(self, data, adapter_out=None, merge_out=None, **kw):
        _run(["lora"] + self._base() + _opts(data=data, adapter_out=adapter_out, merge_out=merge_out, **kw), capture=False)
        return Airtel(merge_out or self.model, adapter=None if merge_out else adapter_out)

    def dpo(self, pairs, adapter_out=None, merge_out=None, **kw):
        _run(["dpo"] + self._base() + _opts(pairs=pairs, adapter_out=adapter_out, merge_out=merge_out, **kw), capture=False)
        return Airtel(merge_out or self.model, adapter=None if merge_out else adapter_out)


def main():
    """Point d'entree console `airtel` : transmet les arguments au binaire Rust."""
    import sys
    sys.exit(subprocess.call([_find_bin()] + sys.argv[1:]))
