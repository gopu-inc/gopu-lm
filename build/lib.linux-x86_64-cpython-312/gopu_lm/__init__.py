"""gopu_lm : API Python (ctypes, aucune dependance) du modele de langage gopu-lm.

    from gopu_lm import Model
    M = Model.load_model("modele.b8n")
    print(M.askInto(prompt="Bonjour", temperature=0.3))
"""
import ast
import ctypes
import json
import operator
import os
import re
import subprocess
import sys
import tempfile

__all__ = ["Model", "Reply", "default_tools"]


# ------------------------------------------------------------------ bibliotheque native

def _find_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    name = {"darwin": "libgopu.dylib", "win32": "gopu.dll"}.get(sys.platform, "libgopu.so")
    cands = [os.environ.get("GOPU_LIB", ""), os.path.join(here, name),
             os.path.join(here, "..", "..", "target", "release", name)]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise OSError("libgopu introuvable : lance ./build.sh (ou cargo build --release) ou definis GOPU_LIB")


_lib = None


def _L():
    global _lib
    if _lib is not None:
        return _lib
    L = ctypes.CDLL(_find_lib())
    c, p, sz = ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t
    sig = {
        "gopu_last_error": (ctypes.c_void_p, []),
        "gopu_string_free": (None, [p]),
        "gopu_lm_load": (p, [c]), "gopu_lm_free": (None, [p]), "gopu_lm_save": (ctypes.c_int, [p, c]),
        "gopu_lm_new": (p, [c, c]), "gopu_lm_train": (p, [p, c, c]),
        "gopu_lm_generate": (p, [p, c, c]), "gopu_lm_future": (p, [p, c, sz]),
        "gopu_lm_encode": (p, [p, c]), "gopu_lm_decode": (p, [p, c]),
        "gopu_lm_embed": (p, [p, c, c]), "gopu_lm_info": (p, [p]),
    }
    for n, (res, args) in sig.items():
        f = getattr(L, n)
        f.restype, f.argtypes = res, args
    _lib = L
    return L


def _take(ptr):
    """Copie une chaine C renvoyee par la lib, puis la libere. Leve une erreur si NULL."""
    L = _L()
    if not ptr:
        e = L.gopu_last_error()
        msg = ctypes.string_at(e).decode("utf-8", "replace") if e else "erreur inconnue"
        if e:
            L.gopu_string_free(e)
        raise RuntimeError(msg)
    s = ctypes.string_at(ptr).decode("utf-8")
    L.gopu_string_free(ptr)
    return s


def _b(s):
    return s.encode("utf-8")


# ------------------------------------------------------------------ outils

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def _calc(expr):
    """Calculatrice sure : nombres et operateurs arithmetiques uniquement."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError("exposant trop grand")
            return _OPS[type(n.op)](a, b)
        raise ValueError("expression non autorisee")
    r = ev(ast.parse(expr.strip(), mode="eval"))
    return str(int(r)) if isinstance(r, float) and r.is_integer() else str(r)


def _python(code, timeout=5.0):
    """Execute du code Python dans un sous-processus isole (python -I, dossier temporaire, limites CPU/memoire sous POSIX).
    Ce n'est PAS un bac a sable complet : le code genere par un modele reste du code arbitraire."""
    # Limite d'espace d'adressage : desactivee sur Android/Termux (le chargeur bionic y reserve beaucoup de
    # memoire virtuelle et Python ne demarre plus). Reglable : GOPU_PY_MEM_MB=0 la desactive partout.
    android = hasattr(sys, "getandroidapilevel") or os.path.exists("/system/bin/linker64") or "ANDROID_ROOT" in os.environ
    mem_mb = int(os.environ.get("GOPU_PY_MEM_MB", "0" if android else "512"))

    def limits():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
            if mem_mb > 0:
                resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        except Exception:
            pass
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout, cwd=d,
                               env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
                               preexec_fn=limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return "Erreur : delai depasse"
    if r.returncode != 0 and "linker" in (r.stderr or "") and os.name == "posix":
        try:  # dernier recours : le systeme refuse les limites, on relance sans (le delai reste applique)
            r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout,
                               cwd=tempfile.gettempdir(), env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"})
        except subprocess.TimeoutExpired:
            return "Erreur : delai depasse"
    out = (r.stdout or "").strip()
    if r.returncode != 0:
        err = (r.stderr or "").strip().splitlines()
        return "Erreur : " + (err[-1] if err else "code %d" % r.returncode)
    return out[:2000]


def default_tools():
    return {"python": _python, "calc": _calc}


# ------------------------------------------------------------------ reponse

class Reply(str):
    """Texte de la reponse (sans reflexion ni appels d'outils), avec les details en attributs."""
    thinking = ""
    tool_calls = ()
    raw = ""
    finish = ""
    n_tokens = 0

    def __repr__(self):
        return "Reply(%s)" % str.__repr__(self)


_TOOL_RE = re.compile(r'<\|tools="([^"]*)">\n?(.*?)\n?</\|tools>', re.S)
_THINK_RE = re.compile(r"<\|thinking\|>\n?(.*?)\n?</\|thinking\|>", re.S)
_RESULT_RE = re.compile(r"<\|tool_result>.*?</\|tool_result>\n?", re.S)
_TAGS = ["<|assistant|>", "</|assistant|>", "<|eos|>", "<|pad|>"]


def _clean_answer(text):
    t = _THINK_RE.sub("", text)
    t = _RESULT_RE.sub("", _TOOL_RE.sub("", t))
    for tag in _TAGS:
        t = t.replace(tag, "")
    return t.strip()


# ------------------------------------------------------------------ modele

class Model:
    """Modele de langage gopu-lm charge depuis un fichier .b8n / .safetensors."""

    def __init__(self, handle):
        self._h = handle
        self._L = _L()

    # -- creation / chargement / sauvegarde
    @classmethod
    def load_model(cls, path=None):
        """Charge un modele .b8n / .safetensors. Sans argument : modele.b8n du dossier courant, sinon celui installe avec le paquet."""
        if path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            cands = ["modele.b8n", os.path.join(here, "modele.b8n"), os.path.join(here, "..", "..", "modele.b8n")]
            path = next((c for c in cands if os.path.exists(c)), None)
            if path is None:
                raise FileNotFoundError("aucun modele.b8n trouve : passe un chemin a load_model(...)")
        h = _L().gopu_lm_load(_b(os.fspath(path)))
        if not h:
            _take(None)
        return cls(h)

    from_pretrained = load_model

    @classmethod
    def new(cls, corpus, vocab=640, d=64, layers=3, heads=4, ctx=128, future=2, ff=None, seed=1):
        """Nouveau modele non entraine ; le tokenizer BPE est appris sur `corpus` (texte)."""
        cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, ctx=ctx, future=future, seed=seed, ff=ff or 4 * d)
        h = _L().gopu_lm_new(_b(json.dumps(cfg)), _b(corpus))
        if not h:
            _take(None)
        return cls(h)

    def save_model(self, path):
        if self._L.gopu_lm_save(self._h, _b(os.fspath(path))) != 0:
            _take(None)

    def __del__(self):
        try:
            if self._h:
                self._L.gopu_lm_free(self._h)
                self._h = None
        except Exception:
            pass

    @property
    def info(self):
        return json.loads(_take(self._L.gopu_lm_info(self._h)))

    # -- entrainement
    def train(self, data, **opts):
        """Entraine (ou continue d'entrainer) sur un corpus : texte, ou chemin d'un fichier.
        Options : steps, batch, lr, warmup, wd, clip, future_weight, user_weight, seed, log_every, eval_every, max_seconds..."""
        if os.path.exists(data):
            with open(data, encoding="utf-8") as f:
                data = f.read()
        return json.loads(_take(self._L.gopu_lm_train(self._h, _b(data), _b(json.dumps(opts)))))

    # -- tokenizer / embeddings / futur
    def tokenize(self, text):
        return json.loads(_take(self._L.gopu_lm_encode(self._h, _b(text))))

    encode = tokenize

    def detokenize(self, ids):
        return _take(self._L.gopu_lm_decode(self._h, _b(json.dumps(list(ids)))))

    decode = detokenize

    def embed(self, text, mode="tokens"):
        """Liste de vecteurs (un par jeton). mode="tokens" : embeddings appris ; "hidden" : etats contextuels (apres attention)."""
        r = json.loads(_take(self._L.gopu_lm_embed(self._h, _b(text), _b(mode))))
        n, d = r["shape"]
        return [r["data"][i * d:(i + 1) * d] for i in range(n)]

    def predict_future(self, prompt, top=3):
        """Jetons les plus probables aux horizons +1, +2, ... (tetes de prediction du futur)."""
        return json.loads(_take(self._L.gopu_lm_future(self._h, _b(self._render(prompt, None, None)), top)))

    # -- conversation
    @staticmethod
    def _render(prompt, system, history):
        s = ""
        if system:
            s += "<|system|>\n%s\n</|system|>\n" % system
        for m in history or []:
            role = m["role"]
            if role == "user":
                s += "<|user|>\n%s\n</|user|>\n" % m["content"]
            elif role == "assistant":
                if m.get("raw"):  # tour precedent tel que genere (reflexion + outils), comme pendant l'entrainement
                    s += m["raw"].rstrip("\n") + "\n"
                else:
                    s += "<|assistant|>\n%s\n</|assistant|>\n" % m["content"]
        return s + "<|user|>\n%s\n</|user|>\n" % prompt

    def askInto(self, prompt, *, system=None, history=None, max_tokens=160, temperature=0.7, top_k=40, top_p=0.95,
                repetition_penalty=1.0, seed=None, stop=None, tools="auto", max_tool_rounds=3,
                show_thinking=False, return_dict=False, **extra):
        """Pose une question au modele.

        prompt : texte (ou dict {"prompt": ..., "temperature": ...} regroupant les arguments)
        tools  : "auto" (python + calc), False/{} (aucun), ou dict {nom: fonction(str) -> str} qui complete/remplace
        Retourne un Reply (str) ; .thinking, .tool_calls, .raw, .finish en donnent le detail. return_dict=True -> dict.
        """
        if isinstance(prompt, dict):
            kw = dict(prompt)
            prompt = kw.pop("prompt")
            return self.askInto(prompt, **{**dict(system=system, history=history, max_tokens=max_tokens, temperature=temperature,
                                                  top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, seed=seed, stop=stop,
                                                  tools=tools, max_tool_rounds=max_tool_rounds, show_thinking=show_thinking,
                                                  return_dict=return_dict), **kw})
        if tools == "auto":
            reg = default_tools()
        elif not tools:
            reg = {}
        else:
            reg = {**default_tools(), **tools}
        base = self._render(prompt, system, history)
        gen = dict(max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                   repetition_penalty=repetition_penalty, **extra)
        if stop:
            gen["stop"] = list(stop)
        acc, calls, finish, ntok = "", [], "", 0
        for rnd in range(max_tool_rounds + 1):
            if seed is not None:
                gen["seed"] = int(seed) + rnd
            r = json.loads(_take(self._L.gopu_lm_generate(self._h, _b(base + acc), _b(json.dumps(gen)))))
            acc += r["text"]
            finish, ntok = r["finish"], ntok + r["n_tokens"]
            if r["stop"] != "</|tools>" or rnd == max_tool_rounds:
                break
            m = list(_TOOL_RE.finditer(acc))
            if not m:
                break
            name, code = m[-1].group(1), m[-1].group(2)
            fn = reg.get(name)
            if fn is None:
                res = "Erreur : outil inconnu '%s'" % name
            else:
                try:
                    res = str(fn(code))
                except Exception as e:  # l'erreur est renvoyee au modele comme resultat
                    res = "Erreur : %s" % e
            calls.append({"tool": name, "input": code, "output": res})
            acc += "\n<|tool_result>\n%s\n</|tool_result>\n" % res
        th = _THINK_RE.search(acc)
        text = _clean_answer(acc)
        if show_thinking and th:
            text = "[reflexion] " + th.group(1).strip() + "\n" + text
        out = Reply(text)
        out.thinking = th.group(1).strip() if th else ""
        out.tool_calls = tuple(calls)
        out.raw, out.finish, out.n_tokens = acc, finish, ntok
        if return_dict:
            return dict(answer=_clean_answer(acc), thinking=out.thinking, tool_calls=calls, raw=acc, finish=finish, n_tokens=ntok)
        return out

    ask_into = askInto
    __call__ = askInto
