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

_HF_MAX_FILE_SIZE = 25_000_000
_SPECIAL_TOKENS = [
    "<|pad|>", "<|eos|>", "<|system|>", "</|system|>", "<|user|>", "</|user|>",
    "<|thinking|>", "</|thinking|>", "<|assistant|>", "</|assistant|>",
    '<|tools="', "</|tools>", "<|tool_result>", "</|tool_result>",
]

_BYTE_VISIBLE = list(range(ord("!"), ord("~") + 1))
_BYTE_VISIBLE += list(range(ord("¡"), ord("¬") + 1))
_BYTE_VISIBLE += list(range(ord("®"), ord("ÿ") + 1))
_BYTE_CHARS = _BYTE_VISIBLE[:]
_byte_extra = 0
for _byte in range(256):
    if _byte not in _BYTE_VISIBLE:
        _BYTE_VISIBLE.append(_byte)
        _BYTE_CHARS.append(256 + _byte_extra)
        _byte_extra += 1
_BYTE_TO_CHAR = dict(zip(_BYTE_VISIBLE, _BYTE_CHARS))


def _read_safetensors(path):
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("fichier safetensors invalide")
        header_len = int.from_bytes(prefix, "little")
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError("en-tete safetensors tronque")
        header = json.loads(header_bytes)
        data_offset = 8 + header_len
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if data_offset > file_size:
            raise ValueError("en-tete safetensors hors limites")
        tensors = {}
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            start, end = descriptor["data_offsets"]
            if start < 0 or end < start or data_offset + end > file_size:
                raise ValueError("offsets safetensors invalides pour %s" % name)
            f.seek(data_offset + start)
            tensors[name] = (descriptor["dtype"], descriptor["shape"], f.read(end - start))
            if len(tensors[name][2]) != end - start:
                raise ValueError("tenseur safetensors tronque: %s" % name)
    return header.get("__metadata__", {}), tensors


def _encode_byte(token_bytes):
    return "".join(chr(_BYTE_TO_CHAR[byte]) for byte in token_bytes)


def _tokenizer_assets(metadata, vocab_size):
    merges = []
    merge_text = metadata.get("tok_merges", "")
    if merge_text:
        try:
            merge_ids = [tuple(int(x) for x in pair.split(",")) for pair in merge_text.split(";") if pair]
        except (TypeError, ValueError) as e:
            raise ValueError("fusions BPE invalides dans le modele") from e
    else:
        merge_ids = []
    byte_tokens = [bytes([i]) for i in range(256)]
    byte_tokens.extend(token.encode("utf-8") for token in _SPECIAL_TOKENS)
    for left, right in merge_ids:
        if left < 0 or right < 0 or left >= len(byte_tokens) or right >= len(byte_tokens):
            raise ValueError("fusion BPE hors limites dans le modele")
        byte_tokens.append(byte_tokens[left] + byte_tokens[right])
    if len(byte_tokens) != vocab_size:
        raise ValueError("vocabulaire tokenizer incompatible avec les poids")

    vocab = {_encode_byte(token): i for i, token in enumerate(byte_tokens[:256])}
    for i, token in enumerate(_SPECIAL_TOKENS, start=256):
        vocab[token] = i
    for i, (left, right) in enumerate(merge_ids, start=270):
        vocab[_encode_byte(byte_tokens[i])] = i
        merges.append("%s %s" % (_encode_byte(byte_tokens[left]), _encode_byte(byte_tokens[right])))
    if len(vocab) != vocab_size:
        raise ValueError("le vocabulaire contient des tokens identiques non exportables")

    added_tokens = [
        {"id": i + 256, "content": token, "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True}
        for i, token in enumerate(_SPECIAL_TOKENS)
    ]
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "vocab": vocab, "merges": merges},
    }
    return tokenizer, byte_tokens


def _safetensors_header(metadata, tensors):
    entries = {"__metadata__": metadata}
    data_size = 0
    for name, (dtype, shape, raw) in tensors.items():
        start = data_size
        data_size += len(raw)
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, data_size]}
    header = json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    return header, data_size


def _safetensors_size(metadata, tensors):
    header, data_size = _safetensors_header(metadata, tensors)
    return 8 + len(header) + data_size


def _write_safetensors(path, metadata, tensors):
    header, _ = _safetensors_header(metadata, tensors)
    with open(path, "wb") as f:
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
        for _, _, raw in tensors.values():
            f.write(raw)
    return 8 + len(header) + sum(len(raw) for _, _, raw in tensors.values())


def _write_hf_weights(directory, metadata, tensors):
    shards = []
    current = {}
    for name, tensor in tensors.items():
        candidate = dict(current)
        candidate[name] = tensor
        if _safetensors_size(metadata, candidate) <= _HF_MAX_FILE_SIZE:
            current = candidate
            continue
        if not current:
            raise ValueError("le tenseur %s depasse la limite Hugging Face de 25 000 000 octets" % name)
        shards.append(current)
        current = {name: tensor}
        if _safetensors_size(metadata, current) > _HF_MAX_FILE_SIZE:
            raise ValueError("le tenseur %s depasse la limite Hugging Face de 25 000 000 octets" % name)
    if current:
        shards.append(current)

    weight_map = {}
    if len(shards) == 1:
        index_path = os.path.join(directory, "model.safetensors.index.json")
        if os.path.exists(index_path):
            os.remove(index_path)
        _write_safetensors(os.path.join(directory, "model.safetensors"), metadata, shards[0])
        weight_map = {name: "model.safetensors" for name in shards[0]}
    else:
        single_path = os.path.join(directory, "model.safetensors")
        if os.path.exists(single_path):
            os.remove(single_path)
        total = len(shards)
        for i, shard in enumerate(shards, start=1):
            filename = "model-%05d-of-%05d.safetensors" % (i, total)
            _write_safetensors(os.path.join(directory, filename), metadata, shard)
            weight_map.update((name, filename) for name in shard)
        index = {"metadata": {"total_size": sum(len(raw) for _, _, raw in tensors.values())},
                 "weight_map": weight_map}
        with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
            f.write("\n")


def _export_hf_directory(model, directory):
    os.makedirs(directory, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = os.path.join(temp_dir, "model.safetensors")
        if model._L.gopu_lm_save(model._h, _b(raw_path)) != 0:
            _take(None)
        metadata, tensors = _read_safetensors(raw_path)

    required = ("vocab", "d", "layers", "heads", "ctx", "ff", "future")
    try:
        cfg = {key: int(metadata[key]) for key in required}
    except (KeyError, ValueError) as e:
        raise ValueError("configuration du modele incomplete pour l'export Hugging Face") from e
    tokenizer, byte_tokens = _tokenizer_assets(metadata, cfg["vocab"])
    _write_hf_weights(directory, metadata, tensors)

    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n</|system|>\n"
        "{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n</|user|>\n"
        "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}\n</|assistant|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}\n"
    )
    config = {
        "architectures": ["GopuForCausalLM"],
        "model_type": "gopu_lm",
        "vocab_size": cfg["vocab"],
        "hidden_size": cfg["d"],
        "num_hidden_layers": cfg["layers"],
        "num_attention_heads": cfg["heads"],
        "max_position_embeddings": cfg["ctx"],
        "intermediate_size": cfg["ff"],
        "future_heads": cfg["future"],
        "pad_token_id": 256,
        "eos_token_id": 257,
        "tie_word_embeddings": True,
        "use_cache": False,
        "auto_map": {
            "AutoConfig": "configuration_gopu.GopuConfig",
            "AutoModel": "modeling_gopu.GopuForCausalLM",
            "AutoModelForCausalLM": "modeling_gopu.GopuForCausalLM",
        },
    }
    files = {
        "config.json": json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        "tokenizer.json": json.dumps(tokenizer, ensure_ascii=False, separators=(",", ":")) + "\n",
        "tokenizer_config.json": json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": cfg["ctx"],
            "pad_token": _SPECIAL_TOKENS[0],
            "eos_token": _SPECIAL_TOKENS[1],
            "additional_special_tokens": _SPECIAL_TOKENS[2:],
            "chat_template": chat_template,
        }, ensure_ascii=False, indent=2) + "\n",
        "special_tokens_map.json": json.dumps({
            "pad_token": _SPECIAL_TOKENS[0],
            "eos_token": _SPECIAL_TOKENS[1],
            "additional_special_tokens": _SPECIAL_TOKENS[2:],
        }, ensure_ascii=False, indent=2) + "\n",
        "chat_template.jinja": chat_template,
    }
    for name, content in files.items():
        with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
            f.write(content)
    with open(os.path.join(directory, "vocab.txt"), "w", encoding="utf-8") as f:
        for token_id, token in enumerate(byte_tokens):
            display = _SPECIAL_TOKENS[token_id - 256] if 256 <= token_id < 270 else _encode_byte(token)
            f.write(display + "\n")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("configuration_gopu.py", "modeling_gopu.py"):
        with open(os.path.join(package_dir, name), encoding="utf-8") as src:
            with open(os.path.join(directory, name), "w", encoding="utf-8") as dst:
                dst.write(src.read())


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
        "gopu_lm_lora_train": (p, [p, c, c]), "gopu_lm_lora_save": (ctypes.c_int, [p, c]),
        "gopu_lm_lora_load": (ctypes.c_int, [p, c]),
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
        path = os.fsdecode(os.fspath(path))
        if os.path.isdir(path) or os.path.splitext(path)[1].lower() not in (".b8n", ".safetensors"):
            _export_hf_directory(self, path)
            return
        if self._L.gopu_lm_save(self._h, _b(path)) != 0:
            _take(None)

    save_pretrained = save_model

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

    def train_lora(self, data, *, rank=8, alpha=None, **opts):
        """Adapte uniquement une petite tete LoRA de sortie.

        Le Transformer charge reste gele. `rank` determine la taille de l'adaptateur ;
        `alpha` vaut `rank` par defaut. Les autres options sont celles de :meth:`train`.
        """
        if os.path.exists(data):
            with open(data, encoding="utf-8") as f:
                data = f.read()
        opts = dict(opts)
        opts["rank"] = int(rank)
        opts["alpha"] = float(rank if alpha is None else alpha)
        return json.loads(_take(self._L.gopu_lm_lora_train(self._h, _b(data), _b(json.dumps(opts)))))

    def save_lora(self, path):
        """Enregistre l'adaptateur LoRA actif dans un fichier `.b8n` ou `.safetensors`."""
        if self._L.gopu_lm_lora_save(self._h, _b(os.fspath(path))) != 0:
            _take(None)

    def load_lora(self, path):
        """Charge un adaptateur LoRA correspondant à l'architecture du modèle."""
        if self._L.gopu_lm_lora_load(self._h, _b(os.fspath(path))) != 0:
            _take(None)
        return self

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
