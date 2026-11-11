#!/usr/bin/env python3
"""Verification croisee : relit airtel-2.1m/{config.json,model.safetensors,vocab.json,merges.txt} SANS le code Rust du modele
(numpy seulement) et compare avec les logits calcules par le binaire `airtel`.

    python3 tools/check_hf_numpy.py airtel-2.1m "<|user|>\nBonjour\n</|user|>\n<|assistant|>\n"

Verifie : lecture safetensors, noms/formes Hugging Face (Llama), RMSNorm, RoPE rotate_half, attention causale, SwiGLU,
tetes liees, ET le decoupage BPE reimplemente en Python a partir de vocab.json + merges.txt.
"""
import json, os, struct, subprocess, sys, unicodedata
import numpy as np


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = np.frombuffer(f.read(b - a), dtype="<f4").reshape(v["shape"]).astype(np.float64)
    return out


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, theta, hd):
    t = x.shape[0]
    half = hd // 2
    inv = theta ** (-np.arange(half) * 2.0 / hd)
    ang = np.arange(t)[:, None] * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x = x.reshape(t, -1, hd).copy()
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * c[:, None] - b * s[:, None], b * c[:, None] + a * s[:, None]], -1).reshape(t, -1)


def llama_logits(dirp, ids):
    cfg = json.load(open(os.path.join(dirp, "config.json")))
    W = read_safetensors(os.path.join(dirp, "model.safetensors"))
    d, L, H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
    hd, eps, th = d // H, cfg["rms_norm_eps"], cfg["rope_theta"]
    x = W["model.embed_tokens.weight"][ids]
    t = len(ids)
    mask = np.triu(np.full((t, t), -np.inf), 1)
    for l in range(L):
        p = "model.layers.%d." % l
        h = rms(x, W[p + "input_layernorm.weight"], eps)
        q = rope(h @ W[p + "self_attn.q_proj.weight"].T, th, hd)
        k = rope(h @ W[p + "self_attn.k_proj.weight"].T, th, hd)
        v = h @ W[p + "self_attn.v_proj.weight"].T
        att = np.zeros((t, d))
        for i in range(H):
            sl = slice(i * hd, (i + 1) * hd)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            att[:, sl] = s @ v[:, sl]
        x = x + att @ W[p + "self_attn.o_proj.weight"].T
        h = rms(x, W[p + "post_attention_layernorm.weight"], eps)
        g = h @ W[p + "mlp.gate_proj.weight"].T
        u = h @ W[p + "mlp.up_proj.weight"].T
        x = x + (g / (1 + np.exp(-g)) * u) @ W[p + "mlp.down_proj.weight"].T
    x = rms(x, W["model.norm.weight"], eps)
    assert "lm_head.weight" not in W and cfg["tie_word_embeddings"]
    return x[-1] @ W["model.embed_tokens.weight"].T


# ---- BPE reimplemente a partir des fichiers HF
def byte_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def pretokenize(text):
    """ ?\\p{L}+|\\p{N}|[^\\s\\p{L}\\p{N}]|\\s+(?!\\S)|\\s+ """
    L = lambda c: unicodedata.category(c).startswith("L")
    N = lambda c: unicodedata.category(c).startswith("N")
    S = lambda c: c.isspace()
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        j = i + 1 if c == " " else i
        if j < n and L(text[j]):
            while j < n and L(text[j]):
                j += 1
            out.append(text[i:j]); i = j
        elif N(c):
            out.append(c); i += 1
        elif not S(c):
            out.append(c); i += 1
        else:
            j = i
            while j < n and S(text[j]):
                j += 1
            if j == n or j - i == 1:
                out.append(text[i:j]); i = j
            else:
                out.append(text[i:j - 1]); i = j - 1
    return out


def encode(dirp, text):
    vocab = json.load(open(os.path.join(dirp, "vocab.json"), encoding="utf-8"))
    merges = [l.split(" ") for l in open(os.path.join(dirp, "merges.txt"), encoding="utf-8").read().split("\n") if l and not l.startswith("#")]
    rank = {tuple(m): i for i, m in enumerate(merges)}
    specials = sorted([k for k in vocab if k.startswith("<|") or k.startswith("</|")], key=len, reverse=True)
    bu = byte_unicode()
    ids, i, buf = [], 0, ""
    def flush(seg):
        for piece in pretokenize(seg):
            parts = [bu[b] for b in piece.encode("utf-8")]
            while len(parts) > 1:
                best = min(((rank.get((parts[k], parts[k + 1]), 1 << 30), k) for k in range(len(parts) - 1)))
                if best[0] == 1 << 30:
                    break
                k = best[1]; parts[k:k + 2] = [parts[k] + parts[k + 1]]
            ids.extend(vocab[p] for p in parts)
    while i < len(text):
        sp = next((s for s in specials if text.startswith(s, i)), None)
        if sp:
            flush(buf); buf = ""; ids.append(vocab[sp]); i += len(sp)
        else:
            buf += text[i]; i += 1
    flush(buf)
    return ids


def main():
    dirp = sys.argv[1]
    text = (sys.argv[2] if len(sys.argv) > 2 else "<|user|>\nBonjour, combien font 12 plus 30 ?\n</|user|>\n<|assistant|>\n").replace("\\n", "\n")
    here = os.path.dirname(os.path.abspath(__file__))
    binp = os.environ.get("GOPU_AIRTEL_BIN") or os.path.join(here, "..", "target", "release", "airtel")
    py_ids = encode(dirp, text)
    out = json.loads(subprocess.check_output([binp, "logits", "--model", dirp, "--text", text]))
    print("jetons Rust == jetons Python (BPE reimplemente) :", out["ids"] == py_ids, "(%d jetons)" % len(py_ids))
    ref = llama_logits(dirp, py_ids)
    rl = np.array(out["logits"], dtype=np.float64)
    err = np.abs(ref - rl).max()
    print("ecart max des logits Rust vs numpy : %.2e (amplitude %.1f)" % (err, np.abs(ref).max()))
    print("jeton suivant identique :", int(ref.argmax()) == int(rl.argmax()))
    sys.exit(0 if out["ids"] == py_ids and err < 1e-2 else 1)


if __name__ == "__main__":
    main()
