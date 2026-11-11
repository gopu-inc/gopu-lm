"""Configuration du modele airtel (compatible Hugging Face `LlamaConfig`), en Python pur.

    from gopu_lm.configuration import AirtelConfig
    cfg = AirtelConfig()                      # airtel-2.1m : d=128, 8 couches, 4 tetes, ff=344, contexte 20000
    cfg.n_params()                            # 2_107_520
    cfg.save("airtel-2.1m/config.json")       # lisible par transformers.LlamaConfig / AutoConfig
    cfg = AirtelConfig.load("airtel-2.1m")    # relit config.json
"""
import json
import os
from dataclasses import dataclass, asdict

CONTEXT_TOKENS = 20000
MAX_TRAIN_TOKENS = 1_000_000

SPECIAL_TOKENS = ["<|pad|>", "<|eos|>", "<|system|>", "</|system|>", "<|user|>", "</|user|>", "<|thinking|>",
                  "</|thinking|>", "<|assistant|>", "</|assistant|>", '<|tools="', "</|tools>", "<|tool_result>",
                  "</|tool_result>"]


@dataclass
class AirtelConfig:
    vocab_size: int = 4096
    hidden_size: int = 128
    intermediate_size: int = 344
    num_hidden_layers: int = 8
    num_attention_heads: int = 4
    max_position_embeddings: int = CONTEXT_TOKENS
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    model_type: str = "llama"
    eos_token_id: int = 257
    pad_token_id: int = 256

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def n_params(self):
        d, f, v, L = self.hidden_size, self.intermediate_size, self.vocab_size, self.num_hidden_layers
        return v * d + L * (2 * d + 4 * d * d + 3 * d * f) + d

    def size_bytes(self, dtype_bytes=4):
        return self.n_params() * dtype_bytes

    def validate(self):
        if self.hidden_size % self.num_attention_heads or self.head_dim % 2:
            raise ValueError("hidden_size doit etre divisible par num_attention_heads, avec une dimension de tete paire (RoPE)")
        return self

    def to_dict(self):
        d = asdict(self)
        d.update(architectures=["LlamaForCausalLM"], hidden_act="silu", num_key_value_heads=self.num_attention_heads,
                 head_dim=self.head_dim, attention_bias=False, mlp_bias=False, initializer_range=0.02,
                 rope_scaling=None, torch_dtype="float32", use_cache=True)
        return d

    def save(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in names}).validate()
