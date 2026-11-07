"""Adaptation LoRA locale d'un :class:`gopu_lm.Model`.

L'adaptateur est applique a la tete de sortie partagee. Il contient deux
petites matrices A et B et laisse tous les poids du Transformer inchanges.
Il se sauvegarde separement du modele de base, ce qui rend les adaptations
faciles a distribuer et a remplacer.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: Optional[float] = None
    learning_rate: float = 1e-3
    steps: int = 300
    batch_size: int = 8
    seed: int = 1

    def __post_init__(self):
        if self.rank < 1:
            raise ValueError("rank doit etre superieur ou egal a 1")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps et batch_size doivent etre superieurs ou egaux a 1")


class LoRATrainer:
    """Entraine, enregistre et charge les adaptateurs LoRA de gopu-lm."""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or LoRAConfig()

    def train(self, corpus, **overrides):
        """Entraine l'adaptateur sur du texte ou un chemin de corpus gopu."""
        cfg = self.config
        opts = dict(
            rank=overrides.pop("rank", cfg.rank),
            alpha=overrides.pop("alpha", cfg.alpha),
            steps=overrides.pop("steps", cfg.steps),
            batch=overrides.pop("batch", cfg.batch_size),
            lr=overrides.pop("lr", cfg.learning_rate),
            seed=overrides.pop("seed", cfg.seed),
        )
        opts.update(overrides)
        return self.model.train_lora(str(corpus) if isinstance(corpus, Path) else corpus, **opts)

    def save(self, path):
        self.model.save_lora(path)
        return Path(path)

    def load(self, path):
        self.model.load_lora(path)
        return self.model

