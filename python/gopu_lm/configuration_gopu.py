from transformers import PretrainedConfig


class GopuConfig(PretrainedConfig):
    model_type = "gopu_lm"

    def __init__(
        self,
        vocab_size=640,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=128,
        intermediate_size=256,
        future_heads=2,
        **kwargs,
    ):
        kwargs.setdefault("pad_token_id", 256)
        kwargs.setdefault("eos_token_id", 257)
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.future_heads = future_heads
