import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_gopu import GopuConfig


class GopuEmbedding(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight)


class GopuLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.attn = GopuAttention(config)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.mlp = GopuMLP(config)

    def forward(self, hidden, attention_mask=None):
        hidden = hidden + self.attn(self.ln1(hidden), attention_mask)
        return hidden + self.mlp(self.ln2(hidden))


class GopuAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.heads = config.num_attention_heads
        self.head_dim = d // self.heads
        self.wq = nn.Parameter(torch.empty(d, d))
        self.wk = nn.Parameter(torch.empty(d, d))
        self.wv = nn.Parameter(torch.empty(d, d))
        self.wo = nn.Parameter(torch.empty(d, d))

    def forward(self, hidden, attention_mask=None):
        batch, length, width = hidden.shape

        def split_heads(weight):
            return (hidden @ weight).view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        query, key, value = split_heads(self.wq), split_heads(self.wk), split_heads(self.wv)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = (weights @ value).transpose(1, 2).contiguous().view(batch, length, width)
        return context @ self.wo


class GopuMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(config.hidden_size, config.intermediate_size))
        self.b1 = nn.Parameter(torch.empty(config.intermediate_size))
        self.w2 = nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size))
        self.b2 = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, hidden):
        hidden = F.gelu(hidden @ self.w1 + self.b1, approximate="tanh")
        return hidden @ self.w2 + self.b2


class GopuFutureHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(config.hidden_size))


class GopuForCausalLM(PreTrainedModel):
    config_class = GopuConfig
    base_model_prefix = ""
    _supports_cache_class = False

    def __init__(self, config):
        super().__init__(config)
        self.config.use_cache = False
        d = config.hidden_size
        self.tok_emb = nn.Parameter(torch.empty(config.vocab_size, d))
        self.pos_emb = nn.Parameter(torch.empty(config.max_position_embeddings, d))
        self.layers = nn.ModuleList(GopuLayer(config) for _ in range(config.num_hidden_layers))
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.future = nn.ModuleList(GopuFutureHead(config) for _ in range(config.future_heads))
        self.post_init()

    def get_input_embeddings(self):
        return GopuEmbedding(self.tok_emb)

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def get_output_embeddings(self):
        return None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if past_key_values is not None or use_cache:
            raise ValueError("Gopu ne prend pas encore en charge le cache KV")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("fournir input_ids ou inputs_embeds")
        if input_ids is not None:
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            batch, length = input_ids.shape
            if inputs_embeds is None:
                inputs_embeds = F.embedding(input_ids, self.tok_emb)
        else:
            batch, length = inputs_embeds.shape[:2]
        if length > self.config.max_position_embeddings:
            raise ValueError("sequence trop longue pour max_position_embeddings")
        if position_ids is None:
            position_ids = torch.arange(length, device=inputs_embeds.device).unsqueeze(0)
        hidden = inputs_embeds + F.embedding(position_ids, self.pos_emb)
        all_hidden = [hidden] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
            if all_hidden is not None:
                all_hidden.append(hidden)
        hidden = self.ln_f(hidden)
        if all_hidden is not None:
            all_hidden.append(hidden)
        logits = F.linear(hidden, self.tok_emb)
        logits[..., self.config.pad_token_id] = torch.finfo(logits.dtype).min
        loss = None
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            shifted_labels = labels[:, 1:].contiguous().clone()
            shifted_labels[shifted_labels == self.config.pad_token_id] = -100
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=tuple(all_hidden) if all_hidden is not None else None,
            attentions=None,
        )
