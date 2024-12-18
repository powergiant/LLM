import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Callable
import math
from transformers.cache_utils import Cache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

@dataclass
class Qwen2MLPConfig:
    hidden_size: int
    intermediate_size: int
    hidden_act: Literal["silu"]


ACT2FN: dict[str, Callable] = {
    "silu": nn.functional.silu,
}

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2MLPConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: Tensor):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, unsqueeze_dim: int=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@dataclass
class Qwen2RotaryEmbeddingConfig:
    hidden_size: int 
    num_attention_heads: int
    max_position_embeddings: int
    device: Optional[torch.device]=None
    rope_theta: float=10000.0
    scaling_factor: float=1.0
    rope_type: str="default"

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen2RotaryEmbeddingConfig,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        # BC: "rope_type" was originally "type"
        self.rope_type = config.rope_type
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, config.device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@dataclass
class Qwen2AttentionConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    attention_dropout: float

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2AttentionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # TODO: implement kv cache
        # self._kv_cache = None

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states: Tensor = self.q_proj(hidden_states)
        key_states: Tensor = self.k_proj(hidden_states)
        value_states: Tensor = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output

@dataclass
class Qwen2DecoderLayerConfig:
    # rms_conf
    rms_norm_eps: float
    # mlp_conf
    hidden_size: int
    intermediate_size: int
    hidden_act: Literal["silu"]
    # attn_conf
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    attention_dropout: float

    def to_mlp_conf(self) -> Qwen2MLPConfig:
        return Qwen2MLPConfig(self.hidden_size, self.intermediate_size, self.hidden_act)
    
    def to_attn_conf(self) -> Qwen2AttentionConfig:
        return Qwen2AttentionConfig(self.hidden_size, self.num_attention_heads, self.num_key_value_heads, self.max_position_embeddings, self.rope_theta, self.attention_dropout)

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2DecoderLayerConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config.to_attn_conf(), layer_idx)

        self.mlp = Qwen2MLP(config.to_mlp_conf())
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = hidden_states
        
        return outputs


@dataclass
class Qwen2Config(nn.Module):
    hidden_size: int
    pad_token_id: Optional[int]
    vocab_size: int
    num_hidden_layers: int
    # rms_conf
    rms_norm_eps: float
    # mlp_conf
    intermediate_size: int
    hidden_act: Literal["silu"]
    # attn_conf
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    attention_dropout: float
    # rope_conf
    scaling_factor: float=1.0
    rope_type: str="default"

    def to_decoder_conf(self) -> Qwen2DecoderLayerConfig:
        return Qwen2DecoderLayerConfig(self.rms_norm_eps,
                                       self.hidden_size,
                                       self.intermediate_size,
                                       self.hidden_act,
                                       self.num_attention_heads,
                                       self.num_key_value_heads,
                                       self.max_position_embeddings,
                                       self.rope_theta,
                                       self.attention_dropout)
    
    def to_rope_conf(self, device:Optional[torch.device]=None) -> Qwen2RotaryEmbeddingConfig:
        return Qwen2RotaryEmbeddingConfig(self.hidden_size,
                                          self.num_attention_heads,
                                          self.max_position_embeddings,
                                          device,
                                          self.rope_theta,
                                          self.scaling_factor,
                                          self.rope_type)

class Qwen2Model(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config.to_decoder_conf(), layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config.to_rope_conf())

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Tensor,
        position_ids: torch.LongTensor
    ) -> Tensor:
        inputs_embeds = self.embed_tokens(input_ids)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        output = hidden_states

        return output

    def _update_causal_mask(
        self,
        attention_mask: Tensor,
        input_tensor: torch.LongTensor,
    ):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_of_the_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = attention_mask.shape[-1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_of_the_dtype=min_of_the_dtype,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_of_the_dtype: float,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_of_the_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_of_the_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_of_the_dtype
            )

    return causal_mask

class Qwen2ForCausalLM(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def sync_from_pretrained_model(self, model: nn.Module):
        state_dict = model.state_dict()
        self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Tensor,
        position_ids: torch.LongTensor,
        num_logits_to_keep: int = 0,
    ) -> Tensor:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        return logits

class QwenCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=151643,
    ):
        """
        Arguments:
            ignore_index: int. If labels == ignore_index, the loss is set to 0.0.
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor):
        if len(input.shape) == 3:
            input = input.reshape(-1, input.size(-1))
            target = target.reshape(-1)
        loss = torch.nn.functional.cross_entropy(
            input,
            target,
            ignore_index=self.ignore_index,
        )
        loss = loss.sum() / (target != self.ignore_index).sum()

        return loss

def _test():
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from transformers import Qwen2ForCausalLM as Qwen2ForCausalLMOld

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    input_text = "Once upon a time,"
    input_ids, attention_mask = tokenizer(input_text, return_tensors="pt").values()
    input_ids: Tensor
    attention_mask: Tensor
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    print("\n"*2)
    print('='*30 + 'test_old_qwen_modeling' + '='*30)
    
    model_old: Qwen2ForCausalLMOld = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", attn_implementation = "eager")

    logits_old = model_old.forward(input_ids, attention_mask, position_ids).logits

    print(logits_old.shape)


    print("\n"*2)
    print('='*30 + 'test_my_qwen_modeling' + '='*30)
    conf = Qwen2Config(hidden_size=1536,
                       pad_token_id=None,
                       vocab_size=151936,
                       num_hidden_layers=28,
                       rms_norm_eps=1e-06,
                       intermediate_size=8960,
                       hidden_act="silu",
                       num_attention_heads=12,
                       num_key_value_heads=2,
                       max_position_embeddings=131072,
                       rope_theta=1000000.0,
                       attention_dropout=0.0
                       )
    model = Qwen2ForCausalLM(conf)
    model.sync_from_pretrained_model(model_old)

    logits = model.forward(input_ids, attention_mask, position_ids)

    print(logits.shape)


    print("\n"*2)
    print('='*30 + 'test_my_modeling_equals_to_old' + '='*30)
    assert torch.equal(logits, logits_old)

    print("\n"*2)
    print('='*30 + 'test_my_cross_entropy_loss' + '='*30)
    input = torch.randn(8, 10, 15, requires_grad=True)
    target = torch.randint(15, (8, 10), dtype=torch.int64)
    print(QwenCrossEntropyLoss()(input, target))

if __name__ == '__main__':
    _test()
