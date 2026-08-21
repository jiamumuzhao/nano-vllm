import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        split_kv_config=None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self._validation_capture = None

        linear_quantization = (split_kv_config or {}).get("quantization", "none")
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quantization=linear_quantization,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quantization=linear_quantization,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            split_kv_config=split_kv_config,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self._validation_capture is not None:
            self._validation_capture("q_projection", q)
            self._validation_capture("k_projection", k)
            self._validation_capture("v_projection", v)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            if self._validation_capture is not None:
                self._validation_capture("q_norm_input", q)
                self._validation_capture("k_norm_input", k)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if self._validation_capture is not None:
                self._validation_capture("q_norm_output", q)
                self._validation_capture("k_norm_output", k)
        q, k = self.rotary_emb(positions, q, k)
        if self._validation_capture is not None:
            self._validation_capture("rope_q", q)
            self._validation_capture("rope_k", k)
        o = self.attn(q, k, v)
        if self._validation_capture is not None:
            self._validation_capture("attention_context", o)
        output = self.o_proj(o.flatten(1, -1))
        if self._validation_capture is not None:
            self._validation_capture("o_proj_output", output)
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quantization: str = "none",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quantization=quantization,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quantization=quantization,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()
        self._validation_capture = None
        # Validation-only A/B switch.  Production FP16 keeps the original
        # merged gate/up projection; the verifier may enable this explicitly
        # when comparing GEMM ordering against Transformers.
        self._validation_split_gates = False

    def forward(self, x):
        if self.gate_up_proj.quantization == "none" and self._validation_split_gates:
            # Transformers invokes gate_proj and up_proj as two independent
            # FP16 GEMMs. Matching that ordering avoids a different GEMM tile
            # accumulation in the first MLP divergence while leaving the
            # quantized/TP path on its existing packed implementation.
            split = self.gate_up_proj.weight.shape[0] // 2
            gate = F.linear(x, self.gate_up_proj.weight[:split], None)
            up = F.linear(x, self.gate_up_proj.weight[split:], None)
            gate_up = torch.cat((gate, up), dim=-1)
            if self._validation_capture is not None:
                self._validation_capture("gate_up_projection", gate_up)
                self._validation_capture("gate_projection", gate)
                self._validation_capture("up_projection", up)
                self._validation_capture("gate_up_proj", gate_up)
            # Keep the reference module's operation boundary: apply SiLU to
            # gate before multiplying by up.  Going through a concatenated
            # tensor and chunking it again can select a different fused
            # kernel/rounding sequence on FP16 CUDA.
            x = F.silu(gate) * up
        else:
            gate_up = self.gate_up_proj(x)
            if self._validation_capture is not None:
                self._validation_capture("gate_up_projection", gate_up)
                self._validation_capture("gate_up_proj", gate_up)
            x = self.act_fn(gate_up)
        if self._validation_capture is not None:
            self._validation_capture("silu_times_up", x)
            self._validation_capture("silu_activation", x)
        x = self.down_proj(x)
        if self._validation_capture is not None:
            self._validation_capture("down_proj", x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        split_kv_config=None,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            split_kv_config=split_kv_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quantization=(split_kv_config or {}).get("quantization", "none"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._validation_capture = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._validation_capture is not None:
            self._validation_capture("layer_input", hidden_states if residual is None else hidden_states + residual)
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self._validation_capture is not None:
            self._validation_capture("input_rmsnorm_output", hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        if self._validation_capture is not None:
            self._validation_capture("first_residual_add", hidden_states + residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self._validation_capture is not None:
            self._validation_capture("post_attention_rmsnorm_output", hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self._validation_capture is not None:
            self._validation_capture("second_residual_add", hidden_states + residual)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        split_kv_config=None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, split_kv_config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        split_kv_config=None,
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config, split_kv_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
