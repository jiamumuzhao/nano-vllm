#!/usr/bin/env python3
"""TP=1, token-by-token Qwen3 logits comparison against Transformers.

This is a correctness verifier, not a serving or throughput benchmark.  Nano's
validation hook returns logits before sampling; the normal serving path is not
changed.  The two implementations are driven with the same prompt and
deterministic continuation token ids.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_LENGTHS = (1, 16, 17, 31, 32, 128, 1024)
DEFAULT_MIXED_LENGTHS = (17, 32, 127)


def validate_tensor_parallel_size(value: int) -> None:
    if value != 1:
        raise ValueError("logits alignment currently supports tensor_parallel_size=1 only")


def valid_token_mask(lengths: list[int] | tuple[int, ...], max_len: int | None = None) -> torch.Tensor:
    """Return a [batch, max_len] mask excluding padding positions."""
    if not lengths or any(not isinstance(x, int) or x <= 0 for x in lengths):
        raise ValueError("lengths must contain positive integers")
    width = max_len if max_len is not None else max(lengths)
    if width < max(lengths):
        raise ValueError("max_len cannot be smaller than a valid sequence length")
    return torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


def compute_metrics(actual: torch.Tensor, expected: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    """Compute auditable FP32 vocabulary-logit metrics on valid positions only."""
    if actual.shape != expected.shape:
        raise ValueError(f"logit shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
    if actual.ndim < 2:
        raise ValueError("logits must have at least [tokens, vocabulary] dimensions")
    if mask is not None:
        if mask.shape != actual.shape[:-1]:
            raise ValueError(f"mask shape mismatch: mask={tuple(mask.shape)} logits={tuple(actual.shape)}")
        actual = actual[mask]
        expected = expected[mask]
    else:
        actual = actual.reshape(-1, actual.shape[-1])
        expected = expected.reshape(-1, expected.shape[-1])
    if actual.numel() == 0:
        raise ValueError("no valid logits selected")
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rmse = float(torch.sqrt((diff * diff).mean()).item())
    top1 = float((actual.argmax(dim=-1) == expected.argmax(dim=-1)).float().mean().item())
    return {
        "elements": int(actual.numel()),
        "logit_rows": int(actual.shape[0]),
        "vocabulary_size": int(actual.shape[1]),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "top1_token_agreement": top1,
    }


def packed_to_padded(packed: torch.Tensor, lengths: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Map Nano's packed [sum(lengths), ...] rows to [batch, max_len, ...]."""
    if packed.shape[0] != sum(lengths):
        raise ValueError(f"packed row count {packed.shape[0]} does not match lengths={lengths}")
    result = packed.new_zeros((len(lengths), max(lengths), *packed.shape[1:]))
    offset = 0
    for row, length in enumerate(lengths):
        result[row, :length] = packed[offset:offset + length]
        offset += length
    return result


def layerwise_record(case: int, phase: str, layer: str, actual: torch.Tensor, expected: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    record = {
        "record_type": "layerwise",
        "case": case,
        "phase": phase,
        "layer": layer,
        "metrics": compute_metrics(actual, expected, mask),
    }
    if mask is not None:
        actual_rows, expected_rows = actual[mask], expected[mask]
    else:
        actual_rows, expected_rows = actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, expected.shape[-1])
    diff = (actual_rows.float() - expected_rows.float()).abs()
    flat_index = int(diff.argmax().item())
    row_index, feature_index = divmod(flat_index, diff.shape[-1])
    record["max_error_location"] = {
        "row": row_index,
        "feature": feature_index,
        "actual": float(actual_rows[row_index, feature_index].item()),
        "expected": float(expected_rows[row_index, feature_index].item()),
    }
    return record


V2_BOUNDARY_ORDER = (
    "layer_input", "input_rmsnorm_output", "q_projection", "k_projection",
    "v_projection", "q_norm_output", "k_norm_output", "rope_q", "rope_k", "attention_context", "o_proj_output",
    "first_residual_add", "post_attention_rmsnorm_output", "gate_projection",
    "up_projection", "silu_times_up", "down_proj_output", "second_residual_add",
    "final_rmsnorm_output", "logits",
)

QK_NORM_BOUNDARIES = {"q_norm_output", "k_norm_output"}


def layerwise_v2_record(case: int, phase: str, layer: str, boundary: str,
                        actual: torch.Tensor, expected: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        layout: str = "[batch, tokens, features]") -> dict:
    """Create a v2 record only after validating equivalent tensor semantics."""
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"layerwise-v2 shape mismatch case={case} phase={phase} "
            f"layer={layer} boundary={boundary}: nano={tuple(actual.shape)} "
            f"transformers={tuple(expected.shape)}"
        )
    record = layerwise_record(case, phase, f"{layer}.{boundary}", actual, expected, mask)
    record["schema"] = "logits_alignment_v2"
    record["boundary"] = boundary
    record["layout"] = layout
    record["dtype"] = {"nano": str(actual.dtype), "transformers": str(expected.dtype)}
    record["shape"] = {"nano": list(actual.shape), "transformers": list(expected.shape)}
    record["valid_mask"] = None if mask is None else list(mask.shape)
    return record


def layerwise_v3_record(case: int, phase: str, layer: str, boundary: str,
                        actual: torch.Tensor, expected: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        normalization_input: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict:
    """Strict full-model boundary record with complete tensor metadata."""
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"layerwise-v3 shape mismatch case={case} phase={phase} layer={layer} "
            f"boundary={boundary}: nano={tuple(actual.shape)} hf={tuple(expected.shape)}"
        )
    if actual.ndim < 2:
        raise ValueError(f"layerwise-v3 boundary must have token/features dimensions: {layer}.{boundary}")
    metric_actual = actual.reshape(actual.shape[0], actual.shape[1], -1) if actual.ndim > 3 else actual
    metric_expected = expected.reshape(expected.shape[0], expected.shape[1], -1) if expected.ndim > 3 else expected
    record = layerwise_record(case, phase, f"{layer}.{boundary}", metric_actual, metric_expected, mask)
    record.update({
        "schema": "logits_alignment_v3",
        "boundary": boundary,
        "shape": {"nano": list(actual.shape), "transformers": list(expected.shape)},
        "dtype": {"nano": str(actual.dtype), "transformers": str(expected.dtype)},
        "layout": "[batch,tokens,features] or [batch,tokens,heads,head_dim]",
        "valid_token_mask": "valid rows only; padding excluded" if mask is not None else "all rows",
    })
    if boundary in QK_NORM_BOUNDARIES:
        if actual.ndim < 3 or actual.shape[-1] <= 0:
            raise ValueError(f"invalid Q/K norm output shape: {tuple(actual.shape)}")
        if normalization_input is None:
            raise ValueError(f"missing Q/K norm input for {layer}.{boundary}")
        norm_actual, norm_expected = normalization_input
        if tuple(norm_actual.shape) != tuple(norm_expected.shape):
            raise ValueError(
                f"Q/K norm input shape mismatch for {layer}.{boundary}: "
                f"nano={tuple(norm_actual.shape)} hf={tuple(norm_expected.shape)}"
            )
        record.update({
            "normalization_dim": -1,
            "normalization_size": int(actual.shape[-1]),
            "input_shape": {"nano": list(norm_actual.shape), "transformers": list(norm_expected.shape)},
            "output_shape": {"nano": list(actual.shape), "transformers": list(expected.shape)},
            # Both sides are loaded and executed in the verifier's FP16 model;
            # snapshots are promoted to FP32 only for stable metric arithmetic.
            "input_dtype": {"nano": "torch.float16", "transformers": "torch.float16"},
            "output_dtype": {"nano": "torch.float16", "transformers": "torch.float16"},
            "normalization_semantics": "RMSNorm over head_dim only; no cross-token/head reduction",
        })
        a = actual.float()
        e = expected.float()
        diff = (a - e).abs()
        if mask is not None:
            diff = diff[mask]
        # [rows, heads, head_dim] after removing batch/token axes.
        if diff.ndim == 2:
            diff = diff.unsqueeze(1)
        elif diff.ndim > 3:
            diff = diff.reshape(-1, diff.shape[-2], diff.shape[-1])
        per_head = []
        for head in range(diff.shape[1]):
            values = diff[:, head, :]
            per_head.append({
                "head": head,
                "max_abs_error": float(values.max().item()),
                "mean_abs_error": float(values.mean().item()),
                "rmse": float(torch.sqrt((values * values).mean()).item()),
            })
        record["per_head_error_summary"] = per_head
    record["layer"] = layer
    return record


def passes_thresholds(metrics: dict, thresholds: dict) -> bool:
    return (
        metrics["max_abs_error"] <= thresholds["max_abs_error"]
        and metrics["mean_abs_error"] <= thresholds["mean_abs_error"]
        and metrics["top1_token_agreement"] == thresholds["top1_token_agreement"]
    )


def first_layerwise_exceedance(records: list[dict], threshold: float = 5e-2) -> dict | None:
    order = [
        "layer_2.attention_residual_input",
        "layer_2.post_attention_layernorm",
        "layer_2.gate_up_proj",
        "layer_2.silu_activation",
        "layer_2.down_proj",
        "layer_2.decoder_layer_output",
    ]
    rank = {name: i for i, name in enumerate(order)}
    candidates = [r for r in records if r.get("layer") in rank and r["metrics"]["max_abs_error"] > threshold]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (r["case"], 0 if r["phase"] == "prefill" else 1, rank[r["layer"]]))


def build_layerwise_v2_records(records: list[dict]) -> list[dict]:
    """Convert only semantically validated layer-2 observations to v2 records.

    Legacy module-level observations are intentionally not relabeled as v2
    boundaries. Missing internal boundaries remain explicit in the contract
    record instead of being compared as tuple/layout-mismatched tensors.
    """
    mapping = {
        "layer_2.attention_residual_input": "layer_input",
        "layer_2.post_attention_layernorm": "post_attention_rmsnorm_output",
        "layer_2.gate_up_proj": "gate_up_projection_combined",
        "layer_2.silu_activation": "silu_times_up",
        "layer_2.down_proj": "down_proj_output",
        "layer_2.decoder_layer_output": "second_residual_add",
        "final_norm": "final_rmsnorm_output",
        "logits": "logits",
    }
    result = []
    for record in records:
        boundary = mapping.get(record.get("layer"))
        if boundary is None:
            continue
        item = dict(record)
        item["schema"] = "logits_alignment_v2"
        item["boundary"] = boundary
        item["layout"] = "[batch, tokens, features]"
        item["shape"] = {"metrics_elements": record["metrics"]["elements"]}
        item["dtype"] = {"nano": "float32_snapshot", "transformers": "float32_snapshot"}
        item["valid_mask"] = "applied_to_valid_token_rows"
        result.append(item)
    return result


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(text)
        tmp = Path(f.name)
    tmp.replace(path)


def parse_lengths(value: str) -> list[int]:
    try:
        result = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid length list: {value!r}") from exc
    if not result or any(x <= 0 for x in result):
        raise ValueError("length list must contain positive integers")
    return result


def deterministic_prompt(length: int, vocab_size: int) -> list[int]:
    return [1 + ((17 * i + 23) % max(2, vocab_size - 1)) for i in range(length)]


def deterministic_token(case_index: int, sequence_index: int, step: int, vocab_size: int) -> int:
    return 1 + ((case_index * 997 + sequence_index * 97 + step * 31) % max(2, vocab_size - 1))


def environment_info(model_path: str, tp: int, dtype: str) -> dict:
    info = {
        "model": model_path,
        "tensor_parallel_size": tp,
        "dtype": dtype,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "git_commit": None,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    return info


@dataclass
class NanoCase:
    lengths: list[int]
    prefill_logits: torch.Tensor
    decode_logits: torch.Tensor
    prefill_layers: dict[str, torch.Tensor] | None = None
    decode_layers: dict[str, torch.Tensor] | None = None
    prefill_boundaries: dict[str, torch.Tensor] | None = None
    decode_boundaries: dict[str, torch.Tensor] | None = None


def _first_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def install_layerwise_hooks(model, names: dict[str, str]):
    captured = {}
    handles = []
    for logical_name, module_name in names.items():
        module = dict(model.named_modules()).get(module_name)
        if module is None:
            raise RuntimeError(f"layerwise module not found: {module_name}")
        handles.append(module.register_forward_hook(
            lambda _m, _i, output, n=logical_name: captured.__setitem__(
                n, _first_tensor(output).detach().float().cpu())))
    return captured, handles


def snapshot_layers(captured: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in captured.items():
        value = _normalize_layer_output(name, value)
        result[name] = value.detach().float().cpu()
    # Do not retain the last GPU activation through the next forward. Layerwise
    # diagnostics are CPU records, not a persistent intermediate workspace.
    captured.clear()
    return result


def _normalize_layer_output(name: str, value: torch.Tensor) -> torch.Tensor:
    """Normalize HF attention's [B,H,T,D] to Nano's [B,T,H*D]."""
    if name.endswith(".rope_q") or name.endswith(".rope_k"):
        if value.ndim == 4:
            return value.transpose(1, 2).reshape(value.shape[0], value.shape[2], -1)
        if value.ndim == 3:
            return value.reshape(value.shape[0], -1)
    if name.endswith(".attention") and value.ndim == 4:
        return value.transpose(1, 2).reshape(value.shape[0], value.shape[2], -1)
    return value


def _decode_layer_row(name: str, value: torch.Tensor) -> torch.Tensor:
    while value.ndim > 2 and value.shape[1] == 1:
        value = value[:, 0]
    return value


def nano_layer_names(model) -> dict[str, str]:
    names = {"embedding": "model.embed_tokens", "final_norm": "model.norm"}
    for i in range(len(model.model.layers)):
        names[f"layer_{i}.attention"] = f"model.layers.{i}.self_attn"
        names[f"layer_{i}.mlp"] = f"model.layers.{i}.mlp"
    return names


def hf_layer_names(model) -> dict[str, str]:
    names = {"embedding": "model.embed_tokens", "final_norm": "model.norm"}
    for i in range(len(model.model.layers)):
        names[f"layer_{i}.attention"] = f"model.layers.{i}.self_attn"
        names[f"layer_{i}.mlp"] = f"model.layers.{i}.mlp"
    return names


def install_layer2_boundary_hooks(model, hf: bool):
    """Capture equivalent layer-2 decoder boundaries without persistent buffers."""
    layer = model.model.layers[2]
    captured = {}
    handles = []

    def save(name, value):
        captured[name] = _first_tensor(value).detach().float().cpu()

    def layer_pre(_module, args):
        if hf:
            hidden = args[0]
            residual = None
        else:
            # Nano Qwen3DecoderLayer.forward(positions, hidden_states, residual)
            hidden = args[1]
            residual = args[2] if len(args) > 2 else None
        save("layer_2.attention_residual_input", hidden if residual is None else hidden + residual)

    handles.append(layer.register_forward_pre_hook(layer_pre))
    handles.append(layer.post_attention_layernorm.register_forward_hook(lambda _m, _i, out: save("layer_2.post_attention_layernorm", out)))

    if hf:
        handles.append(layer.mlp.down_proj.register_forward_hook(lambda _m, _i, out: save("layer_2.down_proj", out)))
        gate = {}
        up = {}

        def gate_hook(_m, _i, out):
            gate["value"] = _first_tensor(out)
            if "value" in up and up["value"].shape[:-1] == gate["value"].shape[:-1]:
                save("layer_2.gate_up_proj", torch.cat((gate["value"], up["value"]), dim=-1))
                save("layer_2.silu_activation", torch.nn.functional.silu(gate["value"]) * up["value"])

        def up_hook(_m, _i, out):
            up["value"] = _first_tensor(out)
            if "value" in gate and gate["value"].shape[:-1] == up["value"].shape[:-1]:
                save("layer_2.gate_up_proj", torch.cat((gate["value"], up["value"]), dim=-1))
                save("layer_2.silu_activation", torch.nn.functional.silu(gate["value"]) * up["value"])

        handles.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))
        handles.append(layer.mlp.up_proj.register_forward_hook(up_hook))
    else:
        # Nano's FP16 validation path deliberately performs gate_proj and
        # up_proj as two independent GEMMs.  Capture the exact semantic
        # boundaries from Qwen3MLP rather than hooking the fused Linear/act_fn
        # modules, which are not necessarily invoked on that path.
        layer.mlp._validation_split_gates = True
        layer.mlp._validation_capture = lambda name, value: save(f"layer_2.{name}", value)

    def layer_hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            save("layer_2.decoder_layer_output", output[0] + output[1])
        else:
            save("layer_2.decoder_layer_output", output)

    handles.append(layer.register_forward_hook(layer_hook))
    return captured, handles


def install_v3_boundary_hooks(model, hf: bool, split_gates: bool = False):
    """Capture equivalent decoder boundaries for every layer in execution order."""
    captured, handles = {}, []

    def save(name, value):
        # v3 is diagnostics-only; copy each boundary immediately so capturing
        # every layer does not retain a full set of GPU activations.
        captured[name] = _first_tensor(value).detach().float().cpu()

    layers = model.model.layers
    if hf:
        # Current Transformers Qwen3 applies RoPE as a functional call rather
        # than a module, so there is no rotary_emb hook to register.  Wrap the
        # exact function only for validation and restore it in the returned
        # cleanup handle; this captures the post-RoPE tensors without changing
        # production code or attention math.
        import transformers.models.qwen3.modeling_qwen3 as qwen3_impl
        original_rope = qwen3_impl.apply_rotary_pos_emb
        state = {"calls": 0}
        def wrapped_rope(query, key, cos, sin, *args, **kwargs):
            index = state["calls"] % len(layers)
            state["calls"] += 1
            query, key = original_rope(query, key, cos, sin, *args, **kwargs)
            save(f"layer_{index}.rope_q", query)
            save(f"layer_{index}.rope_k", key)
            return query, key
        qwen3_impl.apply_rotary_pos_emb = wrapped_rope
        class _RestoreRope:
            def remove(self):
                qwen3_impl.apply_rotary_pos_emb = original_rope
        handles.append(_RestoreRope())
    for index, layer in enumerate(layers):
        prefix = f"layer_{index}."
        if not hf:
            def callback(name, value, p=prefix):
                normalized = {
                    "gate_up_proj": "gate_up_projection",
                    "silu_activation": "silu_times_up",
                    "down_proj": "down_proj_output",
                }.get(name, name)
                save(p + normalized, value)
            layer._validation_capture = callback
            layer.self_attn._validation_capture = callback
            layer.mlp._validation_capture = callback
            layer.mlp._validation_split_gates = bool(split_gates)
            continue

        def layer_pre(_module, args, p=prefix):
            save(p + "layer_input", args[0])
        handles.append(layer.register_forward_pre_hook(layer_pre))
        handles.append(layer.input_layernorm.register_forward_hook(
            lambda _m, _i, out, p=prefix: save(p + "input_rmsnorm_output", out)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda _m, args, p=prefix: save(p + "first_residual_add", args[0])))
        handles.append(layer.post_attention_layernorm.register_forward_hook(
            lambda _m, _i, out, p=prefix: save(p + "post_attention_rmsnorm_output", out)))
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            handles.append(getattr(attn, name).register_forward_hook(
                lambda _m, _i, out, n=name, p=prefix: save(p + n.replace("_proj", "_projection"), out)))
        for name in ("q_norm", "k_norm"):
            if hasattr(attn, name):
                handles.append(getattr(attn, name).register_forward_pre_hook(
                    lambda _m, args, n=name, p=prefix: save(
                        p + n.replace("_norm", "_norm_input"), args[0])))
                handles.append(getattr(attn, name).register_forward_hook(
                    lambda _m, _i, out, n=name, p=prefix: save(p + n.replace("_norm", "_norm_output"), out)))
        # HF's o_proj input is the post-attention context in [B,T,H*D].
        def context_hook(_m, args, p=prefix, a=attn):
            value = args[0]
            if value.ndim == 3:
                heads = getattr(a, "num_heads", getattr(a.config, "num_attention_heads", 1))
                head_dim = getattr(a, "head_dim", value.shape[-1] // heads)
                value = value.reshape(value.shape[0], value.shape[1], heads, head_dim)
            save(p + "attention_context", value)
        handles.append(attn.o_proj.register_forward_pre_hook(context_hook))
        handles.append(attn.o_proj.register_forward_hook(
            lambda _m, _i, out, p=prefix: save(p + "o_proj_output", out)))
        gate_up = {}
        for name in ("gate_proj", "up_proj"):
            def mlp_proj(_m, _i, out, n=name, p=prefix, state=gate_up):
                state[n] = _first_tensor(out)
                save(p + ("gate_projection" if n == "gate_proj" else "up_projection"), out)
                if "gate_proj" in state and "up_proj" in state:
                    if state["gate_proj"].shape != state["up_proj"].shape:
                        raise ValueError(f"layerwise-v3 gate/up shape mismatch at {p}")
                    save(p + "gate_up_projection", torch.cat((state["gate_proj"], state["up_proj"]), dim=-1))
                    save(p + "silu_times_up", torch.nn.functional.silu(state["gate_proj"]) * state["up_proj"])
            handles.append(getattr(layer.mlp, name).register_forward_hook(mlp_proj))
        handles.append(layer.mlp.register_forward_pre_hook(lambda _m, _args, state=gate_up: state.clear()))
        handles.append(layer.mlp.down_proj.register_forward_hook(
            lambda _m, _i, out, p=prefix: save(p + "down_proj_output", out)))

        def layer_out(_module, _inputs, output, p=prefix):
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                output = output[0] + output[1]
            save(p + "second_residual_add", output)
        handles.append(layer.register_forward_hook(layer_out))
    return captured, handles


def collect_nano_cases(model_path: str, cases: list[list[int]], decode_steps: int, gpu_memory_utilization: float, debug_layerwise: bool = False, validation_split_gates: bool = False) -> tuple[list[NanoCase], dict]:
    from multiprocessing import Event
    from nanovllm.config import Config
    from nanovllm.engine.block_manager import BlockManager
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    from transformers import AutoTokenizer

    max_prompt = max(max(lengths) for lengths in cases)
    config = Config(
        model_path,
        max_num_batched_tokens=sum(max(lengths) for lengths in cases),
        max_num_seqs=max(len(lengths) for lengths in cases),
        max_model_len=max(128, max_prompt + decode_steps + 2),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="float16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config.eos = tokenizer.eos_token_id
    Sequence.block_size = config.kvcache_block_size
    runner = ModelRunner(config, 0, Event())
    nano_names = nano_layer_names(runner.model)
    captured, handles = install_layerwise_hooks(runner.model, nano_names) if debug_layerwise else ({}, [])
    boundary_captured, boundary_handles = install_v3_boundary_hooks(runner.model, hf=False, split_gates=validation_split_gates) if debug_layerwise else ({}, [])
    result: list[NanoCase] = []
    try:
        vocab_size = config.hf_config.vocab_size
        for case_index, lengths in enumerate(cases):
            scheduler = Scheduler(config)
            prompts = [deterministic_prompt(length, vocab_size) for length in lengths]
            seqs = []
            for prompt in prompts:
                # The validation hook never invokes Sampler. SamplingParams
                # still requires a positive placeholder temperature; token
                # progression below is deterministic and shared explicitly.
                seqs.append(Sequence(prompt, SamplingParams(max_tokens=decode_steps + 1, temperature=1.0, ignore_eos=True)))
            for seq in seqs:
                scheduler.add(seq)
            scheduled, is_prefill = scheduler.schedule()
            if not is_prefill:
                raise RuntimeError(f"expected prefill schedule for lengths={lengths}")
            prefill = runner.run_logits_for_validation(scheduled, True).float().cpu()
            prefill_layers = {name: packed_to_padded(value, lengths) for name, value in snapshot_layers(captured).items()} if debug_layerwise else None
            prefill_boundaries = {name: packed_to_padded(value, lengths) for name, value in snapshot_layers(boundary_captured).items()} if debug_layerwise else None
            next_tokens = [deterministic_token(case_index, i, 0, vocab_size) for i in range(len(seqs))]
            scheduler.postprocess(scheduled, next_tokens, True)
            decode_rows = []
            decode_layer_rows = {name: [] for name in nano_names} if debug_layerwise else None
            decode_boundary_rows = {name: [] for name in prefill_boundaries} if debug_layerwise else None
            for step in range(decode_steps):
                scheduled, is_prefill = scheduler.schedule()
                if is_prefill or len(scheduled) != len(seqs):
                    raise RuntimeError(f"unexpected decode schedule for lengths={lengths}, step={step}")
                decode_rows.append(runner.run_logits_for_validation(scheduled, False).float().cpu())
                if debug_layerwise:
                    for name, value in snapshot_layers(captured).items():
                        decode_layer_rows[name].append(_decode_layer_row(name, value))
                    for name, value in snapshot_layers(boundary_captured).items():
                        decode_boundary_rows[name].append(_decode_layer_row(name, value))
                next_tokens = [deterministic_token(case_index, i, step + 1, vocab_size) for i in range(len(seqs))]
                scheduler.postprocess(scheduled, next_tokens, False)
            decode_layers = {name: torch.stack(values, dim=1) for name, values in decode_layer_rows.items()} if debug_layerwise else None
            decode_boundaries = {name: torch.stack(values, dim=1) for name, values in decode_boundary_rows.items()} if debug_layerwise else None
            result.append(NanoCase(lengths, packed_to_padded(prefill, lengths), torch.stack(decode_rows).transpose(0, 1).contiguous(), prefill_layers, decode_layers, prefill_boundaries, decode_boundaries))
            for seq in seqs:
                if seq.block_table:
                    scheduler.block_manager.deallocate(seq)
            if debug_layerwise:
                torch.cuda.empty_cache()
    finally:
        if debug_layerwise:
            for layer in runner.model.model.layers:
                layer._validation_capture = None
                layer.self_attn._validation_capture = None
                layer.mlp._validation_capture = None
                layer.mlp._validation_split_gates = False
        for handle in handles:
            handle.remove()
        for handle in boundary_handles:
            handle.remove()
        runner.exit()
        # ModelRunner.exit() tears down process-group/graph state; explicitly
        # drop validation-only model and KV references before loading the
        # separate Transformers model in the same process.
        runner.model = None
        if hasattr(runner, "kv_cache"):
            runner.kv_cache = None
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    return result, {"vocab_size": config.hf_config.vocab_size, "eos_token_id": config.eos}


def collect_transformers_case(model, lengths: list[int], case_index: int, decode_steps: int, vocab_size: int, debug_layerwise: bool = False):
    """Run one true padded batch; never split a mixed batch into single sequences."""
    batch, max_len = len(lengths), max(lengths)
    input_ids = torch.zeros((batch, max_len), device="cuda", dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), device="cuda", dtype=torch.long)
    position_ids = torch.zeros((batch, max_len), device="cuda", dtype=torch.long)
    for row, length in enumerate(lengths):
        input_ids[row, :length] = torch.tensor(deterministic_prompt(length, vocab_size), device="cuda")
        attention_mask[row, :length] = 1
        position_ids[row, :length] = torch.arange(length, device="cuda")
    hf_names = hf_layer_names(model)
    captured, handles = install_layerwise_hooks(model, hf_names) if debug_layerwise else ({}, [])
    boundary_captured, boundary_handles = install_v3_boundary_hooks(model, hf=True) if debug_layerwise else ({}, [])
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        prefill_logits = output.logits.float().cpu()
        # The full-vocabulary prefill logits are already copied to CPU; release
        # the GPU copy before the decode loop, especially for the 1024 case.
        output.logits = None
        prefill_layers = snapshot_layers(captured) if debug_layerwise else None
        prefill_boundaries = snapshot_layers(boundary_captured) if debug_layerwise else None
        past = output.past_key_values
        decode_rows = []
        decode_layer_rows = {name: [] for name in hf_names} if debug_layerwise else None
        decode_boundary_rows = {name: [] for name in prefill_boundaries} if debug_layerwise else None
        for step in range(decode_steps):
            tokens = torch.tensor(
                [[deterministic_token(case_index, row, step, vocab_size)] for row in range(batch)],
                device="cuda", dtype=torch.long,
            )
            positions = torch.tensor([[length + step] for length in lengths], device="cuda", dtype=torch.long)
            decode_mask = torch.zeros((batch, max_len + step + 1), device="cuda", dtype=torch.long)
            for row, length in enumerate(lengths):
                decode_mask[row, :length] = 1
                decode_mask[row, max_len:max_len + step + 1] = 1
            with torch.inference_mode():
                output = model(
                    input_ids=tokens,
                    position_ids=positions,
                    attention_mask=decode_mask,
                    past_key_values=past,
                    cache_position=torch.full((batch,), max_len + step, device="cuda", dtype=torch.long),
                    use_cache=True,
                    return_dict=True,
                )
            decode_rows.append(output.logits[:, 0].float().cpu())
            output.logits = None
            if debug_layerwise:
                for name, value in snapshot_layers(captured).items():
                    decode_layer_rows[name].append(_decode_layer_row(name, value))
                for name, value in snapshot_layers(boundary_captured).items():
                    decode_boundary_rows[name].append(_decode_layer_row(name, value))
            past = output.past_key_values
        decode_logits = torch.stack(decode_rows, dim=1)
        decode_layers = {name: torch.stack(values, dim=1) for name, values in decode_layer_rows.items()} if debug_layerwise else None
        decode_boundaries = {name: torch.stack(values, dim=1) for name, values in decode_boundary_rows.items()} if debug_layerwise else None
        return prefill_logits, decode_logits, prefill_layers, decode_layers, prefill_boundaries, decode_boundaries
    finally:
        for handle in handles:
            handle.remove()
        for handle in boundary_handles:
            handle.remove()


def markdown(records: list[dict], summary: dict, env: dict, thresholds: dict, layerwise_records: list[dict] | None = None) -> str:
    lines = [
        "# Nano-vLLM vs Transformers logits alignment",
        "",
        "This is a TP=1 correctness verification, not a performance benchmark. The Qwen3-0.6B model uses GQA; MHA is covered by synthetic attention tests.",
        "",
        f"- Model: `{env['model']}`",
        f"- GPU: `{env['gpu']}`; torch `{env['torch']}`; CUDA `{env['cuda_runtime']}`",
        f"- Thresholds: max abs `<= {thresholds['max_abs_error']}`, mean abs `<= {thresholds['mean_abs_error']}`, top-1 `== {thresholds['top1_token_agreement']}`",
        f"- Overall: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "| Case | Phase | Rows | Elements | Max abs | Mean abs | RMSE | Top-1 | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        m = r["metrics"]
        lines.append(
            f"| `{r['lengths']}` | {r['phase']} | {m['logit_rows']} | {m['elements']} | "
            f"{m['max_abs_error']:.6g} | {m['mean_abs_error']:.6g} | {m['rmse']:.6g} | "
            f"{m['top1_token_agreement']:.6f} | {'PASS' if r['passed'] else 'FAIL'} |"
        )
    lines.extend(["", "The decode rows use the same deterministic continuation token ids on both implementations; mixed-batch padding is excluded by construction."])
    if layerwise_records:
        first = next((r for r in layerwise_records if r.get("metrics", {}).get("max_abs_error", 0) > thresholds["max_abs_error"]), None)
        if first is None:
            first = first_layerwise_exceedance(layerwise_records, thresholds["max_abs_error"])
        if first is None:
            lines.append("Layerwise v3 diagnostic: no full-model boundary exceeded the max-error threshold.")
        else:
            m = first["metrics"]
            lines.append(
                f"First full-model boundary over max-error threshold: `{first['layer']}.{first.get('boundary', '')}` "
                f"(case={first['case']}, phase={first['phase']}, max_abs={m['max_abs_error']:.6g}, "
                f"mean_abs={m['mean_abs_error']:.6g}, top1={m['top1_token_agreement']:.6f})."
            )
        lines.append("Layerwise v2 applies strict equal-shape packed/padded validation; its contract and records are written to `logits_alignment_qwen3_tp1_layerwise_v2.jsonl`.")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    validate_tensor_parallel_size(args.tensor_parallel_size)
    if args.dtype != "float16":
        raise ValueError("the verifier currently supports dtype=float16 only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for整模型 logits alignment; CPU verification only covers pure helpers")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model path does not exist: {args.model}")
    lengths = parse_lengths(args.lengths)
    mixed = parse_lengths(args.mixed_lengths)
    cases = [[x] for x in lengths] + [mixed]
    thresholds = {
        "max_abs_error": args.max_abs_error,
        "mean_abs_error": args.mean_abs_error,
        "top1_token_agreement": 1.0,
    }
    env = environment_info(args.model, args.tensor_parallel_size, args.dtype)
    nano_cases, model_meta = collect_nano_cases(
        args.model, cases, args.decode_steps, args.gpu_memory_utilization,
        args.debug_layerwise, args.validation_split_gates,
    )
    gc.collect()
    torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM
    model_kwargs = {"torch_dtype": torch.float16}
    if args.hf_attn_implementation:
        model_kwargs["attn_implementation"] = args.hf_attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).cuda().eval()
    records = []
    layerwise_records = []
    try:
        for case_index, (case_lengths, nano) in enumerate(zip(cases, nano_cases)):
            hf_prefill, hf_decode, hf_prefill_layers, hf_decode_layers, hf_prefill_boundaries, hf_decode_boundaries = collect_transformers_case(
                model, case_lengths, case_index, args.decode_steps, model_meta["vocab_size"], args.debug_layerwise
            )
            mask = valid_token_mask(case_lengths, max(case_lengths))
            prefill_metrics = compute_metrics(nano.prefill_logits, hf_prefill, mask)
            decode_metrics = compute_metrics(nano.decode_logits, hf_decode)
            for phase, metrics in (("prefill", prefill_metrics), ("decode", decode_metrics)):
                record = {"case": case_index, "lengths": case_lengths, "phase": phase, "metrics": metrics, "passed": passes_thresholds(metrics, thresholds)}
                records.append(record)
                if not record["passed"]:
                    print(f"FAILED case={case_index} lengths={case_lengths} phase={phase} metrics={metrics}", file=sys.stderr)
            if args.debug_layerwise:
                for layer_name in sorted(nano.prefill_layers):
                    try:
                        layerwise_records.append(layerwise_record(case_index, "prefill", layer_name, nano.prefill_layers[layer_name], hf_prefill_layers[layer_name], mask))
                        layerwise_records.append(layerwise_record(case_index, "decode", layer_name, nano.decode_layers[layer_name], hf_decode_layers[layer_name]))
                    except ValueError as exc:
                        raise RuntimeError(
                            f"layerwise shape mismatch case={case_index} phase=decode layer={layer_name} "
                            f"nano={tuple(nano.decode_layers[layer_name].shape)} hf={tuple(hf_decode_layers[layer_name].shape)}"
                        ) from exc
                for boundary_name in nano.prefill_boundaries:
                    if boundary_name.rsplit(".", 1)[-1] in {"q_norm_input", "k_norm_input"}:
                        continue
                    layer, boundary = boundary_name.split(".", 1)
                    norm_input = None
                    if boundary in QK_NORM_BOUNDARIES:
                        stem = boundary.replace("_output", "_input")
                        norm_input = (nano.prefill_boundaries[f"{layer}.{stem}"], hf_prefill_boundaries[f"{layer}.{stem}"])
                    layerwise_records.append(layerwise_v3_record(case_index, "prefill", layer, boundary, nano.prefill_boundaries[boundary_name], hf_prefill_boundaries[boundary_name], mask, norm_input))
                    decode_norm_input = None
                    if boundary in QK_NORM_BOUNDARIES:
                        stem = boundary.replace("_output", "_input")
                        decode_norm_input = (nano.decode_boundaries[f"{layer}.{stem}"], hf_decode_boundaries[f"{layer}.{stem}"])
                    layerwise_records.append(layerwise_v3_record(case_index, "decode", layer, boundary, nano.decode_boundaries[boundary_name], hf_decode_boundaries[boundary_name], None, decode_norm_input))
                layerwise_records.append(layerwise_record(case_index, "prefill", "logits", nano.prefill_logits, hf_prefill, mask))
                layerwise_records.append(layerwise_record(case_index, "decode", "logits", nano.decode_logits, hf_decode))
    finally:
        del model
        torch.cuda.empty_cache()
    summary = {
        "passed": all(r["passed"] for r in records),
        "case_count": len(cases),
        "record_count": len(records),
        "max_abs_error": max(r["metrics"]["max_abs_error"] for r in records),
        "mean_abs_error": max(r["metrics"]["mean_abs_error"] for r in records),
        "rmse": max(r["metrics"]["rmse"] for r in records),
        "min_top1_token_agreement": min(r["metrics"]["top1_token_agreement"] for r in records),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "logits_alignment_qwen3_tp1.jsonl"
    md = out_dir / "logits_alignment_qwen3_tp1.md"
    lines = [{"record_type": "case", **r, "environment": env, "thresholds": thresholds} for r in records]
    lines.append({"record_type": "summary", "summary": summary, "environment": env, "thresholds": thresholds, "model_metadata": model_meta})
    atomic_write(jsonl, "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines))
    atomic_write(md, markdown(records, summary, env, thresholds, layerwise_records if args.debug_layerwise else None))
    if args.debug_layerwise:
        layerwise_path = out_dir / "logits_alignment_qwen3_tp1_layerwise.jsonl"
        atomic_write(layerwise_path, "".join(json.dumps(line, sort_keys=True) + "\n" for line in layerwise_records))
        v2_path = out_dir / "logits_alignment_qwen3_tp1_layerwise_v2.jsonl"
        v2_lines = [{
            "record_type": "schema",
            "schema": "logits_alignment_v2",
            "boundary_order": list(V2_BOUNDARY_ORDER),
            "semantic_policy": "only equal-shape [batch,tokens,features] tensors are compared; tuple/layout mismatches are rejected",
            "capture_scope": "layer_2 precise boundaries plus final norm/logits; legacy coarse module observations are not relabeled",
            "threshold": thresholds["max_abs_error"],
        }]
        v2_lines.extend(build_layerwise_v2_records(layerwise_records))
        # v1/v2 are historical evidence; never rewrite them during v3 runs.
        if not v2_path.exists():
            atomic_write(v2_path, "".join(json.dumps(line, sort_keys=True) + "\n" for line in v2_lines))
        v3_path = out_dir / "logits_alignment_qwen3_tp1_layerwise_v3.jsonl"
        ordered = sorted(
            [r for r in layerwise_records if r.get("schema") == "logits_alignment_v3"],
            key=lambda r: (
                r["case"],
                int(r["layer"].split("_")[1].split(".")[0]) if r["layer"].startswith("layer_") else 10**9,
                0 if r["phase"] == "prefill" else 1,
                list(V2_BOUNDARY_ORDER).index(r["boundary"]) if r.get("boundary") in V2_BOUNDARY_ORDER else 10**4,
            ),
        )
        first = next((r for r in ordered if r["metrics"]["max_abs_error"] > thresholds["max_abs_error"]), None)
        first_index = ordered.index(first) if first is not None else None
        v3_header = {
            "record_type": "metadata",
            "schema": "logits_alignment_v3",
            "production_mlp_path": "merged_gate_up -> SiluAndMul -> down_proj",
            "validation_split_gates": bool(args.validation_split_gates),
            "boundary_order": list(V2_BOUNDARY_ORDER),
            "first_divergence": first["layer"] + "." + first["boundary"] if first else None,
            "previous_boundary": ordered[first_index - 1]["layer"] + "." + ordered[first_index - 1]["boundary"] if first_index else None,
        }
        atomic_write(v3_path, json.dumps(v3_header, sort_keys=True) + "\n" + "".join(json.dumps(line, sort_keys=True) + "\n" for line in ordered))
        print(f"Layerwise JSONL: {layerwise_path}")
        print(f"Layerwise v2 JSONL: {v2_path}")
        print(f"Layerwise v3 JSONL: {v3_path}")
    print(f"JSONL: {jsonl}")
    print(f"Markdown: {md}")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument("--mixed-lengths", default=",".join(map(str, DEFAULT_MIXED_LENGTHS)))
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--max-abs-error", type=float, default=5e-2)
    parser.add_argument("--mean-abs-error", type=float, default=5e-3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--hf-attn-implementation", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--debug-layerwise", action="store_true")
    parser.add_argument("--validation-split-gates", action="store_true", help="validation-only A/B path; never enabled by --debug-layerwise")
    parser.add_argument("--output-dir", default="docs/benchmarks")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"logits alignment failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
