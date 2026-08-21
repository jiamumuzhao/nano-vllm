"""Benchmark-only W8A8 cuBLASLt INT8->INT32 path with preallocated storage."""
from __future__ import annotations

import torch

from . import w8a8_tensorcore as _w8a8_wmma

try:
    from . import _w8a8_cublaslt
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised on non-CUDA hosts
    _w8a8_cublaslt = None
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


PATH = "w8a8_experimental_cublaslt_int8_i32"
AUTOTUNE_BUDGETS = (0, 1 << 20, 4 << 20, 16 << 20, 64 << 20)


def availability(x=None):
    if _w8a8_cublaslt is None:
        return False, f"cuBLASLt W8A8 extension unavailable: {_IMPORT_ERROR}"
    if not torch.cuda.is_available():
        return False, "CUDA unavailable"
    device = x.device if x is not None and x.is_cuda else torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    if major * 10 + minor < 75:
        return False, f"SM{major}{minor} is below the cuBLASLt W8A8 SM75 requirement"
    return True, "cublaslt_api_verified"


class W8A8Workspace:
    """Persistent buffers and shape plans for one benchmark/model setup."""

    def __init__(self, max_m, max_n, max_k, device):
        self.max_m, self.max_n, self.max_k = int(max_m), int(max_n), int(max_k)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"W8A8 cuBLASLt workspace requires CUDA device, got {self.device}")
        self.xq = torch.empty((self.max_m, self.max_k), device=self.device, dtype=torch.int8)
        self.xs = torch.empty((self.max_m,), device=self.device, dtype=torch.float16)
        self.acc = torch.empty((self.max_m, self.max_n), device=self.device, dtype=torch.int32)
        self.out = torch.empty((self.max_m, self.max_n), device=self.device, dtype=torch.float16)
        self.cublas_workspace = torch.empty((1,), device=self.device, dtype=torch.uint8)
        self.packed_weight = None
        self.packed_weights = {}
        self.col32_layout = None
        self.col32_attempt = None
        self.packed_weight_source = None
        self.bound_weight_identity = None
        self.plans = {}
        self.candidates = {}
        self.layout_results = {}
        self.selected_plans = {}
        self.selected_metadata = {}
        self.algorithm = {}
        self.prepared = False

    def _check_shape(self, m, n, k):
        if m <= 0 or n <= 0 or k <= 0 or m > self.max_m or n > self.max_n or k > self.max_k:
            raise ValueError(f"W8A8 cuBLASLt shape {(m, n, k)} exceeds workspace capacity {(self.max_m, self.max_n, self.max_k)}")
        # The existing activation quantizer writes a contiguous K row. Keep
        # the hot path allocation-free and make this layout requirement clear.
        if k != self.max_k:
            raise ValueError(f"W8A8 workspace requires K == max_k ({self.max_k}) for allocation-free quantization, got {k}")

    def pack_weight(self, weight_int8):
        if weight_int8.device != self.device or weight_int8.dtype != torch.int8 or weight_int8.ndim != 2 or not weight_int8.is_contiguous():
            raise ValueError("weight packing requires contiguous CUDA int8 [N,K] on the workspace device")
        n, k = weight_int8.shape
        self._check_shape(1, n, k)
        packed = torch.empty((k, n), device=self.device, dtype=torch.int8)
        _w8a8_cublaslt.pack_weight_cuda(weight_int8, packed)
        self.packed_weight = packed
        self.packed_weights = {"row_major": packed}
        self.col32_layout = None
        self.col32_attempt = None
        self.bound_weight_identity = self._weight_identity(weight_int8)
        self.packed_weight_source = self.bound_weight_identity.copy()
        return packed

    @staticmethod
    def _weight_identity(weight_int8):
        try:
            version = weight_int8._version
        except Exception as exc:
            raise ValueError(
                "W8A8 cuBLASLt cannot read Tensor version for bound weight; "
                "refusing workspace reuse because in-place modification cannot be detected"
            ) from exc
        if version is None:
            raise ValueError(
                "W8A8 cuBLASLt Tensor version is unavailable; refusing workspace reuse "
                "because in-place modification cannot be detected"
            )
        return {
            "data_ptr": int(weight_int8.data_ptr()),
            "shape": tuple(int(v) for v in weight_int8.shape),
            "dtype": str(weight_int8.dtype),
            "device": str(weight_int8.device),
            "version": int(version),
        }

    def validate_weight(self, weight_int8):
        """Reject stale packed weights before any GPU work is launched."""
        if self.bound_weight_identity is None or self.packed_weight is None:
            raise ValueError("workspace has no bound packed weight; call workspace.prepare(new_weight, ...) first")
        current = self._weight_identity(weight_int8)
        if current != self.bound_weight_identity:
            raise ValueError(
                "W8A8 cuBLASLt packed weight mismatch: "
                f"workspace bound packed weight={self.bound_weight_identity}, "
                f"incoming weight={current}; caller must execute workspace.prepare(new_weight, ...) before running"
            )
        return True

    def prepare(self, weight_int8, shapes=None, max_workspace_bytes=64 * 1024 * 1024, autotune=True):
        """Pack weight and autotune cuBLASLt plans before timing."""
        self.pack_weight(weight_int8)
        n, k = weight_int8.shape
        # COL32 is a real setup-time experiment.  The extension derives the
        # padded physical buffer and invokes cublasLtMatrixTransform; no
        # hand-written COL32 address mapping is used here.
        self.col32_layout = _w8a8_cublaslt.col32_layout_info(k, n)
        self.col32_attempt = {
            "attempted": True,
            "stage": "layout-create",
            "status_code": None,
            "status": None,
            "success": False,
            "logical_shape": [k, n],
            "physical_shape": [self.col32_layout["physical_rows"], self.col32_layout["physical_cols"]],
            "leading_dimension": self.col32_layout["leading_dimension"],
            "padding_cols": self.col32_layout["padding_cols"],
            "buffer_bytes": self.col32_layout["bytes"],
            "layout": "CUBLASLT_ORDER_COL32",
        }
        try:
            col32_buffer = torch.empty((int(self.col32_layout["bytes"]),), device=self.device, dtype=torch.int8)
            result = _w8a8_cublaslt.transform_weight_col32(self.packed_weight, col32_buffer, k, n)
            self.col32_attempt.update(dict(result))
            if bool(result.get("success", False)):
                self.packed_weights["col32"] = col32_buffer
        except RuntimeError as exc:
            self.col32_attempt.update({"success": False, "stage": "layout-create", "status": str(exc), "reason": str(exc)})
        if "col32" not in self.packed_weights:
            self.col32_attempt.setdefault("reason", "COL32 setup failed")
        if shapes is None:
            shapes = [(self.max_m, n, k)]
        max_bytes = 1
        self.plans.clear()
        self.candidates.clear()
        self.layout_results.clear()
        self.selected_plans.clear()
        self.selected_metadata.clear()
        self.algorithm.clear()
        for m, shape_n, shape_k in shapes:
            m, shape_n, shape_k = int(m), int(shape_n), int(shape_k)
            self._check_shape(m, shape_n, shape_k)
            if shape_n != n or shape_k != k:
                raise ValueError("all prepared shapes must match the packed weight N,K")
            shape = (m, shape_n, shape_k)
            if not autotune:
                plan = _w8a8_cublaslt.create_plan(
                    m, shape_n, shape_k, self.max_k, shape_n, self.max_n, max_workspace_bytes
                )
                max_bytes = max(max_bytes, int(plan.workspace_size))
                self.plans[shape] = plan
                self.selected_plans[shape] = plan
                self.algorithm[shape] = plan.algorithm
                self.candidates[shape] = [{
                    "layout": "row_major", "status": "baseline",
                    "algorithm": plan.algorithm, "heuristic_index": int(plan.heuristic_index),
                    "returned_count": int(plan.returned_count), "waves_count": float(plan.waves_count),
                    "workspace_bytes": int(plan.workspace_size), "p50_ms": None,
                }]
                self.layout_results[shape] = {
                    "row_major": {"status": "baseline", "candidates": self.candidates[shape]},
                    "col32": dict(self.col32_attempt, status="passed" if "col32" in self.packed_weights else "failed", candidates=[]),
                }
                self.selected_metadata[shape] = {
                    "layout": "row_major", "algorithm": plan.algorithm,
                    "heuristic_index": int(plan.heuristic_index), "returned_count": int(plan.returned_count),
                    "waves_count": float(plan.waves_count), "workspace_bytes": int(plan.workspace_size),
                    "p50_ms": None, "budget_bytes": max_workspace_bytes,
                    "compute_type": "CUBLAS_COMPUTE_32I", "A_dtype": "CUDA_R_8I", "B_dtype": "CUDA_R_8I",
                    "C_dtype": "CUDA_R_32I", "D_dtype": "CUDA_R_32I",
                    "layout_attempted": True,
                    "packed_buffer_ptr": self.packed_weight.data_ptr(),
                    "packed_buffer_bytes": self.packed_weight.numel(),
                    "logical_shape": [k, n], "physical_padded_shape": [k, n],
                    "leading_dimension": n,
                    "transform_status": "row_major_baseline",
                    "heuristic_candidate_count": 1,
                    "algorithm_config_id": None,
                    "tile": None,
                    "split_k": None,
                    "reduction_scheme": None,
                    "stages": None,
                    "algorithm_metadata": "unavailable_from_current_api",
                }
                continue
            # Candidate timing uses only preallocated xq/acc and CUDA Events.
            calibration = torch.randn((m, shape_k), device=self.device, dtype=torch.float16)
            _w8a8_wmma._w8a8_tensorcore.quantize_cuda(calibration, self.xq[:m, :shape_k], self.xs[:m])
            del calibration
            candidates = []
            layout_records = {
                "row_major": {"status": "searching", "candidates": []},
                "col32": dict(self.col32_attempt, status="searching" if "col32" in self.packed_weights else "failed", candidates=[]),
            }
            for layout_name in ("row_major", "col32"):
                if layout_name == "col32" and "col32" not in self.packed_weights:
                    continue
                for budget in AUTOTUNE_BUDGETS:
                    create_candidates = (_w8a8_cublaslt.create_plan_candidates if layout_name == "row_major" else _w8a8_cublaslt.create_col32_plan_candidates)
                    layout_ld = shape_n if layout_name == "row_major" else int(self.col32_layout["leading_dimension"])
                    try:
                        plans = create_candidates(
                            m, shape_n, shape_k, self.max_k, layout_ld, self.max_n,
                            budget, 16,
                        )
                    except RuntimeError as exc:
                        layout_records[layout_name]["candidates"].append({"layout": layout_name, "budget_bytes": budget, "status": "skipped", "reason": str(exc)})
                        continue
                    for plan in plans:
                        max_bytes = max(max_bytes, int(plan.workspace_size))
                        candidate = {
                            "layout": layout_name,
                            "budget_bytes": budget,
                            "algorithm": plan.algorithm,
                            "heuristic_index": int(plan.heuristic_index),
                            "returned_count": int(plan.returned_count),
                            "waves_count": float(plan.waves_count),
                            "workspace_bytes": int(plan.workspace_size),
                            "status": "pending",
                            "plan": plan,
                            "algorithm_config_id": None,
                            "tile": None,
                            "split_k": None,
                            "reduction_scheme": None,
                            "stages": None,
                            "algorithm_metadata": "unavailable_from_current_api",
                        }
                        candidates.append(candidate)
            if not candidates:
                raise RuntimeError(
                    f"cuBLASLt W8A8 autotune found no valid INT8 candidate for shape={shape}, "
                    f"SM{torch.cuda.get_device_capability(self.device)}, layout=row_major, budgets={AUTOTUNE_BUDGETS}"
                )
            # Workspace is allocated before candidate timing and is shared by all candidates.
            if self.cublas_workspace.numel() < max_bytes:
                self.cublas_workspace = torch.empty((max_bytes,), device=self.device, dtype=torch.uint8)
            acc = self.acc[:m, :shape_n]
            timed_candidates = []
            for candidate in candidates:
                plan = candidate["plan"]
                try:
                    packed = self.packed_weights[candidate["layout"]]
                    for _ in range(2):
                        plan.run(self.xq[:m, :shape_k], packed, acc, self.cublas_workspace)
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(5):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        plan.run(self.xq[:m, :shape_k], packed, acc, self.cublas_workspace)
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end))
                    times.sort()
                    candidate["p50_ms"] = float(times[len(times) // 2])
                    candidate["status"] = "passed"
                    timed_candidates.append(candidate)
                except RuntimeError as exc:
                    candidate["status"] = "skipped"
                    candidate["reason"] = str(exc)
            if not timed_candidates:
                raise RuntimeError(
                    f"cuBLASLt W8A8 autotune candidates were not executable for shape={shape}, "
                    f"SM{torch.cuda.get_device_capability(self.device)}, layout=row_major, budgets={AUTOTUNE_BUDGETS}"
                )
            selected = min(timed_candidates, key=lambda item: item["p50_ms"])
            selected_plan = selected["plan"]
            self.candidates[shape] = [{k: v for k, v in item.items() if k != "plan"} for item in candidates]
            layout_records["row_major"]["status"] = "passed"
            layout_records["row_major"]["candidates"] = self.candidates[shape]
            for layout_name in ("row_major", "col32"):
                layout_records[layout_name]["candidates"] = [c for c in self.candidates[shape] if c["layout"] == layout_name]
                if layout_name == "col32" and layout_records[layout_name]["candidates"]:
                    layout_records[layout_name]["status"] = "passed"
                elif layout_name == "col32":
                    layout_records[layout_name]["status"] = "failed"
                    layout_records[layout_name]["transform_success"] = bool(self.col32_attempt.get("success", False))
                    layout_records[layout_name]["success"] = False
                    if layout_records[layout_name]["transform_success"]:
                        layout_records[layout_name]["stage"] = "heuristic"
                        layout_records[layout_name]["status_code"] = 0
                        layout_records[layout_name]["cuBLASLt_status"] = "CUBLAS_STATUS_SUCCESS"
                        layout_records[layout_name]["reason"] = "cublasLt heuristic returned zero valid COL32 candidates"
            self.layout_results[shape] = layout_records
            self.selected_plans[shape] = selected_plan
            self.plans[shape] = selected_plan
            self.algorithm[shape] = selected["algorithm"]
            self.selected_metadata[shape] = {
                k: selected[k] for k in (
                    "layout", "algorithm", "heuristic_index", "returned_count",
                    "waves_count", "workspace_bytes", "p50_ms", "budget_bytes",
                )
            } | {
                "compute_type": "CUBLAS_COMPUTE_32I",
                "A_dtype": "CUDA_R_8I", "B_dtype": "CUDA_R_8I",
                "C_dtype": "CUDA_R_32I", "D_dtype": "CUDA_R_32I",
                "layout_attempted": True,
                "packed_buffer_ptr": self.packed_weights[selected["layout"]].data_ptr(),
                "packed_buffer_bytes": self.packed_weights[selected["layout"]].numel(),
                "logical_shape": [k, n],
                "physical_padded_shape": [k, n] if selected["layout"] == "row_major" else [self.col32_layout["physical_rows"], self.col32_layout["physical_cols"]],
                "leading_dimension": n if selected["layout"] == "row_major" else self.col32_layout["leading_dimension"],
                "transform_status": "row_major" if selected["layout"] == "row_major" else self.col32_attempt.get("stage"),
                "heuristic_candidate_count": sum(1 for c in self.candidates[shape] if c["layout"] == selected["layout"]),
                "algorithm_config_id": None,
                "tile": None,
                "split_k": None,
                "reduction_scheme": None,
                "stages": None,
                "algorithm_metadata": "unavailable_from_current_api",
            }
        if self.cublas_workspace.numel() < max_bytes:
            self.cublas_workspace = torch.empty((max_bytes,), device=self.device, dtype=torch.uint8)
        self.prepared = True
        return self

    def data_ptrs(self):
        return {
            "xq": self.xq.data_ptr(), "xs": self.xs.data_ptr(),
            "acc": self.acc.data_ptr(), "out": self.out.data_ptr(),
            "cublas_workspace": self.cublas_workspace.data_ptr(),
            "packed_weight": self.packed_weight.data_ptr() if self.packed_weight is not None else None,
            "packed_weight_row_major": self.packed_weights.get("row_major").data_ptr() if "row_major" in self.packed_weights else None,
            "packed_weight_col32": self.packed_weights.get("col32").data_ptr() if "col32" in self.packed_weights else None,
        }

    def run(self, x, weight_scale, bias=None):
        if not self.prepared or self.packed_weight is None:
            raise RuntimeError("W8A8 cuBLASLt workspace must be prepared before run()")
        if x.device != self.device or weight_scale.device != self.device:
            raise ValueError("input, scale, and workspace devices must match")
        if x.dtype != torch.float16 or weight_scale.dtype != torch.float16:
            raise ValueError("cuBLASLt W8A8 dtype mismatch: requires FP16 activation and weight scale")
        if x.ndim < 2 or not x.is_contiguous() or not weight_scale.is_contiguous():
            raise ValueError("activation and weight scale must be contiguous; activation may be 2D or 3D")
        m, k = x.numel() // x.shape[-1], x.shape[-1]
        n = self.packed_weight.shape[1]
        self._check_shape(m, n, k)
        if weight_scale.numel() != n:
            raise ValueError(f"weight scale length {weight_scale.numel()} does not match N={n}")
        if bias is not None and (bias.device != self.device or bias.dtype != torch.float16 or not bias.is_contiguous() or bias.numel() != n):
            raise ValueError("bias must be contiguous CUDA FP16 with length N")
        shape = (m, n, k)
        plan = self.selected_plans.get(shape)
        if plan is None:
            raise ValueError(f"shape {(m, n, k)} was not prepared in the cuBLASLt workspace")
        x2 = x.view(m, k)
        xq = self.xq[:m, :k]
        xs = self.xs[:m]
        acc = self.acc[:m, :n]
        out = self.out[:m, :n]
        _w8a8_wmma._w8a8_tensorcore.quantize_cuda(x2, xq, xs)
        layout = self.selected_metadata[shape]["layout"]
        plan.run(xq, self.packed_weights[layout], acc, self.cublas_workspace)
        _w8a8_cublaslt.epilogue_cuda(acc, xs, weight_scale, bias, out, m, n, self.max_n, self.max_n)
        return out.view(*x.shape[:-1], n)


def linear(x, weight_int8, weight_scale, bias=None, workspace=None):
    allocation_fallback = workspace is None
    if workspace is None:
        workspace = W8A8Workspace(x.numel() // x.shape[-1], weight_int8.shape[0], x.shape[-1], x.device)
        workspace.prepare(weight_int8)
    else:
        workspace.validate_weight(weight_int8)
    out = workspace.run(x, weight_scale, bias)
    linear.last_path = PATH + ("_allocation_fallback" if allocation_fallback else "")
    return out


linear.last_path = "uninitialized"
