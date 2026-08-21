#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAException.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace w8a8_cublaslt {

inline void check_status(cublasStatus_t status, const char* what, int M, int N, int K) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream msg;
        msg << what << " failed for M=" << M << ", N=" << N << ", K=" << K
            << ", dtype=int8->int32, status=" << static_cast<int>(status);
        TORCH_CHECK(false, msg.str());
    }
}

const char* status_text(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

__global__ void pack_weight_kernel(const int8_t* src, int8_t* dst, int N, int K) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(N) * K;
    if (i >= total) return;
    int n = static_cast<int>(i / K);
    int k = static_cast<int>(i % K);
    dst[static_cast<int64_t>(k) * N + n] = src[i];
}

__global__ void epilogue_kernel(const int32_t* acc, const half* xs, const half* ws,
                                const half* bias, half* out, int M, int N,
                                int acc_stride, int out_stride) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(M) * N;
    if (i >= total) return;
    int m = static_cast<int>(i / N);
    int n = static_cast<int>(i % N);
    float value = static_cast<float>(acc[static_cast<int64_t>(m) * acc_stride + n])
        * __half2float(xs[m]) * __half2float(ws[n]);
    if (bias != nullptr) value += __half2float(bias[n]);
    out[static_cast<int64_t>(m) * out_stride + n] = __float2half(value);
}

pybind11::dict col32_layout_info(int K, int N) {
    TORCH_CHECK(K > 0 && N > 0, "COL32 layout requires positive K,N");
    int groups = (N + 31) / 32;
    int64_t ld = static_cast<int64_t>(K) * 32;
    int64_t elements = ld * groups;
    pybind11::dict result;
    result["logical_rows"] = K;
    result["logical_cols"] = N;
    result["physical_rows"] = K;
    result["physical_cols"] = groups * 32;
    result["leading_dimension"] = ld;
    result["padding_cols"] = groups * 32 - N;
    result["bytes"] = elements * sizeof(int8_t);
    result["order"] = "CUBLASLT_ORDER_COL32";
    return result;
}

pybind11::dict transform_weight_col32(torch::Tensor src, torch::Tensor dst, int K, int N) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.scalar_type() == at::kChar &&
                dst.scalar_type() == at::kChar && src.is_contiguous() && dst.is_contiguous(),
                "COL32 transform requires contiguous CUDA int8 tensors");
    auto info = col32_layout_info(K, N);
    TORCH_CHECK(src.numel() == static_cast<int64_t>(K) * N, "COL32 source shape mismatch");
    TORCH_CHECK(dst.numel() >= info["bytes"].cast<int64_t>(), "COL32 destination buffer is too small");
    pybind11::dict result;
    result["attempted"] = true;
    result["stage"] = "layout-create";
    result["status_code"] = 0;
    result["status"] = status_text(CUBLAS_STATUS_SUCCESS);
    cublasLtHandle_t handle = nullptr;
    cublasLtMatrixLayout_t src_desc = nullptr, dst_desc = nullptr;
    cublasLtMatrixTransformDesc_t transform = nullptr;
    auto fail = [&](cublasStatus_t s, const char* stage) {
        result["stage"] = stage;
        result["status_code"] = static_cast<int>(s);
        result["status"] = status_text(s);
        result["success"] = false;
    };
    cublasStatus_t s = cublasLtCreate(&handle);
    if (s != CUBLAS_STATUS_SUCCESS) { fail(s, "layout-create"); goto cleanup; }
    s = cublasLtMatrixLayoutCreate(&src_desc, CUDA_R_8I, K, N, N);
    if (s != CUBLAS_STATUS_SUCCESS) { fail(s, "layout-create"); goto cleanup; }
    s = cublasLtMatrixLayoutCreate(&dst_desc, CUDA_R_8I, K, N, static_cast<int>(info["leading_dimension"].cast<int64_t>()));
    if (s != CUBLAS_STATUS_SUCCESS) { fail(s, "layout-create"); goto cleanup; }
    {
        cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
        s = cublasLtMatrixLayoutSetAttribute(src_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
        if (s != CUBLAS_STATUS_SUCCESS) { fail(s, "layout-create"); goto cleanup; }
        cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
        s = cublasLtMatrixLayoutSetAttribute(dst_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32));
        if (s != CUBLAS_STATUS_SUCCESS) { fail(s, "layout-create"); goto cleanup; }
    }
    s = cublasLtMatrixTransformDescCreate(&transform, CUDA_R_32I);
    if (s != CUBLAS_STATUS_SUCCESS) { fail(s, "transform"); goto cleanup; }
    {
        cublasOperation_t op_n = CUBLAS_OP_N;
        s = cublasLtMatrixTransformDescSetAttribute(transform, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_n, sizeof(op_n));
        if (s != CUBLAS_STATUS_SUCCESS) { fail(s, "transform"); goto cleanup; }
    }
    {
        int32_t alpha = 1, beta = 0;
        auto stream = at::cuda::getCurrentCUDAStream();
        s = cublasLtMatrixTransform(handle, transform, &alpha, src.data_ptr<int8_t>(), src_desc,
                                     &beta, nullptr, nullptr, dst.data_ptr<int8_t>(), dst_desc, stream.stream());
        if (s != CUBLAS_STATUS_SUCCESS) { fail(s, "transform"); goto cleanup; }
    }
    result["success"] = true;
    result["stage"] = "transform";
cleanup:
    if (transform) cublasLtMatrixTransformDescDestroy(transform);
    if (dst_desc) cublasLtMatrixLayoutDestroy(dst_desc);
    if (src_desc) cublasLtMatrixLayoutDestroy(src_desc);
    if (handle) cublasLtDestroy(handle);
    return result;
}

void pack_weight_cuda(torch::Tensor weight, torch::Tensor packed) {
    TORCH_CHECK(weight.is_cuda() && packed.is_cuda(), "cuBLASLt W8A8 packing requires CUDA tensors");
    TORCH_CHECK(weight.scalar_type() == at::kChar && packed.scalar_type() == at::kChar,
                "cuBLASLt W8A8 packing requires int8 tensors");
    TORCH_CHECK(weight.dim() == 2 && packed.dim() == 2 && weight.is_contiguous() && packed.is_contiguous(),
                "weight and packed weight must be contiguous 2D tensors");
    TORCH_CHECK(packed.size(0) == weight.size(1) && packed.size(1) == weight.size(0),
                "packed weight must have shape [K,N] for input weight [N,K]");
    int64_t total = weight.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    pack_weight_kernel<<<(total + 255) / 256, 256, 0, stream.stream()>>>(
        weight.data_ptr<int8_t>(), packed.data_ptr<int8_t>(), weight.size(0), weight.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void epilogue_cuda(torch::Tensor acc, torch::Tensor xs, torch::Tensor ws,
                   c10::optional<torch::Tensor> bias, torch::Tensor out,
                   int M, int N, int acc_stride, int out_stride) {
    TORCH_CHECK(acc.is_cuda() && xs.is_cuda() && ws.is_cuda() && out.is_cuda(),
                "cuBLASLt W8A8 epilogue requires CUDA tensors");
    TORCH_CHECK(acc.scalar_type() == at::kInt && xs.scalar_type() == at::kHalf &&
                ws.scalar_type() == at::kHalf && out.scalar_type() == at::kHalf,
                "cuBLASLt W8A8 epilogue expects int32/FP16 tensors");
    const half* bias_ptr = nullptr;
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kHalf && bias->is_contiguous(),
                    "cuBLASLt W8A8 bias must be contiguous CUDA FP16");
        bias_ptr = reinterpret_cast<const half*>(bias->data_ptr<at::Half>());
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t total = static_cast<int64_t>(M) * N;
    epilogue_kernel<<<(total + 255) / 256, 256, 0, stream.stream()>>>(
        acc.data_ptr<int32_t>(), reinterpret_cast<const half*>(xs.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(ws.data_ptr<at::Half>()), bias_ptr,
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), M, N, acc_stride, out_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

struct CublasLtPlan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0;
    int M, N, K, lda, ldb, ldc;
    int heuristic_index = 0;
    int returned_count = 0;
    float waves_count = 0.0f;
    bool valid = false;

    CublasLtPlan(int m, int n, int k, int lda_, int ldb_, int ldc_, size_t max_workspace,
                 int requested_index = 0, int requested_count = 1, cublasLtOrder_t b_order_ = CUBLASLT_ORDER_ROW)
        : M(m), N(n), K(k), lda(lda_), ldb(ldb_), ldc(ldc_), heuristic_index(requested_index) {
        auto fail = [&](cublasStatus_t s, const char* what) { check_status(s, what, M, N, K); };
        fail(cublasLtCreate(&handle), "cublasLtCreate");
        fail(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I), "cublasLtMatmulDescCreate");
        cublasOperation_t op_n = CUBLAS_OP_N;
        fail(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)), "set TRANSA");
        fail(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)), "set TRANSB");
        fail(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8I, M, K, lda), "create A layout");
        fail(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8I, K, N, ldb), "create B layout");
        fail(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32I, M, N, ldc), "create C layout");
        fail(cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32I, M, N, ldc), "create D layout");
        cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
        fail(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "set A row order");
        fail(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order_, sizeof(b_order_)), "set B order");
        fail(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "set C row order");
        fail(cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)), "set D row order");
        cublasLtMatmulPreference_t preference = nullptr;
        fail(cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate");
        fail(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &max_workspace, sizeof(max_workspace)), "set max workspace");
        std::vector<cublasLtMatmulHeuristicResult_t> results(requested_count);
        fail(cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc, c_desc, d_desc,
                                            preference, requested_count, results.data(), &returned_count), "cublasLtMatmulAlgoGetHeuristic");
        cublasLtMatmulPreferenceDestroy(preference);
        if (requested_index < returned_count && results[requested_index].state == CUBLAS_STATUS_SUCCESS) {
            valid = true;
            algo = results[requested_index].algo;
            workspace_size = results[requested_index].workspaceSize;
            waves_count = results[requested_index].wavesCount;
        }
    }

    ~CublasLtPlan() {
        if (d_desc) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
        if (handle) cublasLtDestroy(handle);
    }

    void run(torch::Tensor xq, torch::Tensor packed, torch::Tensor acc, torch::Tensor workspace) {
        TORCH_CHECK(valid, "invalid cuBLASLt heuristic candidate for M=", M, ", N=", N, ", K=", K);
        TORCH_CHECK(xq.is_cuda() && packed.is_cuda() && acc.is_cuda() && workspace.is_cuda(),
                    "cuBLASLt W8A8 plan run requires CUDA tensors");
        TORCH_CHECK(xq.scalar_type() == at::kChar && packed.scalar_type() == at::kChar && acc.scalar_type() == at::kInt,
                    "cuBLASLt W8A8 plan expects int8 A/B and int32 C");
        TORCH_CHECK(xq.is_contiguous() && packed.is_contiguous(),
                    "cuBLASLt W8A8 plan requires contiguous xq and packed weight views");
        TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(workspace_size),
                    "cuBLASLt workspace is too small: required ", workspace_size, " bytes");
        int32_t alpha = 1, beta = 0;
        auto stream = at::cuda::getCurrentCUDAStream();
        auto status = cublasLtMatmul(handle, op_desc, &alpha, xq.data_ptr<int8_t>(), a_desc,
                                     packed.data_ptr<int8_t>(), b_desc, &beta, acc.data_ptr<int32_t>(), c_desc,
                                     acc.data_ptr<int32_t>(), d_desc, &algo, workspace.data_ptr(),
                                     workspace_size, stream.stream());
        check_status(status, "cublasLtMatmul INT8xINT8->INT32", M, N, K);
    }
};

std::shared_ptr<CublasLtPlan> create_plan(int M, int N, int K, int lda, int ldb, int ldc, int64_t max_workspace) {
    TORCH_CHECK(M > 0 && N > 0 && K > 0 && lda >= K && ldb >= N && ldc >= N,
                "invalid cuBLASLt plan shape/leading dimensions");
    return std::make_shared<CublasLtPlan>(M, N, K, lda, ldb, ldc, static_cast<size_t>(max_workspace));
}

pybind11::list create_plan_candidates(int M, int N, int K, int lda, int ldb, int ldc,
                                       int64_t max_workspace, int max_candidates) {
    TORCH_CHECK(max_candidates > 0 && max_candidates <= 64, "max_candidates must be in 1..64");
    pybind11::list plans;
    for (int i = 0; i < max_candidates; ++i) {
        auto plan = std::make_shared<CublasLtPlan>(M, N, K, lda, ldb, ldc,
                                                    static_cast<size_t>(max_workspace), i, max_candidates);
        if (plan->valid) plans.append(plan);
        else break;
    }
    return plans;
}

pybind11::list create_col32_plan_candidates(int M, int N, int K, int lda, int ldb, int ldc,
                                             int64_t max_workspace, int max_candidates) {
    TORCH_CHECK(max_candidates > 0 && max_candidates <= 64, "max_candidates must be in 1..64");
    pybind11::list plans;
    for (int i = 0; i < max_candidates; ++i) {
        auto plan = std::make_shared<CublasLtPlan>(M, N, K, lda, ldb, ldc,
                                                    static_cast<size_t>(max_workspace), i, max_candidates,
                                                    CUBLASLT_ORDER_COL32);
        if (plan->valid) plans.append(plan);
        else break;
    }
    return plans;
}

} // namespace w8a8_cublaslt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    py::class_<w8a8_cublaslt::CublasLtPlan, std::shared_ptr<w8a8_cublaslt::CublasLtPlan>>(m, "CublasLtPlan")
        .def("run", &w8a8_cublaslt::CublasLtPlan::run)
        .def_property_readonly("valid", [](const w8a8_cublaslt::CublasLtPlan& p) { return p.valid; })
        .def_property_readonly("heuristic_index", [](const w8a8_cublaslt::CublasLtPlan& p) { return p.heuristic_index; })
        .def_property_readonly("returned_count", [](const w8a8_cublaslt::CublasLtPlan& p) { return p.returned_count; })
        .def_property_readonly("waves_count", [](const w8a8_cublaslt::CublasLtPlan& p) { return p.waves_count; })
        .def_property_readonly("workspace_size", [](const w8a8_cublaslt::CublasLtPlan& p) { return p.workspace_size; })
        .def_property_readonly("algorithm", [](const w8a8_cublaslt::CublasLtPlan& p) {
            return std::string("heuristic_index_") + std::to_string(p.heuristic_index);
        });
    m.def("pack_weight_cuda", &w8a8_cublaslt::pack_weight_cuda, "Pack [N,K] int8 weight to contiguous [K,N]");
    m.def("col32_layout_info", &w8a8_cublaslt::col32_layout_info);
    m.def("transform_weight_col32", &w8a8_cublaslt::transform_weight_col32);
    m.def("epilogue_cuda", &w8a8_cublaslt::epilogue_cuda, "INT32 to FP16 scaled epilogue");
    m.def("create_plan", &w8a8_cublaslt::create_plan, py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("lda"), py::arg("ldb"), py::arg("ldc"), py::arg("max_workspace") = 64 * 1024 * 1024);
    m.def("create_plan_candidates", &w8a8_cublaslt::create_plan_candidates,
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("lda"), py::arg("ldb"), py::arg("ldc"),
          py::arg("max_workspace") = 64 * 1024 * 1024, py::arg("max_candidates") = 16);
    m.def("create_col32_plan_candidates", &w8a8_cublaslt::create_col32_plan_candidates,
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("lda"), py::arg("ldb"), py::arg("ldc"),
          py::arg("max_workspace") = 64 * 1024 * 1024, py::arg("max_candidates") = 16);
}
