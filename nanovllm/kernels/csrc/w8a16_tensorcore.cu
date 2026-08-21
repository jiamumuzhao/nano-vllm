#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace w8a16 {
using namespace nvcuda;

// The legacy kernel is retained only for the tiling microbenchmark.
__global__ void legacy_wmma(const half* x, const int8_t* w, const half* scale,
                            const half* bias, half* out, int M, int N, int K) {
    int tm = blockIdx.y * 16, tn = blockIdx.x * 16;
    __shared__ half a[16][16], b[16][16];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k0=0; k0<K; k0+=16) {
        for (int i=threadIdx.x; i<256; i+=32) {
            int r=i/16,c=i%16,m=tm+r,k=k0+c,n=tn+c,bk=k0+r;
            a[r][c]=(m<M&&k<K)?x[m*K+k]:__float2half(0.f);
            b[r][c]=(n<N&&bk<K)?__float2half((float)w[n*K+bk]*__half2float(scale[n])):__float2half(0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
        wmma::load_matrix_sync(af,&a[0][0],16); wmma::load_matrix_sync(bf,&b[0][0],16);
        wmma::mma_sync(acc,af,bf,acc); __syncthreads();
    }
    __shared__ float e[16][16];
    wmma::store_matrix_sync(&e[0][0],acc,16,wmma::mem_row_major); __syncthreads();
    for (int i=threadIdx.x;i<256;i+=32) { int r=i/16,c=i%16,m=tm+r,n=tn+c;
        if(m<M&&n<N) { float v=e[r][c]+(bias?__half2float(bias[n]):0.f); out[m*N+n]=__float2half(v); } }
}

// Benchmark-only wide decode candidate. Four warps share one [16,16] A tile
// and one [16,64] B tile. The only CTA barriers are load->WMMA and
// WMMA->next-load; each warp owns a private epilogue region afterward.
__global__ void decode_wide_wmma(const half* x, const int8_t* w, const half* scale,
                                 const half* bias, half* out, int M, int N, int K) {
    const int tm = blockIdx.y * 16, tn = blockIdx.x * 64;
    const int warp_n = threadIdx.x / 32;
    extern __shared__ unsigned char smem[];
    half* a = reinterpret_cast<half*>(smem);
    half* b = a + 16 * 16;
    float* ep = reinterpret_cast<float*>(b + 16 * 64) + warp_n * 16 * 16;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k0=0; k0<K; k0+=16) {
        for (int i=threadIdx.x;i<16*16;i+=128) {
            int r=i/16,c=i%16,m=tm+r,k=k0+c;
            a[i]=(m<M&&k<K)?x[m*K+k]:__float2half(0.f);
        }
        for (int i=threadIdx.x;i<16*64;i+=128) {
            int r=i/64,c=i%64,n=tn+c,bk=k0+r;
            b[i]=(n<N&&bk<K)?__float2half((float)w[n*K+bk]*__half2float(scale[n])):__float2half(0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
        wmma::load_matrix_sync(af,a,16);
        wmma::load_matrix_sync(bf,b+warp_n*16,64);
        wmma::mma_sync(acc,af,bf,acc);
        __syncthreads();
    }
    wmma::store_matrix_sync(ep,acc,16,wmma::mem_row_major);
    __syncwarp();
    for (int i=threadIdx.x%32;i<256;i+=32) {
        int r=i/16,c=i%16,m=tm+r,n=tn+warp_n*16+c;
        if(m<M&&n<N) {
            float v=ep[i]+(bias?__half2float(bias[n]):0.f);
            out[m*N+n]=__float2half(v);
        }
    }
}

// One CTA computes CTA_M x CTA_N. Each warp owns one 16x16 output tile.
// Shared A is reused by all warp_n values and shared B by all warp_m values.
template<int CTA_M, int CTA_N>
__global__ void tiled_wmma(const half* x, const int8_t* w, const half* scale,
                           const half* bias, half* out, int M, int N, int K) {
    constexpr int WARPS_M = CTA_M / 16;
    constexpr int WARPS_N = CTA_N / 16;
    const int warp_id = threadIdx.x / 32;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int tm = blockIdx.y * CTA_M, tn = blockIdx.x * CTA_N;
    extern __shared__ unsigned char smem[];
    half* a = reinterpret_cast<half*>(smem);
    half* b = a + CTA_M * 16;
    float* ep = reinterpret_cast<float*>(b + 16 * CTA_N);
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k0=0; k0<K; k0+=16) {
        const int total_a = CTA_M*16, total_b = 16*CTA_N;
        for (int i=threadIdx.x;i<total_a;i+=blockDim.x) {
            int r=i/16,c=i%16,m=tm+r,k=k0+c;
            a[i]=(m<M&&k<K)?x[m*K+k]:__float2half(0.f);
        }
        for (int i=threadIdx.x;i<total_b;i+=blockDim.x) {
            int r=i/CTA_N,c=i%CTA_N,n=tn+c,bk=k0+r;
            b[i]=(n<N&&bk<K)?__float2half((float)w[n*K+bk]*__half2float(scale[n])):__float2half(0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
        wmma::load_matrix_sync(af, a + warp_m*16*16, 16);
        wmma::load_matrix_sync(bf, b + warp_n*16, CTA_N);
        wmma::mma_sync(acc, af, bf, acc);
        __syncthreads();
    }
    ep += (warp_m*16)*CTA_N + warp_n*16;
    wmma::store_matrix_sync(ep, acc, CTA_N, wmma::mem_row_major);
    __syncthreads();
    const int total_out = CTA_M*CTA_N;
    for (int i=threadIdx.x;i<total_out;i+=blockDim.x) {
        int r=i/CTA_N,c=i%CTA_N,m=tm+r,n=tn+c;
        if (m<M&&n<N) {
            float v = reinterpret_cast<float*>(b + 16*CTA_N)[i];
            if (bias) v += __half2float(bias[n]);
            out[m*N+n]=__float2half(v);
        }
    }
}

template<int CTA_M, int CTA_N>
void launch_tiled(torch::Tensor x, torch::Tensor wi, torch::Tensor sc,
                  c10::optional<torch::Tensor> bias, torch::Tensor out,
                  cudaStream_t stream) {
    const int M=x.size(0), K=x.size(1), N=wi.size(0);
    const int warps=(CTA_M/16)*(CTA_N/16);
    const size_t smem=CTA_M*16*sizeof(half)+16*CTA_N*sizeof(half)+CTA_M*CTA_N*sizeof(float);
    dim3 grid((N+CTA_N-1)/CTA_N,(M+CTA_M-1)/CTA_M);
    tiled_wmma<CTA_M,CTA_N><<<grid,warps*32,smem,stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),wi.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(sc.data_ptr<at::Half>()),
        bias.has_value()?reinterpret_cast<const half*>(bias->data_ptr<at::Half>()):nullptr,
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),M,N,K);
}

void linear_cuda_legacy(torch::Tensor x, torch::Tensor wi, torch::Tensor sc,
                        c10::optional<torch::Tensor> bias, torch::Tensor out);

void check_inputs(torch::Tensor x, torch::Tensor wi, torch::Tensor sc,
                  c10::optional<torch::Tensor> bias, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda()&&wi.is_cuda()&&sc.is_cuda(),"all tensors must be CUDA");
    TORCH_CHECK(x.scalar_type()==at::kHalf&&wi.scalar_type()==at::kChar&&sc.scalar_type()==at::kHalf,"expected FP16/int8/FP16");
    TORCH_CHECK(x.dim()==2&&wi.dim()==2&&sc.dim()==1,"expected 2D activation and [N,K]/[N]");
    TORCH_CHECK(x.is_contiguous()&&wi.is_contiguous()&&sc.is_contiguous()&&out.is_contiguous(),"tensors must be contiguous");
    if(bias.has_value()) TORCH_CHECK(bias->scalar_type()==at::kHalf&&bias->is_contiguous(),"bias must be contiguous FP16");
    const auto* p=at::cuda::getCurrentDeviceProperties(); TORCH_CHECK(p->major*10+p->minor>=75,"WMMA requires SM75+");
    TORCH_CHECK(wi.size(1)==x.size(1)&&sc.size(0)==wi.size(0),"shape mismatch");
}

// Production selector: decode always uses legacy; only large prefill uses CTA.
// The M=64 case is intentionally legacy because the RTX 2080 Ti evidence
// showed the CTA variant slower there. This is a conservative first policy.
int linear_cuda(torch::Tensor x, torch::Tensor wi, torch::Tensor sc,
                c10::optional<torch::Tensor> bias, torch::Tensor out) {
    check_inputs(x,wi,sc,bias,out);
    auto stream=at::cuda::getCurrentCUDAStream();
    if(x.size(0)<=16) {
        linear_cuda_legacy(x,wi,sc,bias,out);
        return 1;
    }
    if(x.size(0)>=256 && wi.size(0)>=1024 && x.size(1)>=1024) {
        launch_tiled<64,64>(x,wi,sc,bias,out,stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); return 3;
    }
    linear_cuda_legacy(x,wi,sc,bias,out);
    return 2;
}

// Benchmark-only CTA candidate. Production routing must use linear_cuda().
void linear_cuda_cta(torch::Tensor x, torch::Tensor wi, torch::Tensor sc,
                     c10::optional<torch::Tensor> bias, torch::Tensor out) {
    check_inputs(x,wi,sc,bias,out);
    auto stream=at::cuda::getCurrentCUDAStream();
    if(x.size(0)<=16) launch_tiled<16,64>(x,wi,sc,bias,out,stream);
    else launch_tiled<64,64>(x,wi,sc,bias,out,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void linear_cuda_decode_wide_candidate(torch::Tensor x, torch::Tensor wi,
                                       torch::Tensor sc, c10::optional<torch::Tensor> bias,
                                       torch::Tensor out) {
    check_inputs(x,wi,sc,bias,out);
    TORCH_CHECK(x.size(0)<=16, "decode-wide candidate requires M<=16");
    const size_t smem=16*16*sizeof(half)+16*64*sizeof(half)+4*16*16*sizeof(float);
    dim3 grid((wi.size(0)+63)/64,(x.size(0)+15)/16);
    auto stream=at::cuda::getCurrentCUDAStream();
    decode_wide_wmma<<<grid,128,smem,stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),wi.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(sc.data_ptr<at::Half>()),
        bias.has_value()?reinterpret_cast<const half*>(bias->data_ptr<at::Half>()):nullptr,
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),x.size(0),wi.size(0),x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void linear_cuda_legacy(torch::Tensor x, torch::Tensor wi, torch::Tensor sc,
                        c10::optional<torch::Tensor> bias, torch::Tensor out) {
    check_inputs(x,wi,sc,bias,out);
    dim3 grid((wi.size(0)+15)/16,(x.size(0)+15)/16);
    auto stream=at::cuda::getCurrentCUDAStream();
    legacy_wmma<<<grid,32,0,stream>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),wi.data_ptr<int8_t>(),reinterpret_cast<const half*>(sc.data_ptr<at::Half>()),bias.has_value()?reinterpret_cast<const half*>(bias->data_ptr<at::Half>()):nullptr,reinterpret_cast<half*>(out.data_ptr<at::Half>()),x.size(0),wi.size(0),x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("linear_cuda",&w8a16::linear_cuda,"W8A16 tiled WMMA linear");
    m.def("linear_cuda_cta",&w8a16::linear_cuda_cta,"W8A16 CTA WMMA benchmark candidate");
    m.def("linear_cuda_decode_wide_candidate",&w8a16::linear_cuda_decode_wide_candidate,"W8A16 wide decode WMMA benchmark candidate");
    m.def("linear_cuda_legacy",&w8a16::linear_cuda_legacy,"W8A16 legacy single-warp WMMA linear");
}
