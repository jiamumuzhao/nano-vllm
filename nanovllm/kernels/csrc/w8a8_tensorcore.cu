#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace w8a8 {
using namespace nvcuda;

__global__ void quantize_rows(const half* x, int8_t* xq, half* xs, int M, int K) {
    int row = blockIdx.x;
    if (row >= M) return;
    __shared__ float reduce[256];
    float local = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        local = fmaxf(local, fabsf(__half2float(x[row*K+k])));
    reduce[threadIdx.x] = local;
    __syncthreads();
    for (int stride=128; stride; stride>>=1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x+stride]);
        __syncthreads();
    }
    float scale = reduce[0] / 127.f;
    if (scale == 0.f) scale = 1.f;
    if (threadIdx.x == 0) xs[row] = __float2half(scale);
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float q = nearbyintf(__half2float(x[row*K+k]) / scale);
        q = fminf(127.f, fmaxf(-127.f, q));
        xq[row*K+k] = static_cast<int8_t>(q);
    }
}

__global__ void int8_wmma_gemm(const int8_t* xq, const half* xs, const int8_t* w,
                               const half* ws, const half* bias, half* out,
                               int M, int N, int K) {
    int tm=blockIdx.y*16, tn=blockIdx.x*16;
    __shared__ int8_t a[16][16], b[16][16];
    __shared__ int e[16][16];
    wmma::fragment<wmma::accumulator,16,16,16,int> acc;
    wmma::fill_fragment(acc, 0);
    for (int k0=0;k0<K;k0+=16) {
        for (int i=threadIdx.x;i<256;i+=32) {
            int r=i/16,c=i%16,m=tm+r,k=k0+c;
            a[r][c]=(m<M&&k<K)?xq[m*K+k]:0;
        }
        // Weight is [N,K]. Pair adjacent threads on one output channel so
        // each pair reads consecutive K values, then transpose into B[r][n].
        int n_local=threadIdx.x/2, r0=(threadIdx.x&1)*8;
        #pragma unroll
        for (int j=0;j<8;++j) {
            int r=r0+j, n=tn+n_local, bk=k0+r;
            b[r][n_local]=(n<N&&bk<K)?w[n*K+bk]:0;
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::row_major> bf;
        wmma::load_matrix_sync(af,&a[0][0],16);
        wmma::load_matrix_sync(bf,&b[0][0],16);
        wmma::mma_sync(acc,af,bf,acc);
        __syncthreads();
    }
    wmma::store_matrix_sync(&e[0][0],acc,16,wmma::mem_row_major);
    __syncwarp();
    for (int i=threadIdx.x;i<256;i+=32) {
        int r=i/16,c=i%16,m=tm+r,n=tn+c;
        if (m<M&&n<N) {
            float v=(float)e[r][c]*__half2float(xs[m])*__half2float(ws[n]);
            if (bias) v+=__half2float(bias[n]);
            out[m*N+n]=__float2half(v);
        }
    }
}

// Benchmark-only 4-warp CTA candidate. Each CTA computes [16,64].
__global__ void int8_wmma_gemm_cta16(const int8_t* xq, const half* xs, const int8_t* w,
                                     const half* ws, const half* bias, half* out,
                                     int M, int N, int K) {
    int tm=blockIdx.y*16, tn=blockIdx.x*64;
    int warp_n=(threadIdx.x/32)%4;
    extern __shared__ int8_t smem[];
    int8_t* a=smem; int8_t* b=a+16*16; int* e=reinterpret_cast<int*>(b+16*64);
    wmma::fragment<wmma::accumulator,16,16,16,int> acc; wmma::fill_fragment(acc,0);
    for(int k0=0;k0<K;k0+=16) {
        for(int i=threadIdx.x;i<16*16;i+=128) {
            int r=i/16,c=i%16,m=tm+r,k=k0+c; a[i]=(m<M&&k<K)?xq[m*K+k]:0;
        }
        int n_local=threadIdx.x/2, r0=(threadIdx.x&1)*8;
        #pragma unroll
        for(int j=0;j<8;++j) {
            int r=r0+j, n=tn+n_local, bk=k0+r;
            b[r*64+n_local]=(n<N&&bk<K)?w[n*K+bk]:0;
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::row_major> bf;
        wmma::load_matrix_sync(af,a,16); wmma::load_matrix_sync(bf,b+warp_n*16,64);
        wmma::mma_sync(acc,af,bf,acc); __syncthreads();
    }
    int* ep=e+warp_n*16; wmma::store_matrix_sync(ep,acc,64,wmma::mem_row_major); __syncwarp();
    for(int i=threadIdx.x%32;i<256;i+=32) {
        int r=i/16,c=i%16,m=tm+r,n=tn+warp_n*16+c;
        if(m<M&&n<N) { float v=(float)ep[r*64+c]*__half2float(xs[m])*__half2float(ws[n]); if(bias) v+=__half2float(bias[n]); out[m*N+n]=__float2half(v); }
    }
}

// Benchmark-only 8-warp CTA candidate: each CTA computes [32,64].
__global__ void int8_wmma_gemm_cta32(const int8_t* xq, const half* xs, const int8_t* w,
                                   const half* ws, const half* bias, half* out,
                                   int M, int N, int K) {
    int tm=blockIdx.y*32, tn=blockIdx.x*64;
    int warp_id=threadIdx.x/32, warp_m=warp_id/4, warp_n=warp_id%4;
    extern __shared__ int8_t smem[];
    int8_t* a=smem; int8_t* b=a+32*16; int* e=reinterpret_cast<int*>(b+16*64);
    wmma::fragment<wmma::accumulator,16,16,16,int> acc; wmma::fill_fragment(acc,0);
    for(int k0=0;k0<K;k0+=16) {
        for(int i=threadIdx.x;i<32*16;i+=256) { int r=i/16,c=i%16,m=tm+r,k=k0+c; a[i]=(m<M&&k<K)?xq[m*K+k]:0; }
        int n_local=threadIdx.x/4, r0=(threadIdx.x&3)*4;
        #pragma unroll
        for(int j=0;j<4;++j) { int r=r0+j, n=tn+n_local, bk=k0+r; b[r*64+n_local]=(n<N&&bk<K)?w[n*K+bk]:0; }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::row_major> bf;
        wmma::load_matrix_sync(af,a+warp_m*16*16,16); wmma::load_matrix_sync(bf,b+warp_n*16,64); wmma::mma_sync(acc,af,bf,acc);
        __syncthreads();
    }
    int* ep=e+(warp_m*16)*64+warp_n*16; wmma::store_matrix_sync(ep,acc,64,wmma::mem_row_major); __syncwarp();
    for(int i=threadIdx.x%32;i<256;i+=32) { int r=i/16,c=i%16,m=tm+warp_m*16+r,n=tn+warp_n*16+c; if(m<M&&n<N) { float v=(float)ep[r*64+c]*__half2float(xs[m])*__half2float(ws[n]); if(bias) v+=__half2float(bias[n]); out[m*N+n]=__float2half(v); } }
}

void check_cuda(torch::Tensor x, torch::Tensor wi, torch::Tensor ws) {
    TORCH_CHECK(x.is_cuda()&&wi.is_cuda()&&ws.is_cuda(),"W8A8 requires CUDA tensors");
    TORCH_CHECK(x.scalar_type()==at::kChar&&wi.scalar_type()==at::kChar&&ws.scalar_type()==at::kHalf,"expected int8/int8/FP16 GEMM inputs");
    TORCH_CHECK(x.dim()==2&&wi.dim()==2&&ws.dim()==1,"expected 2D x, [N,K] weight and [N] scale");
    TORCH_CHECK(x.is_contiguous()&&wi.is_contiguous()&&ws.is_contiguous(),"inputs must be contiguous");
    const auto* p=at::cuda::getCurrentDeviceProperties(); TORCH_CHECK(p->major*10+p->minor>=75,"W8A8 INT8 MMA requires SM75+");
    TORCH_CHECK(x.size(1)==wi.size(1)&&wi.size(0)==ws.size(0),"shape mismatch");
}

void quantize_cuda(torch::Tensor x, torch::Tensor xq, torch::Tensor xs) {
    TORCH_CHECK(x.is_cuda()&&x.scalar_type()==at::kHalf&&x.dim()==2,"expected CUDA FP16 2D activation");
    TORCH_CHECK(xq.is_cuda()&&xq.scalar_type()==at::kChar&&xs.is_cuda()&&xs.scalar_type()==at::kHalf,"invalid quant buffers");
    auto stream=at::cuda::getCurrentCUDAStream();
    quantize_rows<<<x.size(0),256,0,stream>>>(reinterpret_cast<const half*>(x.data_ptr<at::Half>()),xq.data_ptr<int8_t>(),reinterpret_cast<half*>(xs.data_ptr<at::Half>()),x.size(0),x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm_cuda(torch::Tensor xq, torch::Tensor xs, torch::Tensor wi, torch::Tensor ws,
               c10::optional<torch::Tensor> bias, torch::Tensor out) {
    check_cuda(xq,wi,ws);
    TORCH_CHECK(xs.is_cuda()&&xs.scalar_type()==at::kHalf&&xs.dim()==1,"invalid activation scales");
    if (bias.has_value()) TORCH_CHECK(bias->scalar_type()==at::kHalf&&bias->is_contiguous(),"bias must be FP16 contiguous");
    auto stream=at::cuda::getCurrentCUDAStream();
    dim3 grid((wi.size(0)+15)/16,(xq.size(0)+15)/16);
    int8_wmma_gemm<<<grid,32,0,stream>>>(xq.data_ptr<int8_t>(),reinterpret_cast<const half*>(xs.data_ptr<at::Half>()),wi.data_ptr<int8_t>(),reinterpret_cast<const half*>(ws.data_ptr<at::Half>()),bias.has_value()?reinterpret_cast<const half*>(bias->data_ptr<at::Half>()):nullptr,reinterpret_cast<half*>(out.data_ptr<at::Half>()),xq.size(0),wi.size(0),xq.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm_cuda_cta16(torch::Tensor xq, torch::Tensor xs, torch::Tensor wi, torch::Tensor ws,
                     c10::optional<torch::Tensor> bias, torch::Tensor out) {
    check_cuda(xq,wi,ws);
    TORCH_CHECK(xs.is_cuda()&&xs.scalar_type()==at::kHalf&&xs.dim()==1,"invalid activation scales");
    if (bias.has_value()) TORCH_CHECK(bias->scalar_type()==at::kHalf&&bias->is_contiguous(),"bias must be FP16 contiguous");
    auto stream=at::cuda::getCurrentCUDAStream(); dim3 grid((wi.size(0)+63)/64,(xq.size(0)+15)/16);
    constexpr size_t smem=16*16*sizeof(int8_t)+16*64*sizeof(int8_t)+16*64*sizeof(int);
    int8_wmma_gemm_cta16<<<grid,128,smem,stream>>>(xq.data_ptr<int8_t>(),reinterpret_cast<const half*>(xs.data_ptr<at::Half>()),wi.data_ptr<int8_t>(),reinterpret_cast<const half*>(ws.data_ptr<at::Half>()),bias.has_value()?reinterpret_cast<const half*>(bias->data_ptr<at::Half>()):nullptr,reinterpret_cast<half*>(out.data_ptr<at::Half>()),xq.size(0),wi.size(0),xq.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm_cuda_cta32(torch::Tensor xq, torch::Tensor xs, torch::Tensor wi, torch::Tensor ws,
                   c10::optional<torch::Tensor> bias, torch::Tensor out) {
    check_cuda(xq,wi,ws);
    TORCH_CHECK(xs.is_cuda()&&xs.scalar_type()==at::kHalf&&xs.dim()==1,"invalid activation scales");
    if (bias.has_value()) TORCH_CHECK(bias->scalar_type()==at::kHalf&&bias->is_contiguous(),"bias must be FP16 contiguous");
    auto stream=at::cuda::getCurrentCUDAStream(); dim3 grid((wi.size(0)+63)/64,(xq.size(0)+31)/32);
    constexpr size_t smem=32*16*sizeof(int8_t)+16*64*sizeof(int8_t)+32*64*sizeof(int);
    int8_wmma_gemm_cta32<<<grid,256,smem,stream>>>(xq.data_ptr<int8_t>(),reinterpret_cast<const half*>(xs.data_ptr<at::Half>()),wi.data_ptr<int8_t>(),reinterpret_cast<const half*>(ws.data_ptr<at::Half>()),bias.has_value()?reinterpret_cast<const half*>(bias->data_ptr<at::Half>()):nullptr,reinterpret_cast<half*>(out.data_ptr<at::Half>()),xq.size(0),wi.size(0),xq.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm_cuda_cta(torch::Tensor xq, torch::Tensor xs, torch::Tensor wi, torch::Tensor ws,
                   c10::optional<torch::Tensor> bias, torch::Tensor out) {
    gemm_cuda_cta32(xq,xs,wi,ws,bias,out);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("quantize_cuda",&w8a8::quantize_cuda,"W8A8 per-row activation quantization");
    m.def("gemm_cuda",&w8a8::gemm_cuda,"W8A8 INT8 WMMA GEMM");
    m.def("gemm_cuda_cta16",&w8a8::gemm_cuda_cta16,"W8A8 INT8 CTA16x64 WMMA candidate");
    m.def("gemm_cuda_cta32",&w8a8::gemm_cuda_cta32,"W8A8 INT8 CTA32x64 WMMA candidate");
    m.def("gemm_cuda_cta",&w8a8::gemm_cuda_cta,"W8A8 INT8 CTA32x64 compatibility candidate");
}
