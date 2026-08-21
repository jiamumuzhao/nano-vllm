from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    ext_modules=[
        CUDAExtension(
            name="nanovllm.kernels._w8a16_tensorcore",
            sources=["nanovllm/kernels/csrc/w8a16_tensorcore.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-gencode=arch=compute_75,code=sm_75"]},
        ),
        CUDAExtension(
            name="nanovllm.kernels._w8a8_tensorcore",
            sources=["nanovllm/kernels/csrc/w8a8_tensorcore.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-gencode=arch=compute_75,code=sm_75"]},
        ),
        CUDAExtension(
            name="nanovllm.kernels._w8a8_cublaslt",
            sources=["nanovllm/kernels/csrc/w8a8_cublaslt.cu"],
            libraries=["cublasLt", "cublas"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-gencode=arch=compute_75,code=sm_75"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
