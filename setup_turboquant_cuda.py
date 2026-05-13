from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="turboquant_cuda",
    ext_modules=[
        CUDAExtension(
            name="turboquant_cuda",
            sources=[
                "turboquant/csrc/turboquant_score.cpp",
                "turboquant/csrc/turboquant_score_cuda.cu",
                "turboquant/csrc/turboquant_pack_cuda.cu",
                "turboquant/csrc/turboquant_mse_cuda.cu",
                "turboquant/csrc/turboquant_score_transposed_cuda.cu",
                "turboquant/csrc/turboquant_score_mse_lut_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
