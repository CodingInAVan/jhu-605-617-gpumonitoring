import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

CUDA_PATH = os.environ.get("CUDA_PATH", "/usr/local/cuda")

if sys.platform == "win32":
    INCLUDE_DIRS = [
        "../gpumon_client/include", # Path to your C++ headers
        f"{CUDA_PATH}/include"
    ]
    LIBRARIES = ["cudart"]
    LIBRARY_DIRS = [f"{CUDA_PATH}/lib/x64"]
    EXTRA_COMPILE_ARGS = ["/std:c++17", "/O2"]
else:
    INCLUDE_DIRS = [
        "../gpumon_client/include",
        f"{CUDA_PATH}/include"
    ]
    LIBRARIES = ["cudart"]
    LIBRARY_DIRS = [f"{CUDA_PATH}/lib64"]
    EXTRA_COMPILE_ARGS = ["-std=c++17", "-O3"]

ext_modules = [
    Extension(
        "gpumon_py",
        ["src/bindings.cpp"],
        include_dirs=INCLUDE_DIRS,
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
    ),
]

setup(
    name="gpumon",
    version="0.1.0",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.5.0"],
)