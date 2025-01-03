from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "vccs_supervoxel",
        ["python_interface.cpp"],  # 源代码文件
        include_dirs=["codelibrary","include"],  # 指定头文件目录
        #extra_compile_args=["-std=c++11"],  # 使用 C++17 标准
    ),
]

setup(
    name="vccs_supervoxel",
    version="0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="Pybind11 wrapper for VCCS supervoxel segmentation",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)