from setuptools import setup, Extension
import pybind11
import os
from pathlib import Path

# 获取当前目录的路径
current_dir = Path(__file__).parent.absolute()
cpp_dir = current_dir.parent.parent / "cpp"

# 编译器标志
extra_compile_args = []
if os.name == 'nt':  # Windows
    extra_compile_args = ['/std:c++14']
else:  # Linux/Mac
    extra_compile_args = ['-std=c++14']

# 包含目录
include_dirs = [
    pybind11.get_include(),
    str(cpp_dir),
    str(cpp_dir / "include"),
    "/usr/local/include/opencv4"  # 根据系统调整OpenCV路径
]

# 库目录
library_dirs = []
if os.name == 'nt':  # Windows
    library_dirs.append("C:/opencv/build/x64/vc15/lib")
else:  # Linux/Mac
    library_dirs.append("/usr/local/lib")

# 需要链接的库
libraries = []
if os.name == 'nt':  # Windows
    libraries.append("opencv_world460")
else:  # Linux/Mac
    libraries.extend([
        "opencv_core", 
        "opencv_imgproc", 
        "opencv_highgui",
        "hbrt_bernoulli_aarch64"  # 根据RDK板需要添加
    ])

# 源文件列表
sources = ['bindings.cpp']

# 定义扩展模块
ext_modules = [
    Extension(
        'ball_detector_cpp',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name='ball_detector_cpp',
    version='0.1',
    description='Python bindings for YOLO11 ball detection',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    python_requires='>=3.6',
)