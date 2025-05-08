import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Base directory of the setup.py file
setup_dir = os.path.dirname(__file__)

# Paths relative to this setup.py file
cpp_root_dir = os.path.abspath(os.path.join(setup_dir, "..", "cpp"))
cpp_src_dir = os.path.join(cpp_root_dir, "src")
cpp_bindings_file = os.path.join(cpp_root_dir, "bindings", "yolo11_api_bindings.cpp")
cpp_include_dir = os.path.join(cpp_root_dir, "include")

# Horizon Robotics specific paths (adjust if your RDK environment differs)
horizon_dnn_include_dir = "/usr/include/dnn"
horizon_dnn_lib_dir = "/usr/lib"

# Python include directory - needed for Python.h headers
python_include_dir = sys.prefix + '/include/python' + sys.version[:4]

# OpenCV libraries - ensure OpenCV development headers and libraries are installed.
opencv_libraries = [
    "opencv_core",
    "opencv_imgproc",
    "opencv_dnn",       # For NMSBoxes function
    "opencv_imgcodecs",
    "opencv_highgui",   # Add other OpenCV modules that might be needed
]

ext_modules = [
    Pybind11Extension(
        "yolo11_api",  # Output module name: yolo11_api.so
        sources=[
            cpp_bindings_file,
            os.path.join(cpp_src_dir, "model.cc"),
            os.path.join(cpp_src_dir, "pid_controller.cc"),
            os.path.join(cpp_src_dir, "postprocess.cc"),
            os.path.join(cpp_src_dir, "preprocess.cc"),
            os.path.join(cpp_src_dir, "utils.cc"),
        ],
        include_dirs=[
            cpp_include_dir,
            horizon_dnn_include_dir,
            python_include_dir,
            "/usr/include/opencv4"  # Standard OpenCV include path
        ],
        library_dirs=[
            horizon_dnn_lib_dir,
            "/usr/lib/aarch64-linux-gnu",  # Standard library path for aarch64
        ],
        libraries=[
            "dnn",  # For Horizon DNN library
            *opencv_libraries,
            "python" + sys.version[:4],  # Link against Python library
        ],
        extra_compile_args=["-std=c++11", "-O3", "-fPIC"],
        extra_link_args=["-Wl,-rpath,$ORIGIN"],  # To find shared libs in the same dir
        language="c++",
    ),
]

# Read README for long description
readme_path = os.path.abspath(os.path.join(setup_dir, "..", "..", "README.md"))
try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Python bindings for YOLOv11 inference and PID controller."

setup(
    name="yolo11_api_wrapper", # Name of the pip package
    version="0.1.0",
    author="Sunrise", # Placeholder, please change
    author_email="user@example.com", # Placeholder, please change
    description="Python bindings for C++ YOLOv11 inference and PID controller for Horizon RDK.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False, # C extensions are not zip safe
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.6", # Build-time and runtime (for stubs if generated)
        "numpy",         # For image data transfer
        # "opencv-python", # Add if your Python code using this lib needs it for image loading/display
    ],
    classifiers=[ # Optional metadata
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: Apache Software License", # Assuming Apache 2.0 from C++ files
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
