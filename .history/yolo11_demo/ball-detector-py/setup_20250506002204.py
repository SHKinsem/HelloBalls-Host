from setuptools import setup, find_packages
import os
import subprocess
from pathlib import Path

# 构建C++扩展
def build_cpp_extension():
    try:
        subprocess.check_call(['python', 'cpp_api/setup.py', 'build_ext', '--inplace'], 
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        print("C++ extension built successfully")
        return True
    except Exception as e:
        print(f"Failed to build C++ extension: {e}")
        return False

# 尝试构建C++扩展
build_cpp_extension()

setup(
    name="ball-detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
    ],
    author="Your Name",
    description="A Python package for ball detection using YOLO11",
)