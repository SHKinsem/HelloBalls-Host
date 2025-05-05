import cv2
import numpy as np
import os
import sys
from pathlib import Path

# 尝试导入 C++ 绑定模块
try:
    # 添加 cpp_api 目录到路径，以便能够导入编译好的模块
    sys.path.append(str(Path(__file__).parent.parent / 'cpp_api'))
    from ball_detector_cpp import load_model, detect_balls, cleanup_model
    _has_cpp_module = True
except ImportError:
    print("Warning: C++ module not found, using dummy implementation")
    _has_cpp_module = False

class BallDetector:
    """球体检测器类，封装了C++实现的YOLO检测功能"""
    
    def __init__(self, model_path=None):
        """
        初始化球体检测器
        
        Args:
            model_path (str): 模型文件路径，如果为None则使用默认路径
        """
        self.model_loaded = False
        
        # 如果未提供模型路径，尝试寻找默认位置
        if model_path is None:
            # 尝试几个可能的位置
            possible_paths = [
                Path(__file__).parent.parent.parent.parent / "models" / "yolo11_model.bin",
                Path(__file__).parent.parent.parent / "models" / "yolo11_model.bin",
                Path("/opt/models/yolo11_model.bin")
            ]
            
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break
        
        if not model_path or not Path(model_path).exists():
            print(f"Warning: Model file not found at {model_path}")
            return
            
        # 加载模型
        if _has_cpp_module:
            try:
                self.model_handle = load_model(model_path)
                self.model_loaded = True
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
    def detect(self, frame):
        """
        从图像帧中检测球体
        
        Args:
            frame: OpenCV图像帧
        
        Returns:
            list: 检测到的球体坐标，每个元素为 (x, y, w, h) 边界框
        """
        if not self.model_loaded or not _has_cpp_module:
            # 模拟检测结果（测试用）
            return []
            
        # 确保帧是正确的格式
        if frame.ndim != 3:
            print("Error: Input frame must be a 3-dimensional array")
            return []
            
        # 调用C++检测函数
        try:
            ball_centers = detect_balls(frame)
            
            # 将中心点坐标转换为边界框
            # 假设每个球有一个固定大小的边界框，可以根据实际情况调整
            bboxes = []
            for cx, cy in ball_centers:
                box_size = 50  # 假设球的大小为50x50像素
                x = max(0, int(cx - box_size/2))
                y = max(0, int(cy - box_size/2))
                bboxes.append((x, y, box_size, box_size))
                
            return bboxes
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return []
    
    def release(self):
        """释放资源"""
        if self.model_loaded and _has_cpp_module:
            try:
                cleanup_model()
                print("Model resources released")
            except:
                pass