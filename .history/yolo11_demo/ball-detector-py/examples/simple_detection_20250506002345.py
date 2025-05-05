import cv2
import sys
import os
from pathlib import Path

# 添加父目录到路径以导入模块
sys.path.append(str(Path(__file__).parent.parent))
from python.ball_detector import BallDetector

def main():
    # 初始化球体检测器，可以指定模型路径
    model_path = "../../models/yolo11_model.bin"  # 根据实际路径调整
    detector = BallDetector(model_path)

    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 根据实际设备调整摄像头索引

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    print("Press 'q' to quit")
    
    while True:
        # 捕获帧
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break

        # 执行球体检测
        detections = detector.detect(frame)
        
        # 在图像上显示检测结果
        for x, y, w, h in detections:
            # 计算中心点
            center_x = x + w/2
            center_y = y + h/2
            
            # 绘制边界框
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # 显示坐标
            cv2.putText(frame, f"({int(center_x)}, {int(center_y)})", 
                       (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Ball Detection', frame)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    detector.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()