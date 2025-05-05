import cv2
import sys
import os

# 添加父目录到路径以导入我们的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.ball_detector import BallDetector

def main():
    # 初始化球体检测器 - 需要指定模型路径
    model_path = "../../../models/yolo11_model.bin"  # 根据实际模型位置调整
    detector = BallDetector(model_path)  # 注意这里需要传入模型路径参数

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
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
        detections = detector.detect_balls(frame)  # 确保方法名与ball_detector.py中一致

        # 在检测到球时打印坐标
        for bbox in detections:
            x, y, w, h = bbox
            center_x = x + w/2
            center_y = y + h/2
            print(f"Ball detected at: ({center_x}, {center_y})")
            
            # 在图像上标记球体
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        # 显示结果
        cv2.imshow('Ball Detection', frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    detector.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()