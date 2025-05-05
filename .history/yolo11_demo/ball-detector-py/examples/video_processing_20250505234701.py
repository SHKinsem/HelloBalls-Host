import cv2
from python.ball_detector import BallDetector

def process_video(video_path):
    # Initialize the ball detector
    detector = BallDetector()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Perform ball detection
        detections = detector.detect(frame)

        # Visualize the results
        for bbox in detections:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Video Processing', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file_path = "path/to/your/video.mp4"  # Replace with your video file path
    process_video(video_file_path)