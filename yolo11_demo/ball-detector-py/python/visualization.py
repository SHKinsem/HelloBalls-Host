# File: /ball-detector-py/ball-detector-py/python/visualization.py

import cv2

def draw_detections(frame, bboxes, scores, indices):
    for i in indices:
        bbox = bboxes[i]
        score = scores[i]
        if score > 0.5:  # Confidence threshold
            x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def display_frame(frame):
    cv2.imshow("Detection Results", frame)
    cv2.waitKey(1)  # Display the frame for 1 ms

def cleanup_visualization():
    cv2.destroyAllWindows()