def open_camera(camera_id=0):
    """Open the camera with the specified camera ID."""
    import cv2

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise Exception(f"Error: Unable to open camera with ID {camera_id}")
    
    return cap

def capture_frame(cap):
    """Capture a single frame from the camera."""
    ret, frame = cap.read()
    if not ret:
        raise Exception("Error: Could not read frame from camera")
    
    return frame

def release_camera(cap):
    """Release the camera resource."""
    cap.release()