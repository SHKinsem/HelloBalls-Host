import cv2
import os
import numpy as np
import onnxruntime as ort

# Load the YOLO model
MODEL_PATH = os.path.join(os.getcwd(), "scripts/yolov11_roboflow_ir9.onnx")
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.6  # Restored to original value after debugging
NMS_THRESHOLD = 0.6  # Restored to original value after debugging
CLASS_NAMES = ["Tennis Ball", "Tennis ball", "Tennis racket", "Tennis-Ball", "person", "tennis-ball"]

def preprocess_image(image):
    """
    Preprocess the image for YOLO model input.
    """
    # Maintain aspect ratio while resizing to the input dimensions
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = INPUT_HEIGHT, int(w * (INPUT_HEIGHT / h))
    else:
        new_h, new_w = int(h * (INPUT_WIDTH / w)), INPUT_WIDTH
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a canvas of the target size
    canvas = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    
    # Paste the resized image in the center
    offset_x = (INPUT_WIDTH - new_w) // 2
    offset_y = (INPUT_HEIGHT - new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    # Convert to blob with proper normalization
    blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), 
                                swapRB=True, crop=False, mean=[0, 0, 0])
    return blob

def postprocess(outputs, image_shape):
    """
    Postprocess the outputs to extract bounding boxes and coordinates.
    YOLOv8/v11 uses a transposed output format (batch, features, detections)
    """
    boxes = []
    confidences = []
    class_ids = []
    
    # Get original image dimensions for scaling
    img_height, img_width = image_shape[:2]
    
    # Transpose the output to the format we can process
    # From (1, 10, 8400) to (1, 8400, 10)
    outputs = outputs.transpose((0, 2, 1))
    
    # For multiple detections that are close, use a set to track locations we've seen
    seen_locations = set()
    
    for i in range(outputs.shape[1]):
        detection = outputs[0, i, :]
        
        # First 4 values are box coordinates, remaining are class scores
        box = detection[0:4]
        scores = detection[4:]  # All values after the box coordinates are class scores
        
        # Explicitly apply sigmoid to the class scores
        scores = 1 / (1 + np.exp(-scores))
        
        # Check if any value in scores is significantly different from others
        # This helps filter out uniform predictions
        score_std = np.std(scores)
        if score_std < 0.001:  # If all scores are very similar, skip this detection
            continue
            
        class_id = np.argmax(scores)
        confidence = float(scores[class_id])
        
        # Check if this is a valid class
        if class_id >= len(CLASS_NAMES):
            continue
        
        if confidence > CONFIDENCE_THRESHOLD:
            # Convert box coordinates to pixel values
            # YOLOv8/v11 outputs center_x, center_y, width, height directly in normalized form
            center_x, center_y, width, height = box
            
            # Skip if any of the box values are invalid
            if any(np.isnan([center_x, center_y, width, height])):
                continue
                
            # Skip boxes with zero width or height
            if width <= 0 or height <= 0:
                continue
            
            # Scale coordinates properly - make sure they're in range [0,1]
            # Normalize the values if they're outside the range [0,1]
            if center_x > 1.0 or center_y > 1.0 or width > 1.0 or height > 1.0:
                center_x = center_x / INPUT_WIDTH
                center_y = center_y / INPUT_HEIGHT
                width = width / INPUT_WIDTH
                height = height / INPUT_HEIGHT
            
            # Scale normalized coordinates (0-1) to image dimensions
            center_x *= img_width
            center_y *= img_height
            width *= img_width
            height *= img_height
            
            # Calculate top-left corner from center coordinates
            x = int(center_x - width/2)
            y = int(center_y - height/2)
            
            # Ensure the coordinates are within the image bounds
            x = max(0, min(x, img_width-1))
            y = max(0, min(y, img_height-1))
            width = max(1, min(width, img_width-x))
            height = max(1, min(height, img_height-y))
            
            # Create a location key to avoid duplicate detections
            loc_key = f"{int(center_x//10)},{int(center_y//10)}"
            if loc_key in seen_locations:
                continue
            seen_locations.add(loc_key)
            
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression (NMS)
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        final_boxes = []
        final_class_ids = []
        
        for i in indices:
            if isinstance(i, list) or isinstance(i, np.ndarray):
                i = i[0]  # For older OpenCV versions
            final_boxes.append(boxes[i])
            final_class_ids.append(class_ids[i])
        
        return final_boxes, final_class_ids
    
    return [], []

def process_image(image):
    """
    Process the image to detect tennis balls and return their coordinates.
    """
    # Preprocess the image
    blob = preprocess_image(image)

    # Run inference
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    
    # Pass the raw outputs to the postprocessing function
    coordinates, class_ids = postprocess(outputs[0], image.shape)
    return coordinates, class_ids

def cv_test_onnx():
    try:
        # Open the camera (0 is usually the default camera)
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            return

        print("Press 'q' to quit the program.")

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                break

            # Process the frame
            coordinates, class_ids = process_image(frame)

            # Draw bounding boxes on the frame
            for i, (x, y, w, h) in enumerate(coordinates):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Get the class name based on the class ID
                class_name = CLASS_NAMES[class_ids[i]] if i < len(class_ids) else "Unknown"
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Tennis Ball Detection", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

def test_on_static_image(image_path=None):
    """
    Test the detection on a static image instead of webcam feed.
    If no image path is provided, it will use the example.jpg in the scripts folder.
    """
    if image_path is None:
        image_path = os.path.join(os.getcwd(), "scripts/example.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
        
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
        
    # Process the image
    print(f"Processing image: {image_path}")
    coordinates, class_ids = process_image(image)
    
    # Draw bounding boxes on the image
    for i, (x, y, w, h) in enumerate(coordinates):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get the class name based on the class ID
        class_name = CLASS_NAMES[class_ids[i]] if i < len(class_ids) else "Unknown"
        cv2.putText(image, f"{class_name}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the image with detections
    cv2.imshow("Tennis Ball Detection (Static)", image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()
    
    print(f"Detected {len(coordinates)} objects.")
    return coordinates, class_ids

if __name__ == "__main__":
    # Test on a static image first
    test_on_static_image()
    
    # Then use webcam for detection
    cv_test_onnx()