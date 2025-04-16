import cv2
import numpy as np
import onnxruntime as ort

# Load the YOLO model
MODEL_PATH = "/home/sunrise/Documents/HelloBalls-Host/scripts/yolov11_roboflow_ir9.onnx" # Ensure this path is correct
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.2
CLASS_NAMES = ["Tennis Ball", "Tennis ball", "Tennis racket", "Tennis-Ball", "person", "tennis-ball"]  # Update this if the model has more classes

def preprocess_image(image):
    """
    Preprocess the image for YOLO model input.
    """
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
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
    
    for i in range(outputs.shape[1]):
        detection = outputs[0, i, :]
        
        # First 4 values are box coordinates, remaining are class scores
        box = detection[0:4]
        scores = detection[4:]  # All values after the box coordinates are class scores
        
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Ensure class_id is valid
        if class_id >= len(CLASS_NAMES):
            continue
        
        if confidence > CONFIDENCE_THRESHOLD:
            # Convert box coordinates to pixel values
            # YOLOv8/v11 outputs center_x, center_y, width, height directly in normalized form
            center_x, center_y, width, height = box
            
            # Scale normalized coordinates (0-1) to image dimensions
            center_x *= img_width
            center_y *= img_height
            width *= img_width
            height *= img_height
            
            # Calculate top-left corner from center coordinates
            x = int(center_x - width/2)
            y = int(center_y - height/2)
            
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression (NMS)
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        final_boxes = []
        for i in indices:
            if isinstance(i, list) or isinstance(i, np.ndarray):
                i = i[0]  # For older OpenCV versions
            final_boxes.append(boxes[i])
        return final_boxes
    
    return []

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

    # print(f"Outputs shape: {outputs[0].shape}")
    # print(f"Sample output: {outputs[0][0]}")

    # Postprocess the outputs
    coordinates = postprocess(outputs[0], image.shape)
    return coordinates

def cv_test_onnx():
    try:
        # Open the camera (0 is usually the default camera)
        cap = cv2.VideoCapture(0)

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
            coordinates = process_image(frame)

            # Draw bounding boxes on the frame
            for (x, y, w, h) in coordinates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tennis Ball", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

if __name__ == "__main__":
    cv_test_onnx()