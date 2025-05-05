#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../../cpp/include/model.h"
#include "../../cpp/include/preprocess.h"
#include "../../cpp/include/postprocess.h"

namespace py = pybind11;

// Global variables to hold model state
static void* g_packed_dnn_handle = nullptr;
static void** g_input = nullptr;
static void** g_output = nullptr;
static int g_output_count = 0;

// Function to load the model
bool load_model() {
    const char* model_path = MODEL_PATH;
    return loadModel(model_path, &g_packed_dnn_handle, &g_input, &g_output, &g_output_count);
}

// Function to detect balls and return their coordinates
py::tuple detect_balls(py::array_t<uint8_t> image) {
    // Convert numpy array to cv::Mat
    py::buffer_info buf = image.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.shape[2];
    
    cv::Mat frame(height, width, CV_8UC3, (unsigned char*)buf.ptr);
    
    // Process frame (similar to what's done in main.cc)
    cv::Mat mat_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
    bgr2Nv12(frame, mat_nv12.data, width, height);
    
    // Pre-process and run model
    std::vector<std::vector<cv::Rect2d>> bboxes;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<int>> classIds;
    
    // Run model and post-processing
    runModel(g_packed_dnn_handle, g_input, g_output, mat_nv12.data);
    postProcess(g_output, width, height, bboxes, scores, classIds);
    
    // Create lists to hold ball coordinates
    py::list x_coords, y_coords, confidences;
    
    // Look for balls (sports_ball is typically class 32 in COCO)
    const int ball_class_id = 32;
    for (size_t i = 0; i < classIds.size(); ++i) {
        for (size_t j = 0; j < classIds[i].size(); ++j) {
            if (classIds[i][j] == ball_class_id) {
                cv::Rect2d bbox = bboxes[i][j];
                float conf = scores[i][j];
                
                // Calculate center point
                float x_center = bbox.x + bbox.width/2;
                float y_center = bbox.y + bbox.height/2;
                
                x_coords.append(x_center);
                y_coords.append(y_center);
                confidences.append(conf);
            }
        }
    }
    
    return py::make_tuple(x_coords, y_coords, confidences);
}

// Function to clean up resources
void cleanup_model() {
    if (g_packed_dnn_handle) {
        cleanupModel(g_packed_dnn_handle, g_input, g_output, g_output_count);
        g_packed_dnn_handle = nullptr;
        g_input = nullptr;
        g_output = nullptr;
    }
}

// Create the Python module
PYBIND11_MODULE(ball_detector_cpp, m) {
    m.doc() = "Python bindings for ball detection using YOLOv11";
    
    m.def("load_model", &load_model, "Load the YOLO ball detection model");
    m.def("detect_balls", &detect_balls, "Detect balls in an image and return their coordinates",
          py::arg("image"));
    m.def("cleanup_model", &cleanup_model, "Clean up resources");
}