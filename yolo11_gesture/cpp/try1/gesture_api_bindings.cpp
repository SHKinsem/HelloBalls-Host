#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

// Include necessary C++ headers from the project
#include "config.h"
#include "model.h"
#include "preprocess.h"
#include "postprocess.h"
#include "pid_controller.h"
#include "utils.h"

namespace py = pybind11;

// --- Global Model Variables ---
// Encapsulating these in a class would be a safer, more robust approach,
// but for this binding, we'll use statics with an initialization guard.
static hbPackedDNNHandle_t g_packed_dnn_handle;
static hbDNNHandle_t g_dnn_handle;
static int g_input_H, g_input_W;
static int g_tensor_order[1] = {0}; // Default order
static bool g_model_initialized = false;
static hbDNNTensor g_input_tensor;
static hbDNNTensor* g_output_tensors = nullptr;
static int32_t g_output_count = 0;

// --- Helper Function to Initialize Model (call once) ---
bool ensure_model_initialized() {
    if (g_model_initialized) {
        return true;
    }

    if (loadModel(g_packed_dnn_handle, g_dnn_handle, g_input_H, g_input_W, g_tensor_order) != 0) {
        py::print("[gesture_api] Critical: Failed to load model.");
        return false;
    }

    if (hbDNNGetOutputCount(&g_output_count, g_dnn_handle) != 0) {
        py::print("[gesture_api] Critical: Failed to get model output count.");
        hbDNNRelease(g_packed_dnn_handle); // Clean up partially loaded model
        return false;
    }

    if (prepareTensors(g_input_tensor, g_output_tensors, g_dnn_handle, g_output_count, g_input_H, g_input_W) != 0) {
        py::print("[gesture_api] Critical: Failed to prepare model tensors.");
        hbDNNRelease(g_packed_dnn_handle); // Clean up
        return false;
    }

    g_model_initialized = true;
    py::print("[gesture_api] Model initialized successfully.");
    return true;
}

// --- Python-friendly Output Structure ---
struct PyModelOutput {
    std::vector<std::tuple<double, double, double, double>> bboxes; // (x, y, width, height)
    std::vector<float> scores;
    std::vector<int> class_ids;
};

// --- Inference Function ---
PyModelOutput perform_inference(py::array_t<unsigned char> input_image_np) {
    if (!ensure_model_initialized()) {
        throw std::runtime_error("Model is not initialized or failed to initialize.");
    }

    py::buffer_info buf = input_image_np.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Input image must be a 3-dimensional NumPy array (H, W, C) with 3 channels (e.g., BGR or RGB).");
    }
    if (buf.format != py::format_descriptor<unsigned char>::format()) {
        throw std::runtime_error("Input image NumPy array must be of type uint8.");
    }

    // Create cv::Mat from NumPy array data (no copy)
    cv::Mat frame(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    if (frame.empty()) {
        throw std::runtime_error("Received an empty image frame.");
    }

    // --- Preprocessing ---
    cv::Mat img_nv12; // This will be filled by preprocess()
    float x_scale = 1.0f, y_scale = 1.0f;
    int x_shift = 0, y_shift = 0;
    preprocess(frame, img_nv12, g_input_H, g_input_W, x_scale, y_scale, x_shift, y_shift);
    
    // --- Copy to Input Tensor ---
    // Ensure g_input_tensor is valid and prepared
    copyToInputTensor(img_nv12, g_input_tensor); // This is void, not returning a status code
    
    // --- Run Inference ---
    hbDNNTaskHandle_t task_handle = nullptr;
    if (runInference(g_dnn_handle, g_input_tensor, g_output_tensors, task_handle) != 0) {
        hbDNNReleaseTask(task_handle); // Ensure task is released even on failure
        throw std::runtime_error("DNN inference execution failed.");
    }

    // --- Postprocessing ---
    std::vector<std::vector<cv::Rect2d>> all_bboxes_cv;
    std::vector<std::vector<float>> all_scores_cv;
    postprocess(g_output_tensors, g_tensor_order, g_input_H, g_input_W, all_bboxes_cv, all_scores_cv);
    
    // --- NMS ---
    std::vector<std::vector<int>> all_indices_cv;
    applyNMS(all_bboxes_cv, all_scores_cv, all_indices_cv);

    hbDNNReleaseTask(task_handle); // Release task after use

    // --- Format Output ---
    PyModelOutput result;
    for (size_t class_id = 0; class_id < all_indices_cv.size(); ++class_id) {
        for (size_t i = 0; i < all_indices_cv[class_id].size(); ++i) {
            int idx = all_indices_cv[class_id][i];
            if (class_id < all_bboxes_cv.size() && static_cast<size_t>(idx) < all_bboxes_cv[class_id].size() &&
                class_id < all_scores_cv.size() && static_cast<size_t>(idx) < all_scores_cv[class_id].size()) {
                
                const cv::Rect2d& bbox_cv = all_bboxes_cv[class_id][idx];
                
                // Scale bounding box coordinates back to original image dimensions
                double original_x = (bbox_cv.x - x_shift) / x_scale;
                double original_y = (bbox_cv.y - y_shift) / y_scale;
                double original_w = bbox_cv.width / x_scale;
                double original_h = bbox_cv.height / y_scale;

                result.bboxes.emplace_back(original_x, original_y, original_w, original_h);
                result.scores.push_back(all_scores_cv[class_id][idx]);
                result.class_ids.push_back(static_cast<int>(class_id));
            }
        }
    }
    return result;
}

// --- Cleanup Function ---
void cleanup_model_resources() {
    if (g_model_initialized) {
        cleanupModel(g_packed_dnn_handle, g_input_tensor, g_output_tensors, g_output_count);
        g_model_initialized = false;
        // Reset global pointers/handles for safety, though dylibs might not fully unload state easily
        g_output_tensors = nullptr; 
        py::print("[gesture_api] Model resources released.");
    }
}

// --- PYBIND11 Module Definition ---
PYBIND11_MODULE(gesture_api, m) {
    m.doc() = "Python bindings for YOLOv11 inference and PID controller";

    // Initialize the model when the module is imported.
    // Alternatively, provide an explicit init function.
    // ensure_model_initialized(); // Consider if auto-init is desired or manual init function.
    m.def("initialize_model", &ensure_model_initialized, "Initializes the YOLO model and resources. Call this before inference if not auto-initialized.");


    // --- PyModelOutput Structure Binding ---
    py::class_<PyModelOutput>(m, "ModelOutput")
        .def(py::init<>())
        .def_readwrite("bboxes", &PyModelOutput::bboxes, "List of (x, y, width, height) tuples for bounding boxes")
        .def_readwrite("scores", &PyModelOutput::scores, "List of confidence scores")
        .def_readwrite("class_ids", &PyModelOutput::class_ids, "List of class IDs");

    // --- Inference Function Binding ---
    m.def("inference", &perform_inference, 
          py::arg("image"), 
          "Performs inference on an input image (BGR NumPy uint8 array). "
          "Returns an ModelOutput object with bboxes, scores, and class_ids.");

    // --- PIDController Class Binding ---
    py::class_<PIDController>(m, "PIDController")
        .def(py::init<float, float, float>(), 
             py::arg("kp"), py::arg("ki"), py::arg("kd"),
             "Initializes the PID controller with Kp, Ki, Kd parameters.")
        .def("set_params", [](PIDController& pid, float kp, float ki, float kd) {
            // Since there's no direct setParams method, use the individual setters
            pid.setKp(kp);
            pid.setKi(ki);
            pid.setKd(kd);
        }, py::arg("kp"), py::arg("ki"), py::arg("kd"), 
           "Sets new Kp, Ki, Kd parameters for the PID controller.")
        .def("calculate", &PIDController::calculate, 
             py::arg("error"), 
             "Calculates the PID output based on the given error.")
        .def("reset", &PIDController::reset, 
             "Resets the internal state of the PID controller (integral sum, previous error).");

    // --- Model Cleanup Function Binding ---
    m.def("cleanup_model", &cleanup_model_resources, 
          "Releases all loaded model resources. Call this when done or before application exit.");
    
    // Optional: Register cleanup_model_resources to be called on Python interpreter exit
    // py::module_::import("atexit").attr("register")(py::cpp_function(cleanup_model_resources));
}

