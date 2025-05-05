#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 包含原始C++代码的头文件
#include "../../cpp/include/model.h"
#include "../../cpp/include/utils.h"
#include "../../cpp/include/preprocess.h"
#include "../../cpp/include/postprocess.h"

namespace py = pybind11;

// 全局变量用于保存模型状态
hbPackedDNNHandle_t g_packed_dnn_handle;
hbDNNHandle_t g_dnn_handle;
hbDNNTensor g_input;
hbDNNTensor* g_output = nullptr;
int g_output_count = 0;
int g_input_H = 0;
int g_input_W = 0;

// 模型加载函数
bool load_model_wrapper(const std::string& model_path) {
    int order[6] = {0, 1, 2, 3, 4, 5};  // Default order
    
    // 加载模型
    if (loadModel(g_packed_dnn_handle, g_dnn_handle, g_input_H, g_input_W, order) != 0) {
        throw std::runtime_error("Failed to load model");
    }
    
    // 获取输出数量
    if (hbDNNGetOutputCount(&g_output_count, g_dnn_handle) != 0) {
        throw std::runtime_error("Failed to get output count");
    }
    
    // 准备输入/输出张量
    if (prepareTensors(g_input, g_output, g_dnn_handle, g_output_count, g_input_H, g_input_W) != 0) {
        throw std::runtime_error("Failed to prepare tensors");
    }
    
    return true;
}

// 球体检测函数
std::vector<std::pair<float, float>> detect_balls_wrapper(py::array_t<uint8_t> image) {
    // 将numpy数组转换为cv::Mat
    py::buffer_info buf = image.request();
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = buf.shape[2];
    
    cv::Mat frame(rows, cols, CV_8UC3, (unsigned char*)buf.ptr);
    
    // 结果存储
    std::vector<std::pair<float, float>> ball_centers;
    std::vector<std::vector<cv::Rect2d>> bboxes;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<int>> indices;
    
    // 预处理图像并运行推理
    if (runInference(frame, g_input, g_output, g_dnn_handle, g_output_count) != 0) {
        throw std::runtime_error("Inference failed");
    }
    
    // 后处理推理结果
    float x_scale = 1.0f, y_scale = 1.0f;
    int x_shift = 0, y_shift = 0;
    
    if (postProcess(g_output, g_output_count, bboxes, scores, indices, 
                   x_scale, y_scale, x_shift, y_shift) != 0) {
        throw std::runtime_error("Post-processing failed");
    }
    
    // 提取球体位置 (类别ID 32是COCO数据集中的"sports ball")
    const int ballClassId = 32;
    if (ballClassId < bboxes.size() && !indices[ballClassId].empty()) {
        for (int i = 0; i < indices[ballClassId].size(); i++) {
            int idx = indices[ballClassId][i];
            
            // 计算球的中心点
            float x1 = (bboxes[ballClassId][idx].x - x_shift) / x_scale;
            float y1 = (bboxes[ballClassId][idx].y - y_shift) / y_scale;
            float width = bboxes[ballClassId][idx].width / x_scale;
            float height = bboxes[ballClassId][idx].height / y_scale;
            
            float center_x = x1 + width / 2;
            float center_y = y1 + height / 2;
            
            ball_centers.push_back(std::make_pair(center_x, center_y));
        }
    }
    
    return ball_centers;
}

// 清理资源
void cleanup_model_wrapper() {
    cleanupModel(g_packed_dnn_handle, g_input, g_output, g_output_count);
}

PYBIND11_MODULE(ball_detector_cpp, m) {
    m.doc() = "Ball detection module using YOLO11";
    
    m.def("load_model", &load_model_wrapper, "Load YOLO11 model", 
          py::arg("model_path"));
    m.def("detect_balls", &detect_balls_wrapper, "Detect balls in an image and return center coordinates");
    m.def("cleanup_model", &cleanup_model_wrapper, "Clean up model resources");
}