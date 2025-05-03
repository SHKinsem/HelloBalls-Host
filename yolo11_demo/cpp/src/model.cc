#include "include/model.h"

int loadModel(hbPackedDNNHandle_t &packed_dnn_handle, hbDNNHandle_t &dnn_handle, 
              int &input_H, int &input_W, int order[6]) {
    // 0. Load bin model
    auto begin_time = std::chrono::system_clock::now();
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    std::cout << "\033[31m Load D-Robotics Quantize model time = " << std::fixed << std::setprecision(2) 
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 
              << " ms\033[0m" << std::endl;

    // 1. Print versions and configuration
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "[INFO] MODEL_PATH: " << MODEL_PATH << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
    std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;
    std::cout << "[INFO] CAMERA_ID: " << CAMERA_ID << std::endl;

    // 2. Print model information

    // 2.1 Model name
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    // If this model has multiple packs, use only the first one
    if (model_count > 1) {
        std::cout << "This model file have more than 1 model, only use model 0.";
    }
    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;

    // 2.2 Get the first model handle from the packed model
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 2.3 Check model inputs
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");

    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");

    // 2.3.1 D-Robotics YOLO11 *.bin model should have a single input
    if (input_count > 1) {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    // 2.3.2 D-Robotics YOLO11 *.bin input tensor type should be nv12
    if (input_properties.tensorType == HB_DNN_IMG_TYPE_NV12) {
        std::cout << "input tensor type: HB_DNN_IMG_TYPE_NV12" << std::endl;
    } else {
        std::cout << "input tensor type is not HB_DNN_IMG_TYPE_NV12, please check!" << std::endl;
        return -1;
    }

    // 2.3.3 D-Robotics YOLO11 *.bin input tensor layout should be NCHW
    if (input_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        std::cout << "input tensor layout: HB_DNN_LAYOUT_NCHW" << std::endl;
    } else {
        std::cout << "input tensor layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        return -1;
    }

    // 2.3.4 D-Robotics YOLO11 *.bin input tensor valid shape should be (1,3,H,W)
    if (input_properties.validShape.numDimensions == 4) {
        input_H = input_properties.validShape.dimensionSize[2];
        input_W = input_properties.validShape.dimensionSize[3];
        std::cout << "input tensor valid shape: (" << input_properties.validShape.dimensionSize[0];
        std::cout << ", " << input_properties.validShape.dimensionSize[1];
        std::cout << ", " << input_H;
        std::cout << ", " << input_W << ")" << std::endl;
    } else {
        std::cout << "input tensor validShape.numDimensions is not 4 such as (1,3,640,640), please check!" << std::endl;
        return -1;
    }

    // 2.4 Check model outputs
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");

    // 2.4.1 D-Robotics YOLO11 *.bin model should have 6 outputs
    if (output_count == 6) {
        for (int i = 0; i < 6; i++) {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
                "hbDNNGetOutputTensorProperties failed");
            std::cout << "output[" << i << "] ";
            std::cout << "valid shape: (" << output_properties.validShape.dimensionSize[0];
            std::cout << ", " << output_properties.validShape.dimensionSize[1];
            std::cout << ", " << output_properties.validShape.dimensionSize[2];
            std::cout << ", " << output_properties.validShape.dimensionSize[3] << "), ";
            if (output_properties.quantiType == SHIFT)
                std::cout << "QuantiType: SHIFT" << std::endl;
            if (output_properties.quantiType == SCALE)
                std::cout << "QuantiType: SCALE" << std::endl;
            if (output_properties.quantiType == NONE)
                std::cout << "QuantiType: NONE" << std::endl;
        }
    } else {
        std::cout << "Your Model's outputs num is not 6, please check!" << std::endl;
        return -1;
    }

    // 2.4.2 Adjust output order mapping
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[6][3] = {
        {H_8, W_8, CLASSES_NUM},   // output[order[3]]: (1, H // 8,  W // 8,  CLASSES_NUM)
        {H_8, W_8, 64},            // output[order[0]]: (1, H // 8,  W // 8,  64)
        {H_16, W_16, CLASSES_NUM}, // output[order[4]]: (1, H // 16, W // 16, CLASSES_NUM)
        {H_16, W_16, 64},          // output[order[1]]: (1, H // 16, W // 16, 64)
        {H_32, W_32, CLASSES_NUM}, // output[order[5]]: (1, H // 32, W // 32, CLASSES_NUM)
        {H_32, W_32, 64},          // output[order[2]]: (1, H // 32, W // 32, 64)
    };
    
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, j),
                "hbDNNGetOutputTensorProperties failed");
            int32_t h = output_properties.validShape.dimensionSize[1];
            int32_t w = output_properties.validShape.dimensionSize[2];
            int32_t c = output_properties.validShape.dimensionSize[3];
            if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2]) {
                order[i] = j;
                break;
            }
        }
    }

    // 2.4.3 Print and check adjusted output order mapping
    if (order[0] + order[1] + order[2] + order[3] + order[4] + order[5] == 0 + 1 + 2 + 3 + 4 + 5) {
        std::cout << "Outputs order check SUCCESS, continue." << std::endl;
        std::cout << "order = {";
        for (int i = 0; i < 6; i++) {
            std::cout << order[i] << ", ";
        }
        std::cout << "}" << std::endl;
    } else {
        std::cout << "Outputs order check FAILED, use default" << std::endl;
        for (int i = 0; i < 6; i++)
            order[i] = i;
    }
    
    return 0;
}

int prepareTensors(hbDNNTensor &input, hbDNNTensor *&output, 
                  hbDNNHandle_t dnn_handle, int output_count, 
                  int input_H, int input_W) {
    // Prepare output data space
    output = new hbDNNTensor[output_count];
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        hbSysAllocCachedMem(&mem, out_aligned_size);
    }

    // Get input tensor properties
    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");
    
    // Allocate memory for input tensor
    input.properties = input_properties;
    hbSysAllocCachedMem(&input.sysMem[0], int(3 * input_H * input_W / 2));
    
    return 0;
}

int runInference(hbDNNHandle_t dnn_handle, hbDNNTensor &input, 
                hbDNNTensor *output, hbDNNTaskHandle_t &task_handle) {
    // Run inference
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    RDK_CHECK_SUCCESS(
        hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param),
        "hbDNNInfer failed");
    
    // Wait for task to complete
    RDK_CHECK_SUCCESS(
        hbDNNWaitTaskDone(task_handle, 0),
        "hbDNNWaitTaskDone failed");
        
    return 0;
}

void cleanupModel(hbPackedDNNHandle_t &packed_dnn_handle, 
                 hbDNNTensor &input, hbDNNTensor *output, int output_count) {
    // Free input memory
    hbSysFreeMem(&(input.sysMem[0]));
    
    // Free output memory
    for (int i = 0; i < output_count; i++) {
        hbSysFreeMem(&(output[i].sysMem[0]));
    }
    delete[] output;
    
    // Release model
    hbDNNRelease(packed_dnn_handle);
}