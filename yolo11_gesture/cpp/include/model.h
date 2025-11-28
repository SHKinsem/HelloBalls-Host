#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

#include "config.h"
#include "utils.h"

// Model loading and initialization
int loadModel(hbPackedDNNHandle_t &packed_dnn_handle, hbDNNHandle_t &dnn_handle, 
              int &input_H, int &input_W, int order[6]);

// Prepare model inputs/outputs
int prepareTensors(hbDNNTensor &input, hbDNNTensor *&output, 
                  hbDNNHandle_t dnn_handle, int output_count, 
                  int input_H, int input_W);

// Run model inference
int runInference(hbDNNHandle_t dnn_handle, hbDNNTensor &input, 
                hbDNNTensor *output, hbDNNTaskHandle_t &task_handle);

// Clean up resources
void cleanupModel(hbPackedDNNHandle_t &packed_dnn_handle, 
                 hbDNNTensor &input, hbDNNTensor *output, int output_count);