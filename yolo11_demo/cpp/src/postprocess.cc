#include "postprocess.h"
#include "config.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>

void postprocess(hbDNNTensor* output, int order[6],
                int input_H, int input_W,
                std::vector<std::vector<cv::Rect2d>>& bboxes,
                std::vector<std::vector<float>>& scores) {
    
    float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);
    bboxes.resize(CLASSES_NUM);
    scores.resize(CLASSES_NUM);

    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    
    // Small feature map
    hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    auto *s_cls_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
    auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
    auto *s_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);
    
    for (int h = 0; h < H_8; h++) {
        for (int w = 0; w < W_8; w++) {
            float *cur_s_cls_raw = s_cls_raw;
            int32_t *cur_s_bbox_raw = s_bbox_raw;

            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++) {
                if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }

            if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW) {
                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
                continue;
            }

            float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++) {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++) {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_s_bbox_raw[index_id]) * s_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
                continue;
            }

            float x1 = (w + 0.5 - ltrb[0]) * 8.0;
            float y1 = (h + 0.5 - ltrb[1]) * 8.0;
            float x2 = (w + 0.5 + ltrb[2]) * 8.0;
            float y2 = (h + 0.5 + ltrb[3]) * 8.0;

            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            s_cls_raw += CLASSES_NUM;
            s_bbox_raw += REG * 4;
        }
    }

    // Medium feature map
    hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    auto *m_cls_raw = reinterpret_cast<float *>(output[order[2]].sysMem[0].virAddr);
    auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[3]].sysMem[0].virAddr);
    auto *m_bbox_scale = reinterpret_cast<float *>(output[order[3]].properties.scale.scaleData);
    
    for (int h = 0; h < H_16; h++) {
        for (int w = 0; w < W_16; w++) {
            float *cur_m_cls_raw = m_cls_raw;
            int32_t *cur_m_bbox_raw = m_bbox_raw;

            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++) {
                if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }

            if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW) {
                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
                continue;
            }

            float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++) {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++) {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_m_bbox_raw[index_id]) * m_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
                continue;
            }

            float x1 = (w + 0.5 - ltrb[0]) * 16.0;
            float y1 = (h + 0.5 - ltrb[1]) * 16.0;
            float x2 = (w + 0.5 + ltrb[2]) * 16.0;
            float y2 = (h + 0.5 + ltrb[3]) * 16.0;

            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            m_cls_raw += CLASSES_NUM;
            m_bbox_raw += REG * 4;
        }
    }

    // Large feature map
    hbSysFlushMem(&(output[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    auto *l_cls_raw = reinterpret_cast<float *>(output[order[4]].sysMem[0].virAddr);
    auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[5]].sysMem[0].virAddr);
    auto *l_bbox_scale = reinterpret_cast<float *>(output[order[5]].properties.scale.scaleData);
    
    for (int h = 0; h < H_32; h++) {
        for (int w = 0; w < W_32; w++) {
            float *cur_l_cls_raw = l_cls_raw;
            int32_t *cur_l_bbox_raw = l_bbox_raw;

            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++) {
                if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }

            if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW) {
                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
                continue;
            }

            float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++) {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++) {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_l_bbox_raw[index_id]) * l_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
                continue;
            }

            float x1 = (w + 0.5 - ltrb[0]) * 32.0;
            float y1 = (h + 0.5 - ltrb[1]) * 32.0;
            float x2 = (w + 0.5 + ltrb[2]) * 32.0;
            float y2 = (h + 0.5 + ltrb[3]) * 32.0;

            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            l_cls_raw += CLASSES_NUM;
            l_bbox_raw += REG * 4;
        }
    }
}

void applyNMS(const std::vector<std::vector<cv::Rect2d>>& bboxes,
             const std::vector<std::vector<float>>& scores,
             std::vector<std::vector<int>>& indices) {
    
    indices.resize(CLASSES_NUM);
    for (int i = 0; i < CLASSES_NUM; i++) {
        float threshold = (i == 32) ? SPORTS_BALL_THRESHOLD : SCORE_THRESHOLD;
        cv::dnn::NMSBoxes(bboxes[i], scores[i], threshold, NMS_THRESHOLD, indices[i], 1.f, NMS_TOP_K);
    }
}

void drawDetections(cv::Mat& frame,
                   const std::vector<std::vector<cv::Rect2d>>& bboxes,
                   const std::vector<std::vector<float>>& scores,
                   const std::vector<std::vector<int>>& indices,
                   float x_scale, float y_scale, int x_shift, int y_shift) {
    
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
        if (!isTargetClass(cls_id)) continue; // Only process target classes (person & sports ball)
        
        for (auto it = indices[cls_id].begin(); it != indices[cls_id].end(); ++it) {
            float x1 = (bboxes[cls_id][*it].x - x_shift) / x_scale;
            float height = frame.rows;
            float y1 = (bboxes[cls_id][*it].y - y_shift) / y_scale;
            float x2 = x1 + (bboxes[cls_id][*it].width) / x_scale;
            float y2 = y1 + (bboxes[cls_id][*it].height) / y_scale;
            float score = scores[cls_id][*it];
            
            std::string name = object_names[cls_id % CLASSES_NUM];

            // Calculate center point
            float center_x = (x1 + x2) / 2.0;
            float center_y = (y1 + y2) / 2.0;
            float correct_y = height - center_y; // Correct y-coordinate for OpenCV
            // Draw rectangle
            cv::Scalar color = (cls_id == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255); // Green for person, Red for sports ball
            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, LINE_SIZE);

            // Draw label text
            std::string text = name + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
            cv::putText(frame, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, color, FONT_THICKNESS, cv::LINE_AA);

            // For sports balls, print central point and draw a marker
            if (name == "sports ball") {
                // Print central point coordinates
                std::cout << "Sports Ball Central Point: (" << center_x << ", " << correct_y << ")" << std::endl;
                
                // Draw central point with a cross marker
                cv::drawMarker(frame, cv::Point(center_x, center_y), cv::Scalar(255, 255, 0), 
                               cv::MARKER_CROSS, 10, 2);
                
                // Display central point coordinates on the frame
                std::string center_text = "Center: (" + std::to_string(int(center_x)) + 
                                         ", " + std::to_string(int(correct_y)) + ")";
                cv::putText(frame, center_text, cv::Point(x1, y2 + 20), 
                           cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(255, 255, 0), 
                           FONT_THICKNESS, cv::LINE_AA);
            }
        }
    }
}