// gesture_yolo_api.cpp  (hb_dnn C API 版本)
// 功能：加载 YOLO .bin（hb_dnn），支持 NV12/BGR 输入，解析多尺度网格输出并做 NMS
// 生成模块：gesture_yolo_api.cpython-xxx-aarch64-linux-gnu.so

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// === Horizon C API ===
#include <dnn/hb_dnn.h>   // 你机器上确有此头
#include <hb_sys.h>       // 系统内存管理（hbSysAllocCachedMem/Flush/Free）

namespace py = pybind11;

// ---------- 小工具 ----------
static inline float sigmoid(float x){ return 1.f / (1.f + std::exp(-x)); }

static std::vector<int> nms_keep(const std::vector<float>& boxes, const std::vector<float>& scores,
                                 float iou_thres, int topk=300) {
    // boxes: [N,4]=x1,y1,x2,y2
    const int N = (int)scores.size();
    std::vector<int> order(N); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a,int b){ return scores[a] > scores[b]; });

    std::vector<int> keep; keep.reserve(N);
    for (int u=0; u<N && (int)keep.size()<topk; ++u) {
        int i = order[u];
        bool ok = true;

        float x1i=boxes[i*4+0], y1i=boxes[i*4+1], x2i=boxes[i*4+2], y2i=boxes[i*4+3];
        float areai = std::max(0.f, x2i-x1i+1) * std::max(0.f, y2i-y1i+1);

        for (int j: keep) {
            float x1j=boxes[j*4+0], y1j=boxes[j*4+1], x2j=boxes[j*4+2], y2j=boxes[j*4+3];
            float xx1=std::max(x1i,x1j), yy1=std::max(y1i,y1j);
            float xx2=std::min(x2i,x2j), yy2=std::min(y2i,y2j);
            float w=std::max(0.f, xx2-xx1+1), h=std::max(0.f, yy2-yy1+1);
            float inter=w*h;
            float areaj=std::max(0.f, x2j-x1j+1) * std::max(0.f, y2j-y1j+1);
            float iou= inter / (areai+areaj-inter + 1e-6f);
            if (iou > iou_thres) { ok=false; break; }
        }
        if (ok) keep.push_back(i);
    }
    return keep;
}

// BGR(HxWx3,u8)->NV12(H*1.5 x W,u8)
static py::array_t<uint8_t> bgr_to_nv12(const py::array_t<uint8_t>& bgr){
    auto info = bgr.request();
    if (info.ndim!=3 || info.shape[2]!=3 || info.itemsize!=1)
        throw std::runtime_error("bgr_to_nv12 expects uint8 HxWx3");
    int h=(int)info.shape[0], w=(int)info.shape[1];
    const uint8_t* src=(const uint8_t*)info.ptr;
    cv::Mat bgrm(h,w,CV_8UC3,(void*)src);
    cv::Mat yuv;
    cv::cvtColor(bgrm, yuv, cv::COLOR_BGR2YUV_I420);

    const int Ysz=w*h, UVsz=(w/2)*(h/2);
    const uint8_t* Y = yuv.ptr<uint8_t>(0);
    const uint8_t* U = Y + Ysz;
    const uint8_t* V = U + UVsz;

    py::array_t<uint8_t> nv12({h*3/2, w});
    auto ninfo = nv12.request();
    uint8_t* dst=(uint8_t*)ninfo.ptr;
    std::memcpy(dst, Y, Ysz);
    uint8_t* UV = dst + Ysz;
    for(int j=0;j<h/2;++j){
        for(int i=0;i<w/2;++i){
            UV[j*w + 2*i + 0] = U[j*(w/2)+i];
            UV[j*w + 2*i + 1] = V[j*(w/2)+i];
        }
    }
    return nv12;
}

// ---------- hb_dnn 封装 ----------
class YoloHB {
public:
    YoloHB(): packed_(nullptr), dnn_(nullptr),
              in_w_(640), in_h_(640), num_classes_(5) {}

    // 注意：model_name 可留空，默认取第0个
    bool load(const std::string& model_path, int num_classes,
              int in_w=640, int in_h=640, const std::string& model_name="") {
        num_classes_ = num_classes;
        in_w_ = in_w; in_h_ = in_h;

        const char* files[1] = { model_path.c_str() };
        if (hbDNNInitializeFromFiles(&packed_, files, 1) != 0) return false;   // 创建 packed handle  :contentReference[oaicite:1]{index=1}

        // 取模型名并拿 dnn handle
        const char** names=nullptr; int name_cnt=0;
        if (hbDNNGetModelNameList(&names, &name_cnt, packed_) != 0 || name_cnt<=0) return false;  // :contentReference[oaicite:2]{index=2}
        const char* pick = model_name.empty() ? names[0] : model_name.c_str();
        if (hbDNNGetModelHandle(&dnn_, packed_, pick) != 0) return false;  // :contentReference[oaicite:3]{index=3}

        // 输入属性（用于安全检查）
        int in_cnt=0; hbDNNGetInputCount(&in_cnt, dnn_);                  // :contentReference[oaicite:4]{index=4}
        if (in_cnt != 1) throw std::runtime_error("expect single input");
        hbDNNGetInputTensorProperties(&in_prop_, dnn_, 0);                // :contentReference[oaicite:5]{index=5}

        // 输出个数
        hbDNNGetOutputCount(&out_cnt_, dnn_);                             // :contentReference[oaicite:6]{index=6}
        if (out_cnt_ <= 0) throw std::runtime_error("no outputs");

        // 预取每个输出属性
        out_props_.resize(out_cnt_);
        for (int i=0;i<out_cnt_;++i)
            hbDNNGetOutputTensorProperties(&out_props_[i], dnn_, i);      // :contentReference[oaicite:7]{index=7}
        return true;
    }

    // 直接喂 NV12 (H*1.5 x W, u8)
    py::dict infer_nv12(const py::array_t<uint8_t>& nv12,
                        float conf_thres=0.30f, float iou_thres=0.45f) {
        auto info = nv12.request();
        if (info.ndim!=2 || info.itemsize!=1 ||
            info.shape[0] != in_h_*3/2 || info.shape[1] != in_w_)
            throw std::runtime_error("nv12 must be uint8, shape (H*1.5, W)");

        // --- 准备输入 tensor ---
        hbDNNTensor in_tensor{};
        in_tensor.properties = in_prop_;
        const int need = in_w_ * (in_h_*3/2);
        if (hbSysAllocCachedMem(&in_tensor.sysMem[0], need) != 0)
            throw std::runtime_error("hbSysAllocCachedMem input failed");
        std::memcpy(in_tensor.sysMem[0].virAddr, info.ptr, need);
        hbSysFlushMem(&in_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

        // --- 准备输出 tensor 内存（按 alignedByteSize 分配） ---
        std::vector<hbDNNTensor> outs(out_cnt_);
        for (int i=0;i<out_cnt_;++i){
            outs[i].properties = out_props_[i];
            if (hbSysAllocCachedMem(&outs[i].sysMem[0], (uint32_t)out_props_[i].alignedByteSize) != 0)
                throw std::runtime_error("hbSysAllocCachedMem output failed");
        }

        // --- 推理 ---
        hbDNNTaskHandle_t task = nullptr;
        hbDNNInferCtrlParam ctrl; HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);   // :contentReference[oaicite:8]{index=8}
        hbDNNTensor* outs_ptr = outs.data();
        if (hbDNNInfer(&task, &outs_ptr, &in_tensor, dnn_, &ctrl) != 0)
            throw std::runtime_error("hbDNNInfer failed");
        if (hbDNNWaitTaskDone(task, 0) != 0)
            throw std::runtime_error("hbDNNWaitTaskDone timeout/fail");

        // --- 解析输出：自动配对 (H,W,64) 作为 cls/obj，(H,W,4) 作为 box ---
        struct Head { int H,W; std::vector<float> cls; std::vector<float> box; };
        std::map<std::pair<int,int>, Head> heads;

        for (int i=0;i<out_cnt_;++i){
            auto &p = out_props_[i];
            const int* dims = p.validShape.dimensionSize; // (N,H,W,C)
            int H = dims[1], W = dims[2], C = dims[3];

            // 取原始数据指针
            uint8_t* v = (uint8_t*)outs[i].sysMem[0].virAddr;

            std::vector<float> buf; buf.resize((size_t)H*W*C);

            // 反量化/拷贝到 float
            if (p.quantiType == SCALE && p.scale.scaleData != nullptr && p.quantizeAxis == 3) {
                // int8 per-channel → float   （按通道 scale）
                const float* scales = p.scale.scaleData;
                const int chan = C;
                const int hw = H*W;
                const int8_t* src = (const int8_t*)v;
                for (int c=0;c<chan;++c){
                    float s = scales[c];
                    for (int idx=0; idx<hw; ++idx)
                        buf[c + idx*chan] = src[c + idx*chan] * s;
                }
            } else {
                // 假定 float32
                std::memcpy(buf.data(), v, (size_t)H*W*C*sizeof(float));
            }

            auto key = std::make_pair(H,W);
            auto &hd = heads[key];
            hd.H=H; hd.W=W;
            if (C==4) hd.box = std::move(buf);
            else      hd.cls = std::move(buf); // 约定：cls[0]=obj，后面是 K 类
        }

        // --- 解码 + 合并 ---
        std::vector<float> boxes; boxes.reserve(10000*4);
        std::vector<float> scores; scores.reserve(10000);
        std::vector<int>   clses;  clses.reserve(10000);

        for (auto &kv : heads){
            auto &hd = kv.second;
            if (hd.box.empty() || hd.cls.empty()) continue;
            const int H=hd.H, W=hd.W, hw=H*W;
            const int C = (int)(hd.cls.size() / hw); // obj + K
            const int K = C - 1;
            if (K != num_classes_) {
                // 容错：若导出通道数与 num_classes 不一致，取 min
                // 也可能你的模型没有显式 obj 通道，这里默认有 obj
            }
            const int stride = in_h_ / H;  // 80->8, 40->16, 20->32（假设输入 640）

            for (int i=0;i<H;++i){
                for (int j=0;j<W;++j){
                    const int idx = i*W + j;
                    float obj = sigmoid(hd.cls[idx*C + 0]);
                    // 选最大类别
                    int best_c=-1; float best_p=-1e9f;
                    for (int c=0;c<num_classes_ && c+1<C; ++c){
                        float p = sigmoid(hd.cls[idx*C + 1 + c]);
                        if (p > best_p){ best_p=p; best_c=c; }
                    }
                    float conf = obj * best_p;
                    if (conf < 1e-6f) continue;

                    // box: dx,dy,dw,dh
                    float dx = hd.box[idx*4+0];
                    float dy = hd.box[idx*4+1];
                    float dw = hd.box[idx*4+2];
                    float dh = hd.box[idx*4+3];

                    float cx = (j + (sigmoid(dx)*2.f - 0.5f)) * stride;
                    float cy = (i + (sigmoid(dy)*2.f - 0.5f)) * stride;
                    float ww = std::pow(sigmoid(dw)*2.f, 2.f) * stride;
                    float hh = std::pow(sigmoid(dh)*2.f, 2.f) * stride;

                    float x1 = cx - ww/2, y1 = cy - hh/2;
                    float x2 = cx + ww/2, y2 = cy + hh/2;

                    // 阈值
                    if (conf >= conf_thres){
                        boxes.insert(boxes.end(), {x1,y1,x2,y2});
                        scores.push_back(conf);
                        clses.push_back(best_c<0?0:best_c);
                    }
                }
            }
        }

        // --- NMS ---
        auto keep = nms_keep(boxes, scores, iou_thres, 300);

        // --- 组装返回 ---
        py::array_t<int> class_ids({(int)keep.size()});
        py::array_t<float> bboxes({(int)keep.size(),4});
        py::array_t<float> confs({(int)keep.size()});

        auto ci = class_ids.mutable_unchecked<1>();
        auto bb = bboxes.mutable_unchecked<2>();
        auto sc = confs. mutable_unchecked<1>();

        for (int k=0;k<(int)keep.size();++k){
            int i = keep[k];
            float x1=boxes[i*4+0], y1=boxes[i*4+1];
            float x2=boxes[i*4+2], y2=boxes[i*4+3];
            ci(k) = clses[i];
            bb(k,0)=x1; bb(k,1)=y1; bb(k,2)=x2-x1; bb(k,3)=y2-y1;
            sc(k) = scores[i];
        }

        // 释放内存
        for (auto &t : outs) hbSysFreeMem(&t.sysMem[0]);
        hbSysFreeMem(&in_tensor.sysMem[0]);

        py::dict ret;
        ret["class_ids"]=class_ids;
        ret["bboxes"]=bboxes;
        ret["scores"]=confs;
        return ret;
    }

    // 吃 640×640 BGR（预处理后），内部转 NV12 再推理
    py::dict infer_bgr(const py::array_t<uint8_t>& bgr,
                       float conf_thres=0.30f, float iou_thres=0.45f){
        auto info = bgr.request();
        if (info.ndim!=3 || info.shape[2]!=3 || info.itemsize!=1 ||
            info.shape[0]!=in_h_ || info.shape[1]!=in_w_)
            throw std::runtime_error("expect uint8 BGR HxWx3 matching model input");
        auto nv12 = bgr_to_nv12(bgr);
        return infer_nv12(nv12, conf_thres, iou_thres);
    }

private:
    hbDNNPackedHandle_t packed_;
    hbDNNHandle_t       dnn_;
    hbDNNTensorProperties in_prop_{};
    int out_cnt_{0};
    std::vector<hbDNNTensorProperties> out_props_;

    int in_w_, in_h_, num_classes_;
};

PYBIND11_MODULE(gesture_yolo_api, m){
    py::class_<YoloHB>(m, "YoloHB")
        .def(py::init<>())
        .def("load", &YoloHB::load,
             py::arg("model_path"), py::arg("num_classes"),
             py::arg("in_w")=640, py::arg("in_h")=640, py::arg("model_name")="")
        .def("infer_nv12", &YoloHB::infer_nv12,
             py::arg("nv12"), py::arg("conf_thres")=0.30f, py::arg("iou_thres")=0.45f)
        .def("infer_bgr",  &YoloHB::infer_bgr,
             py::arg("bgr"),  py::arg("conf_thres")=0.30f, py::arg("iou_thres")=0.45f);
}
