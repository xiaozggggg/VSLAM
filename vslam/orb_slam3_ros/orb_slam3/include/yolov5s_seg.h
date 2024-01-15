#ifndef YOLOV5S_SEG_TRT_H
#define YOLOV5S_SEG_TRT_H
#include "../../model/config.h"
#include "../../model/cuda_utils.h"
#include "../../model/logging.h"
#include "../../model/utils.h"
#include "../../model/preprocess.h"
#include "../../model/postprocess.h"
#include "../../model/model.h"
// #include "NvInfer.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>  
#include <algorithm>
#include <utility>
#include <time.h>


// Mr.Chen:add_new.
extern std::string coco_file;
extern std::string engine_file;
using namespace std;
using namespace nvinfer1;

namespace ORB_SLAM3
{
struct RecDepth{
        cv::Rect2i RecArea;
        // float depth;
};


// class StereoDepth;
class YOLOv5_seg
{
public:
    struct DetectRes{
        std::string classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
};


public:
    YOLOv5_seg();
    ~YOLOv5_seg();
   
    ////////////////////////////////////////////////////////


public:
    void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2);
    void infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize);
    void serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name);
    void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);
    cv::Mat scale_mask(cv::Mat mask, cv::Mat img);
    cv::Mat convertTo3Channels(const cv::Mat &binImg);
    void detect(cv::Mat &src_img_L, cv::Mat &src_img_R);

public:
    std::string engine_name;
    std::string labels_filename;
    float gd;
    float gw;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream;
    float* gpu_buffers[3];
    float* cpu_output_buffer1 = nullptr;
    float* cpu_output_buffer2 = nullptr;
    std::unordered_map<int, std::string> labels_map;
    std::vector<cv::Mat> img_batch;
    std::vector<std::string> img_name_batch;


    //bboxes && masks;
    std::vector<std::string> mClassnames;

    vector<string> mvDynamicNames;
    vector<cv::Rect2i> mvDynamicArea;
    // map<string, vector<cv::Rect2i>> mmDetectMap;
    map<string, vector<RecDepth>> mmDetectMap;

    cv::Mat img_L;
    cv::Mat img_R;
};

}  //namespace ORB_SLAM
#endif //YOLOV5S_SEG_TRT_H
