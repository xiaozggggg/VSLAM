#include "../../model/config.h"
#include "../../model/cuda_utils.h"
#include "../../model/logging.h"
#include "../../model/utils.h"
#include "../../model/preprocess.h"
#include "../../model/postprocess.h"
#include "../../model/model.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>  
#include "yolov5s_seg.h"

using namespace nvinfer1;
namespace ORB_SLAM3
{

static Logger gLogger;
const static int kOutputSize1 = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSize2 = 32 * (kInputH / 4) * (kInputW / 4);
static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A,
                                         0x92CC17, 0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF,
                                         0x344593, 0x6473FF, 0x0018EC, 0x8438FF, 0x520085, 0xCB38FF,
                                         0xFF95C8, 0xFF37C7};

YOLOv5_seg::YOLOv5_seg()
{
  /*
  std::ifstream f("/home/hui-lian/ChenShiKai/cl/seg_catkin_orb_slam3_ws/src/SLAM/coco.txt");
  if (!f.is_open())
  {
      std::cerr << "read coco_name file error"  << std::endl;
  }
  */
  std::ifstream f(coco_file);
  if (!f.is_open())
  {
      std::cerr << "read coco_name file error"  << std::endl;
  }

  std::string name = "";
  while (std::getline(f, name))
  {
      mClassnames.push_back(name);
  }
  // mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
  //                   "dog", "horse", "sheep", "crow", "bear"};
  mvDynamicNames = {"person"};
  // mvDynamicNames = mClassnames;

  cudaSetDevice(kGpuId);
  // engine_name = "/home/hui-lian/ChenShiKai/cl/seg_catkin_orb_slam3_ws/src/SLAM/yolov5s-seg.engine";

  engine_name = engine_file;
  // labels_filename = "/home/hui-lian/ChenShiKai/cl/seg_catkin_orb_slam3_ws/src/SLAM/coco.txt";
  gd = 0.0f; 
  gw = 0.0f;

  deserialize_engine(engine_file, &runtime, &engine, &context);
  CUDA_CHECK(cudaStreamCreate(&stream));
  // Init CUDA preprocessing
  cuda_preprocess_init(kMaxInputImageSize);
  prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &gpu_buffers[2], &cpu_output_buffer1, &cpu_output_buffer2);
  read_labels(coco_file, labels_map);
  assert(kNumClass == labels_map.size());
}


YOLOv5_seg::~YOLOv5_seg()
{
  delete[] cpu_output_buffer1;
  delete[] cpu_output_buffer2;
}



void YOLOv5_seg::prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer1, float** gpu_output_buffer2, float** cpu_output_buffer1, float** cpu_output_buffer2) {
  assert(engine->getNbBindings() == 3);
  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine->getBindingIndex(kInputTensorName);
  const int outputIndex1 = engine->getBindingIndex(kOutputTensorName);
  const int outputIndex2 = engine->getBindingIndex("proto");
  assert(inputIndex == 0);
  assert(outputIndex1 == 1);
  assert(outputIndex2 == 2);

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer1, kBatchSize * kOutputSize1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer2, kBatchSize * kOutputSize2 * sizeof(float)));

  // Alloc CPU buffers
  *cpu_output_buffer1 = new float[kBatchSize * kOutputSize1];
  *cpu_output_buffer2 = new float[kBatchSize * kOutputSize2];
}

void YOLOv5_seg::infer(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output1, float* output2, int batchSize) {
  context.enqueue(batchSize, buffers, stream, nullptr);
  CUDA_CHECK(cudaMemcpyAsync(output1, buffers[1], batchSize * kOutputSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(output2, buffers[2], batchSize * kOutputSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

void YOLOv5_seg::serialize_engine(unsigned int max_batchsize, float& gd, float& gw, std::string& wts_name, std::string& engine_name) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = nullptr;

  engine = build_seg_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);

  assert(engine != nullptr);

  // Serialize the engine
  IHostMemory* serialized_engine = engine->serialize();
  assert(serialized_engine != nullptr);

  // Save engine to file
  std::ofstream p(engine_name, std::ios::binary);
  if (!p) {
    std::cerr << "Could not open plan output file" << std::endl;
    assert(false);
  }
  p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

  // Close everything down
  engine->destroy();
  config->destroy();
  serialized_engine->destroy();
  builder->destroy();
}

void YOLOv5_seg::deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context) {
  std::ifstream file(engine_name, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << engine_name << " error!" << std::endl;
    assert(false);
  }
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  char* serialized_engine = new char[size];
  assert(serialized_engine);
  file.read(serialized_engine, size);
  file.close();

  *runtime = createInferRuntime(gLogger);
  assert(*runtime);
  *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
  assert(*engine);
  *context = (*engine)->createExecutionContext();
  assert(*context);
  delete[] serialized_engine;
}


cv::Mat YOLOv5_seg::scale_mask(cv::Mat mask, cv::Mat img) {
  int x, y, w, h;
  float r_w = kInputW / (img.cols * 1.0);
  float r_h = kInputH / (img.rows * 1.0);
  if (r_h > r_w) {
    w = kInputW;
    h = r_w * img.rows;
    x = 0;
    y = (kInputH - h) / 2;
  } else {
    w = r_h * img.cols;
    h = kInputH;
    x = (kInputW - w) / 2;
    y = 0;
  }
  cv::Rect r(x, y, w, h);
  cv::Mat res;
  cv::resize(mask(r), res, img.size());
  return res;
}


cv::Mat YOLOv5_seg::convertTo3Channels(const cv::Mat &binImg)
{
    cv::Mat three_channel = cv::Mat::zeros(binImg.rows,binImg.cols,CV_8UC3);
    vector<cv::Mat> channels;
    for(int i = 0;i < 3; i ++)
    {
        channels.push_back(binImg);
    }
    cv::merge(channels,three_channel);
    return three_channel;
}

void YOLOv5_seg::detect(cv::Mat &src_img_L, cv::Mat &src_img_R) {


  img_L = src_img_L.clone();
  img_R = src_img_R.clone();
  img_L = convertTo3Channels(img_L);
  img_R = convertTo3Channels(img_R);
  img_batch.clear();
  img_name_batch.clear();
  img_batch.push_back(img_L);
  img_name_batch.push_back("1");
 
  // Preprocess
  cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);

  // Run inference
  auto start = std::chrono::system_clock::now();
  infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer1, cpu_output_buffer2, kBatchSize);
  auto end = std::chrono::system_clock::now();
  // std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
  
  // NMS
  std::vector<std::vector<Detection>> res_batch;
  batch_nms(res_batch, cpu_output_buffer1, img_batch.size(), kOutputSize1, kConfThresh, kNmsThresh);
  for (size_t b = 0; b < img_name_batch.size(); b++) {

    auto& res = res_batch[b];
    cv::Mat img = img_batch[b];

    auto masks = process_mask(&cpu_output_buffer2[b * kOutputSize2], kOutputSize2, res);
    //获取框
    for (size_t i = 0; i < res.size(); i++) {
      
      cv::Mat img_mask = scale_mask(masks[i], img);
      auto color = colors[(int)res[i].class_id % colors.size()];
      auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF);

      cv::Rect r = get_rect(img, res[i].bbox);
      /////添加动态物体的框
      if ((count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[(int)res[i].class_id])))
      {
          RecDepth RecAreaDepth;
          cv::Rect2i DynamicArea(r.x, r.y, r.width, r.height);
          RecAreaDepth.RecArea = DynamicArea;
          
          mvDynamicArea.push_back(DynamicArea);
          mmDetectMap[mClassnames[(int)res[i].class_id]].push_back(RecAreaDepth);
      }

      }
    
}
  // delete[] cpu_output_buffer1;
  // delete[] cpu_output_buffer2;

}
}  //namespace ORB_SLAM


