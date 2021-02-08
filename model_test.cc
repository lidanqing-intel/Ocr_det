#include <gflags/gflags.h>
#include <glog/logging.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include "paddle/include/paddle_inference_api.h"

DEFINE_string(model_path, "", "Path of the inference model.");
DEFINE_string(input_path, "", "Path of the input file.");

namespace paddle {
using paddle::AnalysisConfig;

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void PrepareConfig(AnalysisConfig *config, int threads) {
  config->SetModel(FLAGS_model_path + "/model", FLAGS_model_path + "/params");
  config->DisableGpu();
  // We use ZeroCopyTensor here, so we set config->SwitchUseFeedFetchOps(false)
  config->SwitchUseFeedFetchOps(false);
  config->SetCpuMathLibraryNumThreads(threads);
  config->EnableProfile();
  config->EnableMKLDNN();
  config->SwitchIrOptim(true);
}

bool test_map_cnn(int threads, int warmup, int repeat) {
  AnalysisConfig config;
  PrepareConfig(&config, threads);
  auto predictor = CreatePaddlePredictor(config);

  // prepare inputs
  int n = 1, c = 3, h = 960, w = 576;
  int input_num = n * c * h * w;
  float *input = new float[input_num];
  if (!FLAGS_input_path.empty()) {
    std::ifstream ifs(FLAGS_input_path);
    if (!ifs.is_open()) {
      std::cerr << "open file error" << std::endl;
    }
    for (int i = 0; i < input_num; i++) {
      ifs >> input[i];
    }
  } else  {
    for (int i = 0; i < input_num; i++) {
      input[i] = i / 1234.0;
    }
  }
  LOG(INFO) << "input[0]:" << input[0];
  LOG(INFO) << "input[-1]:" << input[input_num -1] << "\n";


  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({n, c, h, w});
  input_t->copy_from_cpu(input);
  delete[] input;

  // run
  for (int i = 0; i < warmup; i++) {
    CHECK(predictor->ZeroCopyRun());
  }

  auto start = time(); 
  for (int i = 0; i < warmup; i++) {
    CHECK(predictor->ZeroCopyRun());
  }
  auto end = time();
  LOG(INFO) << "Threads:" << threads << ", average cost:" 
    << time_diff(start, end) / static_cast<float>(repeat) 
    << "ms\n";

  // get the output
  auto output_names = predictor->GetOutputNames();
  for (auto& name : output_names) {
    LOG(INFO) << "Output " << name;
    auto output_t = predictor->GetOutputTensor(name);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 
        1, std::multiplies<int>());

    std::vector<float> out_data;
    out_data.resize(out_num);
    output_t->copy_to_cpu(out_data.data());

    for (size_t j = 0; j < 10; ++j) {
      LOG(INFO) << "output[" << j << "]: " << out_data[j];
    }
  }
  return true;
}
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::cout << "model path:" << FLAGS_model_path << std::endl;
  std::cout << "input path:" << FLAGS_input_path << std::endl;
  int threads = 1;
  int warmup = 10;
  int repeat = 10;
  paddle::test_map_cnn(threads, warmup, repeat);
  return 0;
}
