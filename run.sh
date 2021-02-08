# 设置
ls

否开启MKL、GPU、TensorRT，如果要使用TensorRT，必须打开GPU
WITH_MKL=ON
WITH_GPU=OFF
USE_TENSORRT=OFF

MODEL_PATH=$1
INPUT_PATH=$2
echo "Model path:${MODEL_PATH}"
echo "Input path:${INPUT_PATH}"

# 按照运行环境设置预测库路径、CUDA库路径、CUDNN库路径、模型路径
#LIB_DIR=/work/autil/_infer_sample/fluid_inference_1.8.4_cpu_avx_mkl/
#LIB_DIR=/work/Paddle/build/paddle_inference_install_dir
LIB_DIR=/home/li/repo/PaddleSlim/demo/mkldnn_quant/fluid_inference
CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
TENSORRT_ROOT_DIR=YOUR_TENSORRT_ROOT_DIR

sh run_impl.sh ${LIB_DIR} model_test ${MODEL_PATH} ${INPUT_PATH} ${WITH_MKL} ${WITH_GPU} ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR} ${USE_TENSORRT} ${TENSORRT_ROOT_DIR}
