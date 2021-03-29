# build paddle inference lib
cd Paddle/build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DWITH_GPU=OFF \
      -DWITH_AVX=ON \
      -DWITH_DISTRIBUTE=OFF \
      -DWITH_MKLDNN=ON \
      -DON_INFER=ON \
      -DWITH_TESTING=ON \
      -DWITH_INFERENCE_API_TEST=OFF \
      -DWITH_NCCL=OFF \
      -DWITH_PYTHON=ON \
      -DPY_VERSION=3.6 \
      -DWITH_LITE=OFF ..
make -j 12
make -j 12 inference_lib_dist

# build application
rm -rf build && mkdir build && cd build
cmake .. -DPADDLE_LIB=path/to/paddle_inference -DDEMO_NAME=model_test
make -j 12
cd ..
./build/model_test --model_path=det_db --input_path=det_input.txt > hardswish_fc_act_det_results.txt  2>&1
