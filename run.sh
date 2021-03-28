#wget https://paddle-inference-lib.bj.bcebos.com/2.0.0-cpu-avx-mkl/paddle_inference.tgz
#tar -xzvf paddle_inference.tgz
rm -rf build && mkdir build && cd build
cmake .. -DPADDLE_LIB=/home/li/repo/Paddle/build/paddle_inference_install_dir -DDEMO_NAME=model_test
make -j 12
cd ..
./build/model_test --model_path=det_db --input_path=det_input.txt
