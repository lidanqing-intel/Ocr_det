Download paddle_inference lib, compile the application and run the application
```
wget https://paddle-inference-lib.bj.bcebos.com/2.0.0-cpu-avx-mkl/paddle_inference.tgz
tar -xzvf paddle_inference.tgz
mkdir build && cd build
cmake .. -DPADDLE_LIB=$HOME/repo/Issue_28554/test_ocr_det/paddle_inference -DDEMO_NAME=model_test
make -j 12
cd ..
./build/model_test --model_path=det_db --input_path=det_input.txt
```
