# Cpp test ocr det model

1. Download paddle inference library: `https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html`

2. Set library path in run.sh and run the command

`sh run.sh $PWD/det_db $PWD/det_input.txt`

# Python test ocr det model

1. Install paddle

2. Run the command

`python test.py --model_file det_db/model --params_file det_db/params`


