import argparse
import time
import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

def set_config(args):
    config = AnalysisConfig(args.model_file, args.params_file)
    config.disable_gpu()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.enable_mkldnn()
    config.enable_profile()
    config.switch_ir_optim(True)
    return config

def main():
    args = parse_args()

    # set AnalysisConfig
    config = set_config(args)

    # create PaddlePredictor
    predictor = create_paddle_predictor(config)

    # set input
    n = 1
    c = 3
    h = 960
    w = 576
    fake_input = []
    for line in open('det_input.txt'):
        fake_input.append(float(line))
    fake_input = np.array(fake_input).astype('float32')
    fake_input.shape = [n, c, h, w]
    
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])
    input_tensor.reshape([n, c, h, w])
    input_tensor.copy_from_cpu(fake_input)

    # run predictor
    warmup = 10
    repeat = 10
    for i in range(warmup):
        predictor.zero_copy_run()
    start = time.time()
    for i in range(repeat):
        predictor.zero_copy_run()
    end = time.time()
    avg_time = (end - start) / repeat
    print('avg time:' + str(avg_time) + "s")

    # get output
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    output_data = output_tensor.copy_to_cpu()

if __name__ == "__main__":
    main()  
