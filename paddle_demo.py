import os
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()

MODEL_NAME = "inference_model"
MODEL_FILE = "model.pdmodel"
PARAMS_FILE = "model.pdiparams"

def main(argv=None):
    # Load model
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    model_dir = "./paddle_model/" + MODEL_NAME
    print("model dir:", model_dir)
    if len(MODEL_FILE) == 0 and len(PARAMS_FILE) == 0:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(model_dir, exe)
    else:
        [program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             model_dir,
             exe,
             model_filename=MODEL_FILE,
             params_filename=PARAMS_FILE)
    print("--- feed_target_names ---")
    print(feed_target_names)
    print("--- fetch_targets ---")
    print(fetch_targets)
    # Preprocess

    input_tensor1 = np.ones([1024,4]).astype(np.float32)
    input_tensor2 = np.ones([1024,17,3]).astype(np.float32)
    input_tensor3 = np.ones([1024,17,4]).astype(np.float32)
    input_tensor4 = np.ones([1024,17,8,12]).astype(np.float32)

    input_tensors = {"x2paddle_goal_input":input_tensor1,"x2paddle_adc_input":input_tensor2,"x2paddle_path_input":input_tensor3,"x2paddle_obs_input":input_tensor4}
    # Inference
    output_tensors = exe.run(program=program,
                             feed=input_tensors,
                             fetch_list=fetch_targets,
                             return_numpy=False)
    # Postprocess
    for output_tensor in output_tensors:
        output_data = np.array(output_tensor)
        print(output_data.shape)
        print(output_data)
    print("Done.")


if __name__ == '__main__':
    main()

  
