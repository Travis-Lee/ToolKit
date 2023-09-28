import onnx

from onnxsim import simplify

import onnxoptimizer

onnx_model=onnx.load("./origin.onnx")

onnx_model.ir_version = 4

new_model=onnxoptimizer.optimize(onnx_model)

model_simp, check = simplify(new_model)
assert check, "Simplified ONNX model could not be validated"

onnx.save(model_simp, "./mpath.onnx")



