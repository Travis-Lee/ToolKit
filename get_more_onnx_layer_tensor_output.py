import numpy as np
import onnxruntime
import onnx 
from onnx import helper, TensorProto, GraphProto  

model = onnx.load("./mpath.onnx")
graph = model.graph
graph.output.append(onnx.helper.make_tensor_value_info("/_expert_layer/Concat_output_0", TensorProto.FLOAT, [1027,17,32]))
new_model = onnx.helper.make_model(graph)
onnx.checker.check_model(new_model)
onnx.save(new_model, "new_model.onnx")
