import numpy as np
import onnxruntime

obs_input  = np.random.rand(1024,17,8,12).astype(np.float32)  
path_input  = np.random.rand(1024,17,4).astype(np.float32)  
adc_input  = np.random.rand(1024,17,3).astype(np.float32)  
goal_input  = np.random.rand(1024, 4).astype(np.float32)  

obs_input.tofile("in.0.bin")
path_input.tofile("in.1.bin")
adc_input.tofile("in.2.bin")
goal_input.tofile("in.3.bin")

print(obs_input.shape)
print(obs_input.ndim) 
print(obs_input.size) 
print(obs_input.dtype)  

print(path_input.shape) 
print(path_input.ndim)  
print(path_input.size)  
print(path_input.dtype)  

print(adc_input.shape) 
print(adc_input.ndim)  
print(adc_input.size)  
print(adc_input.dtype) 

print(goal_input.shape)
print(goal_input.ndim) 
print(goal_input.size) 
print(goal_input.dtype)

onnx_file="./new_model.onnx"

print("onnx_file:",onnx_file)

ort_sess = onnxruntime.InferenceSession(onnx_file, providers=['AzureExecutionProvider', 'CPUExecutionProvider']) 

print("ort_sess:",ort_sess)

input_name0=ort_sess.get_inputs()[0].name
input_name1=ort_sess.get_inputs()[1].name
input_name2=ort_sess.get_inputs()[2].name
input_name3=ort_sess.get_inputs()[3].name

print("input_name 0:",input_name0)
print("input_name 1:",input_name1)
print("input_name 2:",input_name2)
print("input_name 3:",input_name3)

outputs=ort_sess.get_outputs()

print("outputs:",outputs)

output_names=list(map(lambda output:output.name,outputs))

print("output_names:",output_names)

results=ort_sess.run(output_names, {input_name0: obs_input, input_name1: path_input, input_name2: adc_input, input_name3: goal_input})

print("results:",results)

print("data value:",results[0])

results[0].tofile("out.cpu.0.bin")



print("Exported model has been predict by ONNXRuntime!")
