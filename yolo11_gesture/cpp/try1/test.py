from hobot_dnn import pyeasy_dnn as dnn

models = dnn.load("/home/sunrise/Documents/HelloBalls-Host/yolo11_gesture/cpp/try1/converted_model.bin")

model=models[0]

print(f"Output tensors: {len(model.output_tensors)}")
for i, tensor in enumerate(model.output_tensors):
    props = tensor.properties
    print(f"Output {i}: shape={props.shape}, quant={props.quanti_type}, dtype={props.data_type}")
