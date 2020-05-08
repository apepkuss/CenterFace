import onnx
import math

# input_size = (480, 640)
# input_size = (720, 1280)
input_size = (1080, 1920)
model = onnx.load_model("/home/nano/workspace/CenterFace/models/onnx/centerface.onnx")
d = model.graph.input[0].type.tensor_type.shape.dim
print(f"[INFO] input dims: {d}")

rate = (
    int(math.ceil(input_size[0] / d[2].dim_value)),
    int(math.ceil(input_size[1] / d[3].dim_value)),
)
print(f"[INFO] scaling rate: {rate}")

d[0].dim_value = 1
d[2].dim_value *= rate[0]
d[3].dim_value *= rate[1]
print(f"[INFO] scaled input dims: {d}")

for output in model.graph.output:
    d = output.type.tensor_type.shape.dim
    print(f"[INFO] output dims: {d}")
    d[0].dim_value = 1
    d[2].dim_value *= rate[0]
    d[3].dim_value *= rate[1]
    print(f"[INFO] scaled output dims: {d}")

onnx.save_model(
    model, "/home/nano/workspace/CenterFace/models/onnx/centerface_1080_1920.onnx"
)

print("Conversion done!")
