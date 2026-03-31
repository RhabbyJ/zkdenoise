import torch
from models.network_dncnn import DnCNN

# Load the exact same DnCNN model you have in model_zoo
model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
model.load_state_dict(torch.load('model_zoo/dncnn_15.pth', map_location='cpu'), strict=True)

model.eval()

# Fixed input size (grayscale 64x64 — perfect for first proof)
dummy_input = torch.randn(1, 1, 64, 64)

torch.onnx.export(
    model,
    dummy_input,
    "dncnn_15.onnx",
    opset_version=17,
    input_names=['input'],
    output_names=['output']
)

print("✅ ONNX file exported as dncnn_15.onnx")
