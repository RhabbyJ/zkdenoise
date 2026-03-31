import torch
import json
import numpy as np
from models.network_dncnn import DnCNN

model = DnCNN(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
model.load_state_dict(torch.load('model_zoo/dncnn_15.pth', map_location='cpu'), strict=True)
model.eval()

# Use a small dummy noisy image (you can replace later with real data)
dummy_input = torch.randn(1, 1, 64, 64) * 0.1   # very noisy example

with torch.no_grad():
    output = model(dummy_input)

# Save input and output as JSON (ezkl format)
input_data = {"input": dummy_input.squeeze(0).numpy().tolist()}
output_data = {"output": output.squeeze(0).numpy().tolist()}

with open("input.json", "w") as f:
    json.dump(input_data, f)
with open("output.json", "w") as f:
    json.dump(output_data, f)

print("✅ input.json and output.json created")
