1+1
import torch
import dill as pickle

model_path = "/home/ubuntu/U-Mamba-Adjustment/data/nets/UXlstmBot-nnUNetPlans_2d-DC_and_CE_loss-w-1-20-20-dill.pth"
model = torch.load(model_path, pickle_module=pickle)

model.eval()

print(model)

dummy_input = torch.randn(1, 5, 512, 512)  # Format: (batch_size, channels, height, width)

with torch.no_grad():
    model.eval()
    output = model(dummy_input)

try:
    with torch.no_grad():
        model.eval()
        output = model(dummy_input)
    print("The model accepted the input.")
except Exception as e:
    print(f"Error occurred: {e}")

if torch.cuda.is_available():
    model = model.cuda()  # Ensure model is on GPU
    dummy_input = dummy_input.to('cuda')  # Move input tensor to GPU

# try in cpu
model = model.cpu()

# Ensure the input is also on CPU
dummy_input = dummy_input.to('cpu')

try:
    with torch.no_grad():
        model.eval()
        output = model(dummy_input)
    print("The model accepted the cpu input.")
except Exception as e:
    print(f"Error occurred: {e}")

onnx_model_path = model_path.replace('.pth', '.onnx')


if torch.cuda.is_available():
    model = model.cuda()  # Ensure model is on GPU
    dummy_input = dummy_input.to('cuda')  # Move input tensor to GPU

torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, do_constant_folding=True, opset_version=15, verbose=True)

torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, verbose=True)
