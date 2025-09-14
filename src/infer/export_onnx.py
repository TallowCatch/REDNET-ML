import torch, onnx
from models.reg_cnn import TinyRegressor

m = TinyRegressor()
m.load_state_dict(torch.load('outputs/reg/best.pt', map_location='cpu'))
m.eval()
dummy = torch.randn(1,3,224,224)
torch.onnx.export(m, dummy, 'outputs/reg/best.onnx', input_names=['input'], output_names=['chl'])
print('Exported to outputs/reg/best.onnx')
