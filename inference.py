from model import Net
import torch

model_path = "checkpoints/model.pt"

model = Net()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

print (model)