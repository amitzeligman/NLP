from models import *
import cv2
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

img = cv2.imread('/Users/amitzeligman/Desktop/1.png')
img = to_tensor(img)
img = img.unsqueeze(0).unsqueeze(0)
img = torch.cat([img, img], dim=1)

model = VGG_LSTM(hidden_size=768, n_layers=8, dropt=0.25, bi=True)
model.eval()

out = model(img)

out1 = out[0, ...]
out2 = out[1, ...]
diff = out1 - out2
print(diff.abs().sum().item())
print(out.shape)

