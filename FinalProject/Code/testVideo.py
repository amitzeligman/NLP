from models import *
import cv2
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

img = cv2.imread('/Users/amitzeligman/Desktop/Screen Shot 2020-02-04 at 13.44.25.png')
#img = torch.from_numpy(img)
img = torch.tensor(img)
img = img.permute(-1, 0, 1).unsqueeze(0).unsqueeze(0)
img = torch.cat([img, img], dim=1)

model = VGG_LSTM(hidden_size=768, n_layers=8, dropt=0.25, bi=True)
model.eval()

out = model(img)

out1 = out[0, ...]
out2 = out[1, ...]
diff = out1 - out2
print(diff.abs().sum().item())
print(out.shape)

