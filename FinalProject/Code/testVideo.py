from FinalProject.Code.vid_models import *
import cv2
import torch
import torchvision

device = ('cuda' if torch.cuda.is_available() else 'cpu')
cap = cv2.VideoCapture('/media/cs-dl/HD_6TB/Data/Amit/trainval/0af00UcTOSc/50001.mp4')
vid_model = VGG_LSTM(hidden_size=768, n_layers=8, dropt=0.25, bi=True)
vid_model.eval()
vid_model = vid_model.to(device)
frames_per_inference = 5

idx = 0
out = None

while True:

    ret, frame = cap.read()
    if not ret:
        break
    frame = torch.tensor(frame / 255, dtype=torch.float32)
    frame = frame.permute(-1, 0, 1).unsqueeze(0).unsqueeze(0)
    if idx == 0:
        vid = frame
    else:
        vid = torch.cat([vid, frame], dim=1)

    if not idx % (frames_per_inference - 1) and idx:
        vid = vid.to('cuda')
        output, last_hidden = vid_model(vid)
        if out is None:
            out = last_hidden
        else:
            pass
            out = torch.cat([out, last_hidden], dim=1)

            print(out.shape)
        idx = 0
    else:
        idx += 1


print(out.shape)


