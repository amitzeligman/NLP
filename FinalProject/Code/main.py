import torch
from FinalProject.Code.VideoDataSet import VGGDataset, collate_fn
from torch.utils.data import DataLoader


data_dir = '/media/cs-dl/HD_6TB/Data/Amit/trainval'
data_set = VGGDataset(data_dir)


loader = DataLoader(data_set,
                    collate_fn=collate_fn,
                        batch_size=1,
                        shuffle=False,
                        num_workers=8,
                        pin_memory=True)


for vid, sent in loader:

    print(vid.shape, sent)



