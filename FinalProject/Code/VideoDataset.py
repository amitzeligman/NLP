import torch
import torchvision
import pandas as pd
import os
import glob
import torch.utils.data as data

def collate_fn(batch):
    videos_batch = []
    subs_batch = ''
    for i, item in enumerate(batch):
        videos_batch.append(item['video'])
        subs_batch += '<Start>' + item['subtitle']['Text']
    videos_batch = torch.cat(videos_batch, dim=0)

    return videos_batch, subs_batch



class VGGDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        videos_and_subtitles = self.get_videos_and_subtitles(root_dir)
        self.metadata = pd.DataFrame({'VideoFile': videos_and_subtitles[0], 'SubtitleFile': videos_and_subtitles[1]})
        self.data_path = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = self.metadata['VideoFile'].iloc[idx]
        subtitle_path = self.metadata['SubtitleFile'].iloc[idx]

        video_obj = torchvision.io.read_video(video_path, start_pts=0, end_pts=None, pts_unit='pts')
        video = video_obj[0]
        video_metadata = video_obj[2]
        subtitle = self.read_txt(subtitle_path)

        if self.transform:
            video = self.transform(video)

        sample = {'video': video, 'video_md': video_metadata, 'subtitle': subtitle}

        return sample


    def read_txt(self, sub_file):
        y = {}
        with open(sub_file, "r") as infile:
            for line in infile:
                key, value = line.strip().split(':')
                value = value[2:]
                y[key] = (value)
        return y

    def get_videos_and_subtitles(self, data_path):
        mp4_list = []
        txt_list = []
        all_dirs = os.listdir('./data')

        for dir in all_dirs:
            mp4_in_dir = glob.glob('{}/{}/*.mp4'.format(data_path, dir))
            txt_in_dir = [mp4.split('.mp4')[0] + '.txt' for mp4 in mp4_in_dir]

            mp4_list = mp4_list + mp4_in_dir
            txt_list = txt_list + txt_in_dir

        return [mp4_list, txt_list]




vid = VGGDataset('./data')
vid[0]
