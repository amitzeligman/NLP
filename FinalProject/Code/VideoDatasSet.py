import torch
import torchvision
import pandas as pd
import os
import glob
from math import ceil
import torch.utils.data as data


def collate_fn(batch):
    videos_batch = []
    subs_batch = ''
    for i, item in enumerate(batch):
        videos_batch.append(item['video'])
        subs_batch += item['subtitle']['Text'] #'<Start>' + item['subtitle']['Text']
    videos_batch = torch.cat(videos_batch, dim=0)

    return videos_batch, subs_batch


class VGGDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, subtitles_dir, increase_one_word_every_N_epochs=2, video_crop_size=100, transform=None,
                 crop_lips=False):
        self.data_path = root_dir
        self.subtitles_dir = subtitles_dir
        if crop_lips:
            videos_and_subtitles = self.get_videos_and_subtitles_with_lip_crods()
        else:
            videos_and_subtitles = self.get_videos_and_subtitles()
        self.metadata = pd.DataFrame({'VideoFile': videos_and_subtitles[0], 'SubtitleFile': videos_and_subtitles[1]})
        self.video_crop_size = video_crop_size
        self.transform = transform
        self.sep_line = ' \t' if crop_lips else ' '
        self.increase_one_word_every_N_epochs = increase_one_word_every_N_epochs
        self.increase_one_word_every_N_iters = self.increase_one_word_every_N_epochs * self.__len__()
        self.counter_iterations = 0
        self.num_of_words = 0


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        if not(self.counter_iterations % self.increase_one_word_every_N_iters):
            self.num_of_words += 1

        self.counter_iterations += 1

        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = self.metadata['VideoFile'].iloc[idx]
        subtitle_path = self.metadata['SubtitleFile'].iloc[idx]

        video_obj = torchvision.io.read_video(video_path, start_pts=0, end_pts=None, pts_unit='pts')
        video = video_obj[0]
        sound = video_obj[1]
        video_metadata = video_obj[2]
        subtitle = self.read_txt(subtitle_path, self.sep_line)

        video, subtitle = self.create_video_using_num_of_words(video, sound, video_metadata, subtitle)

        # Crop video in center
        video = self.crop_center(video, self.video_crop_size, self.video_crop_size)

        if self.transform:
            video = self.transform(video)

        # Permute channels to [n_frames, channels, height, width] and normalize to [0, 1]
        video = (video.permute(0, -1, 1, 2) / 255).type(torch.float32)

        sample = {'video': video, 'video_md': video_metadata, 'subtitle': subtitle}

        return sample

    def get_videos_and_subtitles_with_lip_crods(self):
        mp4_list = []
        txt_list = []
        all_dirs = os.listdir(self.data_path)

        for _dir in all_dirs:
            mp4_in_dir = glob.glob('{}/{}/*.mp4'.format(self.data_path, _dir))
            txt_in_dir = [mp4.split('.mp4')[0] + '.txt' for mp4 in mp4_in_dir]
            txt_in_dir = [f.replace(self.data_path, self.subtitles_dir) for f in txt_in_dir]
            txt_in_dir = ['/'.join(f.split('/' or '\\')[:-1] + ['0' + f.split('/' or '\\')[-1][1:]]) for f in txt_in_dir]

            mp4_list = mp4_list + mp4_in_dir
            txt_list = txt_list + txt_in_dir

        return [mp4_list, txt_list]

    def get_videos_and_subtitles(self):
        mp4_list = []
        txt_list = []
        all_dirs = os.listdir(self.data_path)

        for _dir in all_dirs:
            mp4_in_dir = glob.glob('{}/{}/*.mp4'.format(self.data_path, _dir))
            txt_in_dir = [mp4.split('.mp4')[0] + '.txt' for mp4 in mp4_in_dir]

            mp4_list = mp4_list + mp4_in_dir
            txt_list = txt_list + txt_in_dir

        return [mp4_list, txt_list]

    def create_video_using_num_of_words(self, video, sound, video_metadata, subtitle):
        if self.num_of_words > subtitle['Text'].split(' ').__len__():
            num_of_words = subtitle['Text'].__len__()
        else:
            num_of_words = self.num_of_words
        subtitle['Text'] = ' '.join(subtitle['Text'].split(' ')[:num_of_words])

        time_for_each_frame = 1 / video_metadata['video_fps']
        subtitle_timing_df = subtitle['frame_info']
        subtitle_timing_for_current_word_timing = float(subtitle_timing_df.iloc[num_of_words]['END'])
        n_frames = int(ceil(subtitle_timing_for_current_word_timing / time_for_each_frame))
        video = video[:n_frames]

        time_for_each_frame_sound = 1 / video_metadata['audio_fps']
        n_frames_sound = int(ceil(time_for_each_frame_sound / time_for_each_frame))
        sound = sound[:, :n_frames_sound]

        return video, sound, subtitle

    @staticmethod
    def read_txt(sub_file, sep_line):
        y = {}
        frames_info = []
        flag = True
        with open(sub_file, "r") as infile:
            for line in infile:
                if line == '\n':
                    flag = False
                    continue
                if flag:
                    key, value = line.strip().split(':')
                    value = value[2:]
                    y[key] = value
                else:
                    frames_info.append(line.replace(' \n', '').split(sep_line))
        frames_info_df = pd.DataFrame(data=frames_info[1:], columns=frames_info[0])
        y['frame_info'] = frames_info_df
        return y

    @staticmethod
    def crop_center(video, cropx, cropy):
        _, y, x, _ = video.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return video[:, starty:starty + cropy, startx:startx + cropx, :]

