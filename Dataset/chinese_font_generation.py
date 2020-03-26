import numpy as np
from os.path import join
import pickle
import glob
import os
import sys
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections


DATA_DIR = '/media/jhopo/DATA/NTU/MSLAB/Font Generation/dataset/'
DEFAULT_CHARSET = "/media/jhopo/DATA/NTU/MSLAB/Font Generation/src/Dataset/charset/cjk.json"
PROTOTYPE_FONTFILE = 'DFKingGothicTC1-Regular_TTF_TW.ttf'#'DFPT_ZK7_TTF_TW.ttf'

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None


def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
    return img



def load_ttf_list(char_size=216):
    all_font = []
    font_list, font_dict = [], {}
    filenames = sorted(glob.glob(os.path.join(DATA_DIR, 'small', 'unpack', '*.ttf')))
    for filename in filenames:
        if 'TW' in filename:
            font = ImageFont.truetype(filename, size=char_size)
            all_font.append(font)

            font_name = filename.split('/')[-1].replace('.ttf', '')
            font_list.append(font_name)
            font_dict[font_name] = len(font_dict)

    return all_font, font_list, font_dict


def create_repeating_images(charset, all_font, font_list, canvas_size=256, x_offset=20, y_offset=20):
    all_hashing = []
    for i, font in enumerate(all_font):
        _charset = charset[:]
        np.random.shuffle(_charset)
        sample = _charset[:3000]
        hash_count = collections.defaultdict(int)
        for c in sample:
            img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
            hash_count[hash(img.tobytes())] += 1
        recurring_hashes = filter(lambda d: d[1] > 10, hash_count.items())

        all_hashing.append([rh[0] for rh in recurring_hashes])
        break

    return all_hashing[0][0]


def create_dictionary(blank_hashing, charset, prototype_font, all_font, font_list, canvas_size=256, x_offset=20, y_offset=20):
    char_dict = {}
    for i, font in enumerate(all_font):
        char_dict[font_list[i]] = set()
        for c in charset:
            img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
            img_hash = hash(img.tobytes())
            if img_hash != blank_hashing:
                char_dict[font_list[i]].add(c)

    with open(join(DATA_DIR, 'small', 'char_dictionary.pkl'), 'wb') as f:
        pickle.dump(char_dict, f)
    '''
    proto_dict = set()
    for c in charset:
        img = draw_single_char(c, prototype_font, canvas_size, x_offset, y_offset)
        img_hash = hash(img.tobytes())
        if img_hash != blank_hashing:
            proto_dict.add(c)

    print (len(proto_dict))
    char_dict = {'proto': proto_dict}
    with open(join(DATA_DIR, 'prototype', 'char_dictionary.pkl'), 'wb') as f:
        pickle.dump(char_dict, f)
    '''


def load_dictionary():
    with open(join(DATA_DIR, 'small', 'char_dictionary.pkl'), 'rb') as f:
        char_dict = pickle.load(f)

    #with open(join(DATA_DIR, 'prototype', 'char_dictionary.pkl'), 'rb') as f:
    #    proto_dict = pickle.load(f)

    return char_dict#, proto_dict


load_global_charset()
charset = CN_CHARSET
prototype_font = ImageFont.truetype(os.path.join(DATA_DIR, 'small', 'unpack', PROTOTYPE_FONTFILE), size=216)
all_font, font_list, font_dict = load_ttf_list()

#blank_hashing = create_repeating_images(charset, all_font, font_list)
#create_dictionary(blank_hashing, charset, prototype_font, all_font, font_list)
char_dict = load_dictionary()

'''
print ('total', len(charset))
cnt_list = []
for i, font in enumerate(all_font):
    font_name = font_list[i]
    cnt = 0
    for c in charset:
        if c in char_dict[font_name]:
            cnt += 1
    cnt_list.append(cnt)
'''
'''
for i, font in enumerate(all_font):
    font_name = font_list[i]
    for ic, c in enumerate(charset[:100]):
        if c in char_dict[font_name]:
            img = draw_single_char(c, font, 256, 20, 20)
            img.save('../../dataset/test/{}_{}.jpg'.format(font_name, ic), "JPEG")
'''
'''
for font_name in ['DFPT_K7_TTF_TW', 'DFPT_S5_TTF_TW']:
    font_id = font_dict[font_name]
    font = all_font[font_id]
    for ic, c in enumerate(charset[2000:2100]):
        if c in char_dict[font_name]:
            img = draw_single_char(c, font, 256, 20, 20)
            img.save('../dataset/test/{}_{}.jpg'.format(font_name, ic), "JPEG")
'''

class ChineseFontImages(Dataset):
    def __init__(self, phase, num_sample=5, canvas_size=256, x_offset=20, y_offset=20):
        if phase == 'train':
            self.all_font = all_font[:25]
            self.font_list = font_list[:25]
            self.len = 72000
            self.num_sample = num_sample
        if phase == 'test':
            self.all_font = all_font[25:]
            self.font_list = font_list[25:]
            self.len = 6000
            self.num_sample = num_sample

        self.prototype_font = prototype_font
        self.prototype_name = PROTOTYPE_FONTFILE.replace('.ttf', '')
        self.prototype_id = font_dict[self.prototype_name]
        self.font_dict = font_dict
        self.char_dict = char_dict

        self.canvas_size = canvas_size
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.num_font = len(self.font_list)
        self.transform = transforms.Compose([
                         transforms.ToTensor()])



    def __getitem__(self, index):
        # get image
        img_prototype, img_target, img_source_list = self.sample_training_pairs()

        if self.transform is not None:
            img_prototype = self.transform(img_prototype)
            img_target = self.transform(img_target)
            img_source_list = [self.transform(img) for img in img_source_list]

        img_source_list = torch.cat(img_source_list, 0)

        real_label = 1
        fake_label = 0

        return img_prototype, img_target, img_source_list, torch.tensor(real_label).float(), torch.tensor(fake_label).float()


    def __len__(self):
        return self.len


    def sample_training_pairs(self):
        source_font_id = np.random.randint(0, self.num_font)
        source_font = self.all_font[source_font_id]
        source_font_name = self.font_list[source_font_id]

        while True:
            target_char_id = np.random.randint(0, len(charset))
            target_c = charset[target_char_id]
            if target_c in self.char_dict[source_font_name] and target_c in self.char_dict[self.prototype_name]:
                break

        img_prototype, img_target, img_source_list = self.sample_character_from_font(target_c, source_font_id, source_font, source_font_name)

        return img_prototype, img_target, img_source_list


    def sample_character_from_font(self, target_c, source_font_id, source_font, source_font_name):
        img_prototype = draw_single_char(target_c, self.prototype_font, self.canvas_size, self.x_offset, self.y_offset)
        img_target = draw_single_char(target_c, source_font, self.canvas_size, self.x_offset, self.y_offset)

        img_source_list = []
        while True:
            char_id = np.random.randint(0, len(charset))
            c = charset[char_id]
            if c in self.char_dict[source_font_name] and c != target_c:
                img = draw_single_char(c, source_font, self.canvas_size, self.x_offset, self.y_offset)
                img_source_list.append(img)

            if len(img_source_list) >= self.num_sample:
                break

        return img_prototype, img_target, img_source_list


class ChineseFontImagesEvaluate(Dataset):
    def __init__(self, phase, canvas_size=256, x_offset=20, y_offset=20):
        self.num_sample = 200

        if phase == 'train':
            self.all_font = all_font[:25]
            self.font_list = font_list[:25]
            self.len = 25 * self.num_sample
        if phase == 'test':
            self.all_font = all_font[25:]
            self.font_list = font_list[25:]
            self.len = 5 * self.num_sample

        self.font_dict = font_dict

        self.canvas_size = canvas_size
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.num_font = len(self.font_list)
        self.transform = transforms.Compose([
                         transforms.ToTensor()])


    def __getitem__(self, index):
        # get image
        font_id = int(index / self.num_sample)
        img, char = self.sample_character_from_font(font_id)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(font_id).long()


    def __len__(self):
        return self.len


    def sample_character_from_font(self, font_id):
        font = self.all_font[font_id]
        font_name = self.font_list[font_id]

        while True:
            char_id = np.random.randint(0, len(charset))
            c = charset[char_id]
            if c in char_dict[font_name]:
                img = draw_single_char(c, font, self.canvas_size, self.x_offset, self.y_offset)
                break

        return img, c



if __name__ == '__main__':
    train_dataset = ChineseFontImages(phase='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_dataset = ChineseFontImages(phase='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    for batch_idx, (img_prototype, img_target, img_source_list, r_label, f_label) in enumerate(train_loader):
        #for i in img:
        #    print (i.tolist())
        #print (img)
        print (img_prototype.shape)
        print (img_target.shape)
        print (img_source_list.shape)
        print (r_label.shape)
        print (f_label.shape)
        break
