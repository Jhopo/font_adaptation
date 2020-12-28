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
#PROTOTYPE_FONTFILE = 'DFT_B5.ttc'#'DFT_M5.ttc'
SOURCE = 'small'

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
    filenames = sorted(glob.glob(os.path.join(DATA_DIR, SOURCE, '*.ttf')) + glob.glob(os.path.join(DATA_DIR, SOURCE, '*.ttc')))
    for filename in filenames:
        font = ImageFont.truetype(filename, size=char_size)
        all_font.append(font)

        if 'ttf' in filename:
            font_name = filename.split('/')[-1].replace('.ttf', '')
        if 'ttc' in filename:
            font_name = filename.split('/')[-1].replace('.ttc', '')
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

    with open(join(DATA_DIR, SOURCE, 'char_dictionary.pkl'), 'wb') as f:
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
    with open(join(DATA_DIR, SOURCE, 'char_dictionary.pkl'), 'rb') as f:
        char_dict = pickle.load(f)

    #with open(join(DATA_DIR, 'prototype', 'char_dictionary.pkl'), 'rb') as f:
    #    proto_dict = pickle.load(f)

    return char_dict#, proto_dict


load_global_charset()
charset = CN_CHARSET
prototype_font = ImageFont.truetype(os.path.join(DATA_DIR, SOURCE, PROTOTYPE_FONTFILE), size=216)
all_font, font_list, font_dict = load_ttf_list()

#blank_hashing = create_repeating_images(charset, all_font, font_list)
#create_dictionary(blank_hashing, charset, prototype_font, all_font, font_list)
char_dict = load_dictionary()

if SOURCE == 'small':
    indices = np.array([0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 1, 19, 28, 29])

    all_font = list(np.array(all_font)[indices])
    font_list = list(np.array(font_list)[indices])

if SOURCE == 'big':
    indices = np.array([121,  86, 157, 126,  33,  15, 165, 164, 143,  65,  57,  27,  75,
           169, 118,  34, 120,  41,  51,  22, 105,  94, 115, 167,  70, 153,
            61,  73, 149,  97, 154,   2,  36,  93, 114,  11,   6, 117,  30,
           128, 111,  21,  98,  59,  50, 103, 107, 147,  90, 134,  60,  24,
            95,  23, 125, 168,  25,  39,  49,  20,   5,  28, 119, 110,  77,
            82,   9, 104,  54,  18, 112,  83, 158,  38,  69,   3,  10, 148,
            89, 135, 159,  16,  72,  48, 102,  44,  53,  78,  31, 137, 141,
           129,  14,  52, 108,  26, 123,  91,   7, 155,   4, 113, 146,  87,
            17,  47, 133, 136,  63, 130,  43, 138,  99, 106,  80,  68,  12,
            62, 139,  76, 144,  42, 152,   1,  55, 156,  13,  29,  32, 127,
           122, 131,  56,  66, 140,  67,  37, 150, 151,  84, 162, 142,  74,
             0,  96,  19,  64, 132, 109, 124,  92,   8,  85, 101,  35,  81,
           163,  79, 160,  40, 170,  88, 116, 166,  58,  45,  46, 100, 145,
            71, 161])

    all_font = list(np.array(all_font)[indices])
    font_list = list(np.array(font_list)[indices])

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
    def __init__(self, phase, num_sample=10, canvas_size=256, x_offset=20, y_offset=20):
        if SOURCE ==  'small':
            self.train_num = 26
            self.train_sample = 72000
            self.test_sample = 6000
        elif SOURCE ==  'big':
            self.train_num = 161
            self.train_sample = 180000
            self.test_sample = 18000

        if phase == 'train':
            self.all_font = all_font[:self.train_num]
            self.font_list = font_list[:self.train_num]
            self.len = self.train_sample
            self.num_sample = num_sample
        if phase == 'test':
            self.all_font = all_font[self.train_num:]
            self.font_list = font_list[self.train_num:]
            self.len = self.test_sample
            self.num_sample = num_sample

        self.prototype_font = prototype_font
        self.prototype_name = PROTOTYPE_FONTFILE.replace('.ttc', '').replace('.ttf', '')
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
    def __init__(self, phase, num_sample=10, canvas_size=256, x_offset=20, y_offset=20):
        if SOURCE ==  'small':
            self.train_num = 26
            self.train_sample = 72000
            self.test_sample = 6000
        elif SOURCE ==  'big':
            self.train_num = 161
            self.train_sample = 180000
            self.test_sample = 18000

        if phase == 'train':
            self.all_font = all_font[:self.train_num]
            self.font_list = font_list[:self.train_num]
            self.len = 11
            self.num_sample = num_sample
        if phase == 'test':
            self.all_font = all_font[self.train_num:]
            self.font_list = font_list[self.train_num:]
            self.len = 11
            self.num_sample = num_sample

        self.prototype_font = prototype_font
        self.prototype_name = PROTOTYPE_FONTFILE.replace('.ttc', '').replace('.ttf', '')
        self.prototype_id = font_dict[self.prototype_name]
        self.font_dict = font_dict
        self.char_dict = char_dict

        self.canvas_size = canvas_size
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.num_font = len(self.font_list)
        self.transform = transforms.Compose([
                         transforms.ToTensor()])


        self.source_font_id1 = np.random.randint(0, self.num_font)
        self.source_font1 = self.all_font[self.source_font_id1]
        self.source_font_name1 = self.font_list[self.source_font_id1]

        self.source_font_id2 = np.random.randint(0, self.num_font)
        self.source_font2 = self.all_font[self.source_font_id2]
        self.source_font_name2 = self.font_list[self.source_font_id2]

        while True:
            self.target_char_id = np.random.randint(0, len(charset))
            self.target_c = charset[self.target_char_id]
            if self.target_c in self.char_dict[self.source_font_name1] and self.target_c in self.char_dict[self.prototype_name]:
                break



    def __getitem__(self, index):
        # get image
        img_prototype, img_target1, img_target2, img_source_list = self.sample_character_from_font(index, self.target_c, self.source_font_id1, self.source_font1, self.source_font_name1, self.source_font_id2, self.source_font2, self.source_font_name2)

        if self.transform is not None:
            img_prototype = self.transform(img_prototype)
            img_target1 = self.transform(img_target1)
            img_target2 = self.transform(img_target2)
            img_source_list = [self.transform(img) for img in img_source_list]

        img_source_list = torch.cat(img_source_list, 0)

        real_label = 1
        fake_label = 0

        return img_prototype, img_target1, img_target2, img_source_list, torch.tensor(real_label).float(), torch.tensor(fake_label).float()


    def __len__(self):
        return self.len


    def sample_character_from_font(self, index, target_c, source_font_id1, source_font1, source_font_name1, source_font_id2, source_font2, source_font_name2):
        img_prototype = draw_single_char(target_c, self.prototype_font, self.canvas_size, self.x_offset, self.y_offset)
        img_target1 = draw_single_char(target_c, source_font1, self.canvas_size, self.x_offset, self.y_offset)
        img_target2 = draw_single_char(target_c, source_font2, self.canvas_size, self.x_offset, self.y_offset)

        def sampling(target_c, source_font_name, source_font):
            img_source_list = []
            while True:
                char_id = np.random.randint(0, len(charset))
                c = charset[char_id]
                if c in self.char_dict[source_font_name] and c != target_c:
                    img = draw_single_char(c, source_font, self.canvas_size, self.x_offset, self.y_offset)
                    img_source_list.append(img)

                if len(img_source_list) >= self.num_sample:
                    break

            return img_source_list

        img_source_list1 = sampling(target_c, self.source_font_name1, self.source_font1)
        img_source_list2 = sampling(target_c, self.source_font_name2, self.source_font2)

        img_source_list = img_source_list1[index:] + img_source_list2[:index]


        return img_prototype, img_target1, img_target2, img_source_list



if __name__ == '__main__':
    '''
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
    '''

    train_dataset = ChineseFontImagesEvaluate(phase='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    test_dataset = ChineseFontImagesEvaluate(phase='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    for batch_idx, (img_prototype, img_target1, img_target2, img_source_list, r_label, f_label) in enumerate(train_loader):
        #for i in img:
        #    print (i.tolist())
        #print (img)
        print (batch_idx)
        print (img_prototype.shape)
        print (img_target1.shape)
        print (img_target2.shape)
        print (img_source_list.shape)
        print (r_label.shape)
        print (f_label.shape)
        #break
