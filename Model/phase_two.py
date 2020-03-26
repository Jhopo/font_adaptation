import os
import sys
import time, pdb
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from Model.encoder import StyleLearner
from Model.unet_generator import UNetGenerator
from utils.util import *


class FontGenerator(nn.Module):
    def __init__(self, encoding_dim, device):
        super(FontGenerator, self).__init__()
        self.encoding_dim = encoding_dim
        self.device = device

        StyleLearner_model = StyleLearner(encoding_dim=encoding_dim)
        StyleLearner_model.encoder.load_state_dict(torch.load('../saved_model/encoder.pkl'))
        self.encoder = StyleLearner_model.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.generator = UNetGenerator()
        #self.discriminator = Discriminator_wgan()

    def forward(self, img_prototype, img_source_list):
        batch_size = img_prototype.shape[0]
        num_sample = int(img_source_list.shape[1] / 3)
        size = img_prototype.shape[2]

        source_encoding = self.encoder(img_source_list.view(batch_size*num_sample, 3, size, size))
        source_encoding = source_encoding.view(batch_size, num_sample, -1)
        source_encoding = torch.mean(source_encoding, dim=1)

        img_fake, encoded_prototype = self.generator(img_prototype, source_encoding)

        return img_fake, encoded_prototype


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FontGenerator(encoding_dim=128, device=device)

    img_prototype = torch.randn(8, 3, 256, 256)
    img_target = torch.randn(8, 3, 256, 256)
    img_source_list = torch.randn(8, 9, 256, 256)

    output_1, output_2 = model(img_prototype, img_target, img_source_list)
    print (output_1.shape, output_2.shape)
