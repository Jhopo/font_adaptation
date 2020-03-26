import sys

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torch.nn.utils.weight_norm as weightNorm

LAMBDA = 100 # Gradient penalty lambda hyperparameter
def cal_gan_gradient_penalty(netD, real_data, fake_data, batch_size, args, device):
    #print real_data.size()
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def cal_acgan_gradient_penalty(netD, real_data, fake_data, batch_size, args, device):
    #print real_data.size()
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class TReLU(nn.Module):
    def __init__(self):
            super(TReLU, self).__init__()
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


class Discriminator_wgan(nn.Module):
    def __init__(self):
        super(Discriminator_wgan, self).__init__()

        self.conv0 = weightNorm(nn.Conv2d(6, 16, 5, 2, 2))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 1, 5, 2, 2))

        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()

        self.fc = nn.Linear(4, 1)


    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(batch_size, -1)

        x = self.fc(x)
        
        return x


class Discriminator_ac_wgan(nn.Module):
    def __init__(self, args):
        super(Discriminator_ac_wgan, self).__init__()

        self.conv0 = weightNorm(nn.Conv2d(3, 16, 5, 2, 2))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 1, 5, 2, 2))

        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()

        self.aux_layer = nn.Linear(128 * 8 * 8, args.num_class)


    def forward(self, x):
        x = self.conv0(x)  # shape = (B, 16, 64, 64)
        x = self.relu0(x)

        x = self.conv1(x)  # shape = (B, 32, 32, 32)
        x = self.relu1(x)

        x = self.conv2(x)  # shape = (B, 64, 16, 16)
        x = self.relu2(x)

        x = self.conv3(x)  # shape = (B, 128, 8, 8)
        x = self.relu3(x)

        conv_out = self.conv4(x)  # shape = (B, 1, 4, 4)
        conv_out = F.avg_pool2d(conv_out, 4)  # shape = (B, 1, 1, 1)

        validity = conv_out.view(-1, 1)
        label = self.aux_layer(x.view(-1, 128 * 8 * 8))

        return validity, label

class Discriminator_gan(nn.Module):
    def __init__(self, args):
        super(Discriminator_gan, self).__init__()

        self.conv0 = weightNorm(nn.Conv2d(3, 16, 5, 2, 2))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 1, 5, 2, 2))
        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()


    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 1)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    class Args():
        def __init__(self):
            self.acgan = True
            self.num_class = 10

    args = Args()
    device = torch.device("cpu")

    model_D = Discriminator_wgan()

    input = torch.randn(8, 6, 256, 256)
    validity = model_D(input)
    print (validity.shape)  # (8, 11)
