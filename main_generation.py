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
from Model.phase_two import FontGenerator
from Model.discriminator import Discriminator_wgan
from Model.discriminator import cal_gan_gradient_penalty
from utils.util import *

np.random.seed(1124)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--train', dest='train', help='train mode',
                      action='store_true')
    parser.add_argument('--acgan', dest='acgan', help='acgan mode',
                      action='store_true')

    parser.add_argument('--mode', dest='mode',
                      help='mode', default="gan",
                      type=str)
    parser.add_argument('--model', dest='model',
                      help='model', default="fc",
                      type=str)

    parser.add_argument('--dataset', dest='dataset',
                      help='dataset name', default="mnist",
                      type=str)
    parser.add_argument('--log_dir', dest='log_dir',
                      help='log_dir path', default="../train_log/rnn/",
                      type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='output_dir path', default="../output/",
                      type=str)

    # model hyperparamater
    parser.add_argument('--input_size', dest='input_size',
                      help='input size of model',
                      default=128, type=int)
    parser.add_argument('--hidden_size', dest='hidden_size',
                      help='hidden size',
                      default=256, type=int)
    parser.add_argument('--num_rnn_layers', dest='num_rnn_layers',
                      help='num_rnn_layers',
                      default=1, type=int)
    parser.add_argument('--action_space', dest='action_space',
                      help='number of action space',
                      default=9, type=int)
    parser.add_argument('--width', dest='width',
                      help='width of pictures',
                      default=128, type=int)
    parser.add_argument('--max_step', dest='max_step',
                      help='number of maximum steps',
                      default=5, type=int)
    parser.add_argument('--acgan_num_testing', dest='acgan_num_testing',
                      help='number of testing instances for acgan',
                      default=10, type=int)
    parser.add_argument('--update_critic', dest='update_critic',
                      help='number of times to update critic',
                      default=1, type=int)
    parser.add_argument('--label_dim', dest='label_dim',
                      help='dimension of label embeddings',
                      default=4, type=int)
    parser.add_argument('--data_aug_scale', dest='data_aug_scale',
                      help='scale of data augmentation',
                      default=1, type=int)

    # training arguments
    parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=2000, type=int)
    parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=8, type=int)
    parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=32, type=int)
    parser.add_argument('--num_class', dest='num_class',
                      help='num_class',
                      default=10, type=int)
    parser.add_argument('--early_stop', dest='early_stop',
                      help='patience for early stopping',
                      default=10, type=int)
    parser.add_argument('--cuda', dest='use_cuda', help='whether use CUDA',
                      default=False, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.0001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=str)

    # evaluation
    parser.add_argument('--test', dest='test', help='test mode',
                      action='store_true')

    parser.add_argument('--visualize', dest='visualize', help='visualization mode',
                      action='store_true')


    args = parser.parse_args()
    return args


def train(model_G, model_D, train_dataloader, test_dataloader, args, device):
    # setup optimizer
    #optimizer_G = optim.Adam(model_A.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model_D.parameters(), lr=3e-4, betas=(0.5, 0.999))

    optimizer_G = optim.RMSprop(model_G.parameters(), lr=1e-3)
    #optimizer_D = optim.RMSprop(model_D.parameters(), lr=1e-3)

    criterion = nn.BCELoss()
    test_criterion = nn.MSELoss()

    if args.resume == True:
        pass
        #model_G, model_D, optimizer_G, optimizer_D, args.start_epoch = load_checkpoint(args.checkpoint, model_G, model_D, optimizer_G, optimizer_D)

    f = open('../generation_result.txt', 'w')
    f.write('D_loss,G_loss,D(x),D(G(z))_1,D(G(z))_2,train_mse,test_mse\n')
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        print ('\nepoch = {}'.format(epoch))
        np.random.seed(int(time.time()))

        d_losses, G_losses, mse_losses = AverageMeter(), AverageMeter(), AverageMeter()
        train_pbar = tqdm(total=len(train_dataloader), ncols=100, leave=True)

        model_G.train()
        model_D.train()
        for batch_idx, (img_prototype, img_target, img_source_list, r_label, f_label) in enumerate(train_dataloader):
            img_prototype, img_target, img_source_list = img_prototype.to(device), img_target.to(device), img_source_list.to(device)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            model_D.zero_grad()
            batch_size = img_prototype.size(0)

            real_img = img_target
            real = torch.cat([real_img, real_img], 1)

            real_output = model_D(real)
            D_x = real_output.mean().item()

            # train with fake
            fake_img, _ = model_G(img_prototype, img_source_list)
            fake = torch.cat([real_img, fake_img], 1)

            fake_output = model_D(fake.detach())
            D_G_z1 = fake_output.mean().item()
            errD = -real_output.mean() + fake_output.mean() + cal_gan_gradient_penalty(model_D, real, fake, batch_size, args, device)
            errD.backward()
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            model_G.zero_grad()
            fake_output = model_D(fake)
            errG = -fake_output.mean()
            errG.backward()
            D_G_z2 = fake_output.mean().item()
            optimizer_G.step()

            err_mse = test_criterion(real_img, fake_img)
            mse_losses.update(err_mse.data.item(), batch_size)
            d_losses.update(errD.data.item(), batch_size)
            G_losses.update(errG.data.item(), batch_size)
            train_pbar.update()
            train_pbar.set_postfix({'D_loss':'{:.4f}'.format(d_losses.avg), 'G_loss':'{:.4f}'.format(G_losses.avg),
                                    'D(x)':'{:.4f}'.format(D_x), 'D(G(z))_1':'{:.4f}'.format(D_G_z1), 'D(G(z))_2':'{:.4f}'.format(D_G_z2)})

            if batch_idx == 0:
                out_img = torch.cat([real_img, fake_img], 0)
                vutils.save_image(out_img * 255,
                        '%s/train_samples_epoch_%03d.png' % (args.output_dir, epoch - 1),
                        normalize=False, nrow=batch_size)

        train_pbar.close()

        # testing
        t_mse_losses = AverageMeter()
        test_pbar = tqdm(total=len(test_dataloader), ncols=100, leave=True)

        model_G.eval()
        model_D.eval()
        for batch_idx, (img_prototype, img_target, img_source_list, r_label, f_label) in enumerate(test_dataloader):
            img_prototype, img_target, img_source_list = img_prototype.to(device), img_target.to(device), img_source_list.to(device)
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            with torch.no_grad():
                batch_size = img_prototype.size(0)

                real_img = img_target
                fake_img, _ = model_G(img_prototype, img_source_list)
                err_mse = test_criterion(real_img, fake_img)

            t_mse_losses.update(err_mse.data.item(), batch_size)
            test_pbar.update()
            test_pbar.set_postfix({'mse_loss':'{:.4f}'.format(t_mse_losses.avg)})

            if batch_idx == 0:
                out_img = torch.cat([real_img, fake_img], 0)
                vutils.save_image(out_img * 255,
                        '%s/test_samples_epoch_%03d.png' % (args.output_dir, epoch - 1),
                        normalize=False, nrow=batch_size)

        test_pbar.close()

        # saving
        torch.save(model_G.state_dict(), "../saved_model/stage_2/generator_{}.pkl".format(epoch))
        torch.save(model_D.state_dict(), "../saved_model/stage_2/discriminator_{}.pkl".format(epoch))


        f.write('{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(d_losses.avg, G_losses.avg, D_x, D_G_z1, D_G_z2, mse_losses.avg, t_mse_losses.avg))
        f.flush()





if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build models
    model_G = FontGenerator(encoding_dim=128, device=device).to(device)
    model_D = Discriminator_wgan().to(device)


    if args.train:
        # create dataset loader
        from Dataset.chinese_font_generation import ChineseFontImages
        train_dataset = ChineseFontImages(phase='train')
        test_dataset = ChineseFontImages(phase='test')

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

        train(model_G, model_D, train_dataloader, test_dataloader, args, device)

    if args.test:
        # create dataset loader
        from Dataset.chinese_font_generation import ChineseFontImagesEvaluate
        train_dataset = ChineseFontImagesEvaluate(phase='train')
        test_dataset = ChineseFontImagesEvaluate(phase='test')

        train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=False, num_workers=8, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=8, drop_last=False)

        test(model, train_dataloader, test_dataloader, args, device)
