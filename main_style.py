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


def train(model, train_dataloader, test_dataloader, args, device):
    # setup optimizer
    #optimizer_agent = optim.Adam(model_A.parameters(), lr=1e-4, betas=(0.5, 0.999))
    #optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    #model.encoder.load_state_dict(torch.load('../saved_model/encoder_8.pkl'))

    f = open('../result.txt', 'w')
    f.write('training_loss,testing_loss\n')
    best_test = 1000
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        print ('\nepoch = {}'.format(epoch))
        np.random.seed(int(time.time()))

        d_losses = AverageMeter()
        train_pbar = tqdm(total=len(train_dataloader), ncols=100, leave=True)

        model.train()
        for batch_idx, (img_a1, img_a2, img_b, r_label, f_label) in enumerate(train_dataloader):
            img_a1, img_a2, img_b, r_label, f_label = img_a1.to(device), img_a2.to(device), img_b.to(device), r_label.to(device), f_label.to(device)

            model.zero_grad()
            batch_size = img_a1.size(0)

            real_valid = model(img_a1, img_a2)
            errD_real = criterion(real_valid, r_label)
            errD_real.backward()
            R_x = real_valid.mean().item()

            # train with fake
            fake_valid = model(img_a1, img_b)
            errD_fake = criterion(fake_valid, f_label)
            errD_fake.backward()
            F_x = fake_valid.mean().item()

            optimizer.step()
            errD = errD_real + errD_fake


            d_losses.update(errD.data.item(), img_a1.shape[0])

            train_pbar.update()
            train_pbar.set_postfix({'D_loss':'{:.4f}'.format(d_losses.avg),
                                    'M(x, x)':'{:.4f}'.format(R_x), 'M(x, y)':'{:.4f}'.format(F_x)})
        train_pbar.close()

        # testing
        t_d_losses = AverageMeter()
        test_pbar = tqdm(total=len(test_dataloader), ncols=100, leave=True)

        model.eval()
        for batch_idx, (img_a1, img_a2, img_b, r_label, f_label) in enumerate(test_dataloader):
            img_a1, img_a2, img_b, r_label, f_label = img_a1.to(device), img_a2.to(device), img_b.to(device), r_label.to(device), f_label.to(device)

            with torch.no_grad():
                batch_size = img_a1.size(0)

                real_valid = model(img_a1, img_a2)
                errD_real = criterion(real_valid, r_label)
                R_x = real_valid.mean().item()

                # train with fake
                fake_valid = model(img_a1, img_b)
                errD_fake = criterion(fake_valid, f_label)
                F_x = fake_valid.mean().item()

                errD = errD_real + errD_fake

            t_d_losses.update(errD.data.item(), img_a1.shape[0])

            test_pbar.update()
            test_pbar.set_postfix({'D_loss':'{:.4f}'.format(t_d_losses.avg),
                                    'M(x, x)':'{:.4f}'.format(R_x), 'M(x, y)':'{:.4f}'.format(F_x)})
        test_pbar.close()

        if t_d_losses.avg < best_test:
            best_test = t_d_losses.avg

        torch.save(model.encoder.state_dict(), "../saved_model/encoder_{}.pkl".format(epoch))

        f.write('{:.4f},{:.4f}\n'.format(d_losses.avg, t_d_losses.avg))
        f.flush()


def test(model, train_dataloader, test_dataloader, args, device):
    np.random.seed(1124)

    model = model.encoder
    model.load_state_dict(torch.load('../saved_model/encoder_36.pkl'))

    def get_encodings(train_dataloader):
        encodings, labels, imgs = [], [], []

        model.eval()
        train_pbar = tqdm(total=len(train_dataloader), ncols=100, leave=True)
        for batch_idx, (img, label) in enumerate(train_dataloader):
            img, label = img.to(device), label.to(device)

            batch_size = img.size(0)
            with torch.no_grad():
                encoding = model(img)

            encodings += encoding.cpu().tolist()
            labels += label.cpu().tolist()
            '''
            if batch_idx % 200 == 0:
                imgs += img.cpu().tolist()
            '''
            train_pbar.update()
        train_pbar.close()

        return np.array(encodings), np.array(labels), np.array(imgs)

    train_encodings, train_labels, train_imgs = get_encodings(train_dataloader)
    test_encodings, test_labels, test_imgs = get_encodings(test_dataloader)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg
    from sklearn.manifold import TSNE

    def create_tsne(file_name, encodings, labels, font_list):
        tsne = TSNE(init='pca')
        encodings = tsne.fit_transform(encodings)

        def generate_color_list():
            color_list = []
            '''
            for i in range(7):
                color_list.append(plt.cm.Set2(i))

            for i in range(8):
                color_list.append(plt.cm.Set1(i))

            for i in range(10):
                color_list.append(plt.cm.Set3(i))
            '''

            for _ in range(200):
                color_list.append((np.random.rand(), np.random.rand(), np.random.rand(), 1.0))

            return color_list

        color_list = generate_color_list()

        legends = []
        for i, font_name in enumerate(font_list):
            legends.append(mpatches.Patch(color=color_list[i], label='{}'.format(font_name)))

        size = 3
        for idx, feature in enumerate(encodings):
            label = labels[idx]
            plt.scatter(feature[0], feature[1], c=color_list[label], s=size)

        # adding images
        '''
        fig, ax = plt.subplots()
        ax.set_xlim(-160, 160)
        ax.set_ylim(-160, 160)

        encodings = encodings.reshape(-1, 200, 2)
        encodings = np.mean(encodings, axis=1)

        for idx, feature in enumerate(encodings):

            img = np.moveaxis(train_imgs[idx], 0, 2)
            imagebox = OffsetImage(img, zoom=0.1)
            ab = AnnotationBbox(imagebox, (feature[0], feature[1]))
            ax.add_artist(ab)
        '''

        #plt.legend(handles=legends)
        plt.savefig('../{}_tsne.png'.format(file_name), bbox_inches='tight')
        plt.gcf().clear()

    create_tsne('train', train_encodings, train_labels, train_dataset.font_list)
    create_tsne('test', test_encodings, test_labels, test_dataset.font_list)
    #create_tsne('all', np.array(list(train_encodings) + list(test_encodings)), np.array([20 for _ in train_labels] + list(test_labels)), test_dataset.font_list)





if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build models
    StyleLearner_model = StyleLearner(encoding_dim=16).to(device)
    #encoder_model = StyleLearner_model.encoder.load_state_dict(torch.load('../saved_model/encoder.pkl'))

    if args.train:
        # create dataset loader
        from Dataset.chinese_font_style import ChineseFontImages
        train_dataset = ChineseFontImages(phase='train')
        test_dataset = ChineseFontImages(phase='test')

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

        train(StyleLearner_model, train_dataloader, test_dataloader, args, device)

    if args.test:
        # create dataset loader
        from Dataset.chinese_font_style import ChineseFontImagesEvaluate
        train_dataset = ChineseFontImagesEvaluate(phase='train')
        test_dataset = ChineseFontImagesEvaluate(phase='test')

        train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=False, num_workers=8, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=8, drop_last=False)

        test(StyleLearner_model, train_dataloader, test_dataloader, args, device)
