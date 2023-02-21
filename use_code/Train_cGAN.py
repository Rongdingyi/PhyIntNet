import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#########################################

class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1  = nn.Linear(64*28*28+1000, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(10, 1000)

    def forward(self, x, labels):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28,28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64*28*28)
        y_ = self.fc3(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.sigmoid(x)

class ModelG(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
        self.fc2 = nn.Linear(10, 1000)
        self.fc = nn.Linear(self.z_dim+1000, 64*28*28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x) 
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.sigmoid(x)
        return x

class Options_cGAN():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--nepoch', type=int, default=2500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='Eyes')
        parser.add_argument('--pretrain_weights',type=str, default='/data1/rong/code/Eyes/data/Uformer_B.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='6,7', help='GPUs')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='/data1/rong/code/Eyes/exps/',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')
        
        # args for training
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--train_dir', type=str, default ='/data1/rong/code/Eyes/data',  help='dir of train data')
        parser.add_argument('--train_txt', type=str, default ='/data1/rong/code/Eyes/data/txts/train/main.txt',  help='txt of train data')
        parser.add_argument('--val_dir', type=str, default ='/data1/rong/code/Eyes/data',  help='dir of val data')
        parser.add_argument('--val_txt', type=str, default ='/data1/rong/code/Eyes/data/txts/val/main.txt',  help='txt of val data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 

        # args for model
        parser.add_argument('--nz', type=int,default=3, help='nz') 

        # args for experiment
        parser.add_argument('--train_mask', action='store_true',default=False)

        return parser

def psnr(target, ref):
    target_data = np.array(target)
    ref_data = np.array(ref)
#  20*math.log10(1.0/rmse)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    return 20*math.log10(np.max(target_data)/rmse)

def psnr_mask(target, ref, mask):
    target_data = np.array(target)
    ref_data = np.array(ref)
#  20*math.log10(1.0/rmse)
    diff = (ref_data - target_data)*mask
    if diff.sum() == 0:
        return 100
    else:
        diff = diff.flatten('C')
        rmse = math.sqrt(np.sum(diff ** 2.)/np.sum(mask))
        return 20*math.log10(np.max(target_data[mask!=0])/rmse)

if __name__ == '__main__':

    from torch.autograd import Variable

    import os
    import sys

    # add dir
    dir_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(dir_name,'../dataset/'))
    sys.path.append(os.path.join(dir_name,'..'))
    # print(sys.path)
    # print(dir_name)

    from skimage import img_as_ubyte
    import argparse

    import utils

    from dataset.dataset_eye import *
    import torch

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import random
    import time
    import numpy as np
    import datetime

    from losses import CharbonnierLoss,MaskLoss,ValidLoss

    from tqdm import tqdm 
    from warmup_scheduler import GradualWarmupScheduler
    from torch.optim.lr_scheduler import StepLR
    from timm.utils import NativeScaler
    from matplotlib import pyplot as plt
    import math

    INPUT_SIZE = 784
    SAMPLE_SIZE = 80
    NUM_LABELS = 10

    opt = Options_cGAN().init(argparse.ArgumentParser(description='Predicting eyes')).parse_args()

    model_d = ModelD()
    model_g = ModelG(opt.nz)
    criterion = nn.BCELoss()
    input = torch.FloatTensor(opt.batch_size, INPUT_SIZE)
    noise = torch.FloatTensor(opt.batch_size, (opt.nz))
    
    fixed_noise = torch.FloatTensor(SAMPLE_SIZE, opt.nz).normal_(0,1)
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS)
    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i*(SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0
    
    label = torch.FloatTensor(opt.batch_size)
    one_hot_labels = torch.FloatTensor(opt.batch_size, 10)
    if opt.cuda:
        model_d.cuda()
        model_g.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        one_hot_labels = one_hot_labels.cuda()
        fixed_labels = fixed_labels.cuda()

    optim_d = optim.SGD(model_d.parameters(), lr=opt.lr)
    optim_g = optim.SGD(model_g.parameters(), lr=opt.lr)
    fixed_noise = Variable(fixed_noise)
    fixed_labels = Variable(fixed_labels)

    real_label = 1
    fake_label = 0

    for epoch_idx in range(opt.epochs):
        model_d.train()
        model_g.train()
            
        d_loss = 0.0
        g_loss = 0.0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)
            train_x = train_x.view(-1, INPUT_SIZE)
            if opt.cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            input.resize_as_(train_x).copy_(train_x)
            label.resize_(batch_size).fill_(real_label)
            one_hot_labels.resize_(batch_size, NUM_LABELS).zero_()
            one_hot_labels.scatter_(1, train_y.view(batch_size,1), 1)
            inputv = Variable(input)
            labelv = Variable(label)

            output = model_d(inputv, Variable(one_hot_labels))
            optim_d.zero_grad()
            errD_real = criterion(output, labelv)
            errD_real.backward()
            realD_mean = output.data.cpu().mean()
            
            one_hot_labels.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size,1))).cuda()
            one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1)
            noise.resize_(batch_size, opt.nz).normal_(0,1)
            label.resize_(batch_size).fill_(fake_label)
            noisev = Variable(noise)
            labelv = Variable(label)
            onehotv = Variable(one_hot_labels)
            g_out = model_g(noisev, onehotv)
            output = model_d(g_out, onehotv)
            errD_fake = criterion(output, labelv)
            fakeD_mean = output.data.cpu().mean()
            errD = errD_real + errD_fake
            errD_fake.backward()
            optim_d.step()

            # train the G
            noise.normal_(0,1)
            one_hot_labels.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, NUM_LABELS, size=(batch_size,1))).cuda()
            one_hot_labels.scatter_(1, rand_y.view(batch_size,1), 1)
            label.resize_(batch_size).fill_(real_label)
            onehotv = Variable(one_hot_labels)
            noisev = Variable(noise)
            labelv = Variable(label)
            g_out = model_g(noisev, onehotv)
            output = model_d(g_out, onehotv)
            errG = criterion(output, labelv)
            optim_g.zero_grad()
            errG.backward()
            optim_g.step()
            
            d_loss += errD.data[0]
            g_loss += errG.data[0]
            if batch_idx % opt.print_every == 0:
                print(
                "\t{} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
                    format(epoch_idx, batch_idx, len(train_loader), fakeD_mean,
                        realD_mean))

                g_out = model_g(fixed_noise, fixed_labels).data.view(
                    SAMPLE_SIZE, 1, 28,28).cpu()

        print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
            d_loss, g_loss))
        # if epoch_idx % opt.save_every == 0:
        #     torch.save({'state_dict': model_d.state_dict()},
        #                 '{}/model_d_epoch_{}.pth'.format(
        #                     opt.save_dir, epoch_idx))
        #     torch.save({'state_dict': model_g.state_dict()},
        #                 '{}/model_g_epoch_{}.pth'.format(
        #                     opt.save_dir, epoch_idx))