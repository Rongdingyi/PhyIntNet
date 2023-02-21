from cProfile import label
import os
from turtle import forward
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as trans
import torchvision.models as models
from models.model_main import Precoding, PicPrecoding
# from 
from dataset.dataset_eye import *
# 设置参数
batch_size = 128
iters = 5000
init_lr = 5e-4
optimizer_type = 'adam'
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

def gen_decayed_len_with_time(len_in, how_long_time):

    decay = torch.exp(-how_long_time/24).unsqueeze(1).unsqueeze(1)
    decayed_len = torch.clone(len_in)
    decayed_len = decayed_len*decay

    return decayed_len

def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, evl_interval, device):
    iter = 0
    model.train()
    avg_loss_list = []
    best_dice = 1000
    while iter < iters:
        for _, data in enumerate(train_dataloader):
            iter += 1
            if iter > iters:
                break
            target = data[0].cuda()
            input_ = data[1].cuda()
            sixAL = data[2].cuda()
            spar_with_time = data[3].cuda()
            filenames = data[4]
            decayed_len = gen_decayed_len_with_time(input_[:,4:5,:,:], spar_with_time[:,0:1])
            merge_len = torch.concat([input_[:,4:5,:,:], decayed_len], dim = 1)
            # LEN_y = self.LEN_proj(merge_len)
            code, logits = model(input_[:,0:4,:,:])
            # print(gt_label)
            loss = criterion(logits, input_[:,0:4,:,:])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss_list.append(loss.cpu().detach().numpy())

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_loss_list = []
                print("[TRAIN] iter={}/{} avg_loss={:.4f}".format(iter, iters, avg_loss))

            if iter % evl_interval == 0:
                avg_loss = val(model, val_dataloader, criterion, device)
                print("[EVAL] iter={}/{} avg_loss={:.4f}".format(iter, iters, avg_loss))
                if avg_loss <= best_dice:
                    best_dice = avg_loss
                    torch.save(model.state_dict(), '/data1/rong/code/Eyes/exp/pic_encode.pth')

                model.train()

# 验证函数
def val(model, val_dataloader, criterion, device):
    model.eval()
    avg_loss_list = []
    with torch.no_grad():
        for i,data in enumerate(val_dataloader):
            target = data[0].cuda()
            input_ = data[1].cuda()
            sixAL = data[2].cuda()
            spar_with_time = data[3].cuda()
            filenames = data[4]
            decayed_len = gen_decayed_len_with_time(input_[:,4:5,:,:], spar_with_time[:,0:1])
            merge_len = torch.concat([input_[:,4:5,:,:], decayed_len], dim = 1)
            code, logits = model(input_[:,0:4,:,:])
            loss = criterion(logits, input_[:,0:4,:,:])
            # print([i,spar_with_time,pred, sixAL,filenames])
            avg_loss_list.append(loss.cpu().detach().numpy())

    avg_loss = np.array(avg_loss_list).mean() # list转array
    print(avg_loss_list)
    return avg_loss

# 训练阶段

# 设置传入参数
img_options_train = {'patch_size':128}

train_dataset = get_training_data('/data1/rong/code/Eyes/data', '/data1/rong/code/Eyes/data/txts/train/all.txt', 'origin')
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, 
        num_workers=4, pin_memory=False, drop_last=True)
val_dataset = get_validation_data('/data1/rong/code/Eyes/data', '/data1/rong/code/Eyes/data/txts/val/all.txt', 'origin')
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, 
        num_workers=4, pin_memory=False, drop_last=False)


model = PicPrecoding(in_channel=4, out_channel=32, kernel_size=3, stride=1, act_layer=nn.LeakyReLU).cuda()
# model = nn.Sequential(nn.Linear(13, 32), nn.LeakyReLU(inplace=True),  nn.Linear(32, 1)).to(device)
# resnet.resnet50(num_classes=1, include_top=True).to(device), nn.LeakyReLU(inplace=True), 
                                    # nn.Linear(64, 32), nn.LeakyReLU(inplace=True), nn.Linear(32, 1)
if optimizer_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = torch.nn.MSELoss().cuda()


# 开始训练
train(model, iters, train_loader, val_loader, optimizer, criterion, log_interval=25, evl_interval=125, device=device)
        