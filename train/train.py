import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
print(sys.path)
print(dir_name)

from skimage import img_as_ubyte
import argparse
import options

import utils
from models import model_utils

from dataset.dataset_eye import *
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from skimage.metrics import structural_similarity as ssim_loss
from losses import CharbonnierLoss,MaskLoss,ValidLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from matplotlib import pyplot as plt
from models.model_main import Precoding,PicPrecoding
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Predicting eyes')).parse_args()
print(opt)

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
torch.backends.cudnn.benchmark = True

######### Logs dir ###########
exp_dir = os.path.join(opt.save_dir, opt.env)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

print("Now time is : ",datetime.datetime.now().isoformat())
log_dir = os.path.join(exp_dir, datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
img_dir = os.path.join(log_dir, 'results', 'img')
img_last_dir = os.path.join(log_dir, 'results', 'img_last')
len_dir = os.path.join(log_dir, 'results', 'len')
bar_dir = os.path.join(log_dir, 'results', 'bar')
mask_dir = os.path.join(log_dir, 'results', 'mask')
mask_img_dir = os.path.join(log_dir, 'results', 'mask_img')
model_dir  = os.path.join(log_dir, 'models')

logname = os.path.join(log_dir, 'log.txt') 
opt_record_name = os.path.join(log_dir, 'opt_record.txt') 

utils.mkdir(result_dir)
utils.mkdir(model_dir)
utils.mkdir(img_dir)
utils.mkdir(img_last_dir)
utils.mkdir(len_dir)
utils.mkdir(bar_dir)
utils.mkdir(mask_dir)
utils.mkdir(mask_img_dir)
os.system(command='rsync -rq {} {} --exclude=.git*'.format(os.path.abspath(os.path.join(dir_name, '../../source_first_dist')), os.path.abspath(log_dir)))


def gen_decayed_len_with_time(len_in, how_long_time):

    decay = torch.exp(-how_long_time/24).unsqueeze(1).unsqueeze(1)
    decayed_len = torch.clone(len_in)
    decayed_len = decayed_len*decay

    return decayed_len


# ######### Set Seeds ###########
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

######### Model ###########
model_restoration = model_utils.get_arch(opt)

with open(opt_record_name,'w') as f:
    f.write('Namespace:\n')
    for k in list(vars(opt).keys()):
        f.write('\t%s: %s' % (k, vars(opt)[k]) + '\n')

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    # f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ########### 
model_restoration = torch.nn.DataParallel(model_restoration) 
model_restoration.cuda() 

checkpoint = torch.load(weights)
######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ########### 
if opt.resume: 
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    model_utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = model_utils.load_start_epoch(path_chk_rest) + 1 
    # lr = model_utils.load_optim(optimizer, path_chk_rest) 

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

######### Loss ###########
criterion_mask = MaskLoss().cuda()
criterion = CharbonnierLoss().cuda()
if opt.predict_AL:
    criterion2 = torch.nn.MSELoss().cuda()
    # criterion2 = ValidLoss().cuda()
    # criterion2 = CharbonnierLoss(1e-10).cuda()

w1 = opt.w1
w2 = opt.w2

######### DataLoader ###########opt.batch_size
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, opt.train_txt, opt.len_mode)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=True)
val_dataset = get_validation_data(opt.val_dir, opt.val_txt, opt.len_mode)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
######### validation ###########
if True:
    with torch.no_grad():
        model_restoration.eval()
        psnr_dataset = []
        psnr_model_init = []
        al_mse = []
        all_lens = []
        all_filenames = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            # print(input_.shape)
            sixAL = data_val[2].cuda()
            spar_with_time = data_val[3].cuda()
            filenames = data_val[4]
            all_filenames.extend(filenames)
            with torch.cuda.amp.autocast():
                if opt.predict_AL:
                    restored, pre_length  = model_restoration(input_, spar_with_time)
                    al_mse.append(criterion2(sixAL, pre_length).cpu().detach().numpy())
                else:
                    restored = model_restoration(input_, spar_with_time)
                    al_mse.append(0)
            
            all_lens.append(input_.cpu().numpy())
            if opt.train_mask:
                psnr_dataset.append(utils.batch_PSNR_mask(input_[:,0,:,:], target[:,0,:,:], input_[:,3,:,:], False).item())
                psnr_model_init.append(utils.batch_PSNR_mask(restored[:,0,:,:], target[:,0,:,:], input_[:,3,:,:], False).item())
            else:
                psnr_dataset.append(utils.batch_PSNR(input_[:,0,:,:], target, False).item())
                psnr_model_init.append(utils.batch_PSNR(restored[:,0,:,:], target, False).item())
        
        all_lens = np.concatenate(all_lens, axis=0)
        fig1 = plt.figure()
        for j in range(len(all_filenames)):
            len_i = all_lens[j, -2, ...]
            plt.clf()
            plt.imsave(os.path.join(len_dir, all_filenames[j]+'.png'), len_i)
            plt.imshow(len_i)
            plt.colorbar()
            fig1.savefig(os.path.join(bar_dir, all_filenames[j]+'.png'))

            mask_i = all_lens[j, 3:4, ...]
            plt.imsave(os.path.join(mask_img_dir, all_filenames[j]+'.png'), mask_i[0, ...])
            np.save(os.path.join(mask_dir, all_filenames[j]+'.npy'), mask_i)

        psnr_dataset = sum(psnr_dataset)/len_valset
        psnr_model_init = sum(psnr_model_init)/len_valset
        al_mse = np.array(al_mse).mean()
        print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_model_init))
        print('MSE Init -->%.4f'%(al_mse))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
best_mse = 10000
best_psnr_img = 0
eval_now = len(train_loader)//4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    train_id = 1

    print("------------------------------------------------------------------")
    print("----- Begin Epoch: {} -----".format(epoch))
    with open(logname,'a') as f:
        f.write('\n' + "----- Begin Epoch: {} -----".format(epoch)+'\n')

    for i, data in enumerate(tqdm(train_loader), 0): 
        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()
        sixAL = data[2].cuda()
        spar_with_time = data[3].cuda()

        if epoch>5:
            target, input_, sixAL, spar_with_time = utils.MixUp_AUG().aug_all(target, input_, sixAL, spar_with_time)

        with torch.cuda.amp.autocast():          
            if opt.predict_AL:
                restored, pre_length  = model_restoration(input_, spar_with_time)
                # print(pre_length)
                loss2 = criterion2(sixAL, pre_length)
                # print(sixAL.cpu().detach().numpy())
                # print(pre_length.cpu().detach().numpy())
            else:
                restored = model_restoration(input_, spar_with_time)
                loss2 = torch.tensor(0)
            
            if opt.train_mask:
                loss = criterion_mask(restored, target, input_[:,3:4,:,:])
                total_loss = w1 * loss + w2 * loss2
                
            else:
                loss = criterion(restored, target)
                total_loss = w1 * loss + w2 * loss2
                

        loss_scaler(total_loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss += total_loss.item()
        epoch_loss1 += loss.item()
        epoch_loss2 += loss2.item()

        #### Evaluation ####

        all_filenames = []
        all_restored = []

        if (i+1)%eval_now==0 and i>0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                psnr_list = [] 
                al_mse = []
                # mse_axial = []
                ssims = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    sixAL = data_val[2].cuda()
                    spar_with_time = data_val[3].cuda()
                    filenames = data_val[4]
                    all_filenames.extend(filenames)

                    with torch.cuda.amp.autocast(): 
                        if opt.predict_AL:
                            restored, pre_length  = model_restoration(input_, spar_with_time)
                            al_mse.append(criterion2(sixAL, pre_length).cpu().detach().numpy())
                        else:
                            restored = model_restoration(input_, spar_with_time)
                            # restored1 = torch.clamp(restored,0,1).cpu().detach().numpy().squeeze(0).transpose((1,2,0))
                            # target1 = torch.clamp(target,0,1).cpu().detach().numpy().squeeze(0).transpose((1,2,0))
                            # ssims.append(ssim_loss(restored1[:,:,0]*6+7, target1[:,:,0]*6+7,data_range=6))
                            al_mse.append(0)

                    all_restored.append(torch.clamp(restored,0,1).cpu().numpy())

                    if opt.train_mask:
                        psnr_val_rgb.append(utils.batch_PSNR_mask(restored, target, input_[:,3:4,:,:], False).item())
                        psnr_list.append(utils.batch_PSNR_mask(restored*6+7, target*6+7, input_[:,3:4,:,:], False).item())
                    else:
                        psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
                        psnr_list.append(utils.batch_PSNR(restored*6+7, target*6+7, False).item())
                
                al_mse1 = np.array(al_mse).mean()
                # print(np.mean(ssims))
                psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                psnr_img = sum(psnr_list)/len_valset
                # mse_axial = sum(mse_axial)/len_valset
                # print(mse_axial)
                all_restored = np.concatenate(all_restored, axis=0)
                for j in range(len(all_filenames)):
                    restored_i = all_restored[j, ...].transpose((1,2,0))
                    utils.save_img(os.path.join(img_last_dir, all_filenames[j]+'.png'), img_as_ubyte(restored_i))

                if psnr_img > best_psnr:
                    best_psnr = psnr_img
                    best_mse = al_mse1
                    best_epoch = epoch
                    best_psnr_img = psnr_img
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                    # all_restored = np.concatenate(all_restored, axis=0)\t AL_pre: {}\t AL_tar: {:.4f} al_mse, sixAL
                    for j in range(len(all_filenames)):
                        restored_i = all_restored[j, ...].transpose((1,2,0))
                        utils.save_img(os.path.join(img_dir, all_filenames[j]+'.png'), img_as_ubyte(restored_i))

                timeinfo = time.strftime("[%y.%m.%d|%X]\t", time.localtime())
                print(timeinfo + "Epoch: {:d}\t iter: {}\t PSNR: {:.4f}\t PSNR_img: {:.4f}\t AL_mse: {:.4f}".format(epoch, str(i).rjust(2, '0'), psnr_val_rgb, psnr_img, al_mse1))
                with open(logname,'a') as f:
                    f.write(timeinfo + "Epoch: {:d}\t iter: {}\t PSNR: {:.4f}\t PSNR_img: {:.4f}\t AL_mse: {:.4f}".format(epoch, str(i).rjust(2, '0'), psnr_val_rgb, psnr_img, al_mse1) + '\n')
                model_restoration.train()
                torch.cuda.empty_cache()

    scheduler.step()
    
    print("End Epoch: {}\tTime: {:.4f}\ttotal_Loss: {:.4f}\tLoss_eye: {:.4f}\tLoss_AL: {:.4f}\tLearningRate {:.6f} ---- [best_epoch: {:d} best_iter: {:d} best_PSNR: {:.4f} best_PSNR_img: {:.4f} best_mse: {:.4f}]"\
            .format(epoch, time.time()-epoch_start_time,epoch_loss,epoch_loss1,epoch_loss2, scheduler.get_lr()[0], best_epoch, best_iter, best_psnr, best_psnr_img, best_mse))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("----- [best_epoch: {:d} best_iter: {:d} best_PSNR: {:.4f} best_PSNR_img: {:.4f} best_mse: {:.4f}] -----\n".format(best_epoch, best_iter, best_psnr, best_psnr_img, best_mse) + \
            "----- End Epoch: {}\tTime: {:.4f}\ttotal_Loss: {:.4f}\tLoss_eye: {:.4f}\tLoss_AL: {:.4f}\tLearningRate {:.6f} -----\n".format(epoch, time.time()-epoch_start_time,epoch_loss,epoch_loss1,epoch_loss2, scheduler.get_lr()[0]))
    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    # if epoch%opt.checkpoint == 0:
    #     torch.save({'epoch': epoch, 
    #                 'state_dict': model_restoration.state_dict(),
    #                 'optimizer' : optimizer.state_dict()
    #                 }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 

print("Now time is : ",datetime.datetime.now().isoformat())
