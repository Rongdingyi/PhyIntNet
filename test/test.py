import numpy as np

import os,sys
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
import options
import utils
import argparse
from tqdm import tqdm
from einops import rearrange, repeat
from models import model_utils
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset.dataset_eye import *
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from models.model import Precoding, PicPrecoding, Uformer
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Predicting eyes')).parse_args()
print(opt)

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
precoding_model = PicPrecoding(in_channel=4, out_channel=32, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
model_restoration = Uformer(img_size=opt.train_ps, out_chans=1,
                embed_dim=opt.embed_dim, depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                win_size=opt.win_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                patch_norm=True, use_checkpoint=False, token_projection='linear', token_mlp='leff',
                shift_flag=True, modulator=True, cross_modulator=False, precoding=precoding_model, use_LENmu=opt.use_LENmu, 
                time_encode=opt.time_encode, qk_mode=opt.qk_mode, predict_AL=opt.predict_AL)

if opt.resume:
    path_chk_rest = opt.pretrain_weights
    print("Resume from "+path_chk_rest)
    model_utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = model_utils.load_start_epoch(path_chk_rest) + 1 
    print(start_epoch)

model_restoration.cuda()
test_dataset = get_validation_data(opt.test_dir, opt.test_txt, opt.len_mode)
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
with torch.no_grad():
    model_restoration.eval()
    ssims= []
    psnr = []
    for ii, data_val in enumerate((test_loader), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        sixAL = data_val[2].cuda()
        spar_with_time = data_val[3].cuda()
        filenames = data_val[4]
        with torch.cuda.amp.autocast():
            if opt.predict_AL:
                restored, pre_length  = model_restoration(input_, spar_with_time)
            else:
                restored = model_restoration(input_, spar_with_time)
            psnr.append(utils.batch_PSNR_mask(restored[:,0,:,:], target[:,0,:,:], input_[:,3,:,:], False).item())
            restored = torch.clamp(restored,0,1).cpu().detach().numpy().squeeze(0).transpose((1,2,0))
            target = torch.clamp(target,0,1).cpu().detach().numpy().squeeze(0).transpose((1,2,0))
            ssims.append(ssim_loss(restored[:,:,0], target[:,:,0]))
    print(np.mean(ssims),sum(psnr)/len(psnr))