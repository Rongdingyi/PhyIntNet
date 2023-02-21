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
# from torchcam.methods import SmoothGradCAMpp, GradCAM, LayerCAM
from dataset.dataset_eye import *
from pytorch_grad_cam import GradCAM
# from torchcam.utils import overlay_mask
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from models.model_main import Precoding,PicPrecoding
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Predicting eyes')).parse_args()
print(opt)

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

weights = './model_best.pth'
model_restoration = model_utils.get_arch(opt)
# model.load_state_dict(torch.load(weights))


model_utils.load_checkpoint(model_restoration,weights)
print("===>Testing using weights: ", weights)

model_restoration.cuda()
val_dataset = get_validation_data(opt.val_dir, opt.val_txt, opt.len_mode)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)


class SemanticSegmentationTarget:
    def __init__(self, mask):
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[0, :, : ] * self.mask).sum()

mask = np.float32(np.ones((128,128)))
# print(mask)
# with torch.no_grad():
model_restoration.eval()
ssims= []
psnr = []
for ii, data_val in enumerate((val_loader), 0):
    target = data_val[0].cuda()
    input_ = data_val[1].cuda()
    sixAL = data_val[2].cuda()
    spar_with_time = data_val[3].cuda()
    filenames = data_val[4]
    with torch.cuda.amp.autocast():
        if opt.predict_AL:
            restored, pre_length  = model_restoration(input_, spar_with_time)
            # al_mse.append(criterion2(sixAL, pre_length).cpu().detach().numpy())
        else:
            restored = model_restoration(input_, spar_with_time)
        restored = torch.clamp(restored,0,1).cpu().detach().numpy().squeeze(0).transpose((1,2,0))
        target = torch.clamp(target,0,1).cpu().detach().numpy().squeeze(0).transpose((1,2,0))
        ssims.append(ssim_loss(restored[:,:,0]*6+7, target[:,:,0]*6+7,data_range=6))
        # psnr.append(psnr_loss(input_[:,:,0]*6+7, restored[:,:,0]*6+7))
print(np.mean(ssims))
        # print(restored.shape, input_.shape),data_range=6
        # utils.save_img(os.path.join('/data1/rong/code/Eyes/source_rong/results/grad_maindot/', filenames[0]+'.png'), img_as_ubyte(restored))

    # print(torch.clamp(restored,0,1).cpu().numpy().squeeze(0).shape)
    # restored = torch.clamp(restored,0,1).cpu().numpy().squeeze(0).transpose((1,2,0))
    # print(restored, pre_length)
    # utils.save_img(os.path.join(args.result_dir,filenames[0]+'.png'), img_as_ubyte(restored))
