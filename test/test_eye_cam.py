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
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Predicting eyes')).parse_args()
print(opt)

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

weights = 'model_best.pth'
model_restoration = model_utils.get_arch(opt)

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

def generate_mask(img_height,img_width,radius,center_x,center_y):
 
    y,x=np.ogrid[0:img_height,0:img_width]
 
    # circle mask
 
    mask = (x-center_x)**2+(y-center_y)**2<=radius**2
 
    return mask

mask = np.float32(np.ones((128,128)))
# print(mask)
# with torch.no_grad():
model = model_restoration.eval()
target_layers = [model.output_proj]
# cam_extractor = GradCAM(model, 'output_proj')

for ii, data_val in enumerate((val_loader), 0):
    target = data_val[0].cuda()
    input_ = data_val[1].cuda()
    sixAL = data_val[2].cuda()
    spar_with_time = data_val[3].cuda()
    filenames = data_val[4]
    a = [input_, spar_with_time]
    # print(len(filenames))
    with torch.cuda.amp.autocast():
        if opt.predict_AL:
            restored = model(a)
            # al_mse.append(criterion2(sixAL, pre_length).cpu().detach().numpy())
        else:
            restored = model(a)

        # show_cam_on_image
        # models = model_restoration
        # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        # targets = [SemanticSegmentationTarget(mask)]
        # grayscale_cam = cam(input_tensor=a, targets=targets)
        # mask_cir1  = generate_mask(128,128,40,64,64)
        # mask_cir2  = generate_mask(128,128,48,64,64)
        # mask_cam1 = mask_cir1 * grayscale_cam
        # mask_cam2 = mask_cir2 * grayscale_cam
        # mask_cam = mask_cam2 - mask_cam1
        # mask_cam[:,64:128,0:64] = 0

        # print(sum(sum(sum(mask_cam)))/sum(sum(sum(grayscale_cam))))
        # print(grayscale_cam.shape)
        # input_ = input_.cpu().detach().numpy().squeeze(0)
        # input_ = input_[0:3,:,:]

        # input_ =np.transpose(input_,[1,2,0])
        # input_ = input_.astype(np.uint8)
        # print(grayscale_cam.shape)
        # cam_images = [show_cam_on_image(np.transpose(np.float32(img.cpu().detach().numpy()[0:3,:,:])/255.,[1,2,0]), grayscale, use_rgb=True) for img, grayscale in zip(input_, grayscale_cam)]
        # print(cam_images[0].shape)
        # utils.save_img(os.path.join('/data1/rong/code/Eyes/source_first_dist_mae/result/', filenames[0]+'.png'), img_as_ubyte(cam_images[0]))
        # cam_image = show_cam_on_image(input_, grayscale_cam, use_rgb=True)
    output = torch.clamp(restored,0,1).detach().cpu().numpy().squeeze(0)
    input1 = torch.clamp(input_,0,1).detach().cpu().numpy().squeeze(0)
    # print(torch.clamp(restored,0,1).detach().cpu().numpy().squeeze(0).shape)
    change1 = output - input1[0:1,:,:]
    change = change1[0:1,8:120,8:120]
    change = change - change[0,:,:].min()
    change = change / change[0,:,:].max()
    # print(max(change))
    # change = change / (max(change) - min(change))
    print(change)
    # print(change.shape)
    # pic = change[0:1,16:112,16:112]
    # pic = pic - pic[0,:,:].min()
    # pic = pic / pic[0,:,:].max()
    # pic = pic / (max(pic) - min(pic))
    # print(pic.shape)
    # change[0,16:112,16:112] = change[0,16:112,16:112]  / (max(change[0,16:112,16:112]) - min(change[0,16:112,16:112]))
    # print(change.shape)
    # print(change[0,:,:].min())
    # restored = torch.clamp(restored,0,1).cpu().numpy().squeeze(0).transpose((1,2,0))filenames[0]+
    # print(restored, pre_length)
    utils.save_img(os.path.join('/data1/rong/code/Eyes/source_first_dist_mae/','pic.png'), img_as_ubyte(change[0,:,:]))
