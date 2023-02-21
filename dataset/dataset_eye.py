import numpy as np
import os
from torch.utils.data import Dataset
import torch

import sys
sys.path.append('./source')

from utils import is_png_file, load_img, Augment_RGB_torch
from copy import deepcopy
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
import csv
from skimage.transform import resize
import skimage.feature
import skimage.segmentation
import cv2
from sklearn.preprocessing import StandardScaler
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

def isFloat(x):
    try:
        float(x)
        return True
    except:
        return False

def pickmatrix(matirx):
    idx_row = np.argwhere(np.all(matirx[...,:] == 0, axis = 0))
    matirx = np.delete(matirx, idx_row, axis = 1)
    idx2_col = np.argwhere(np.all(matirx[...,:] == 0, axis = 1))
    matirx = np.delete(matirx, idx2_col, axis = 0)
    matirx = resize(matirx , (100,100))
    H, W, C = matirx.shape
    return matirx[(H//2 - 40):(H//2 + 40), (W//2 - 40):(W//2 + 40), :]

def load_csv(filepath):
    CSVfiles = sorted(os.listdir(filepath))
    # print(CSVfiles)
    for file in CSVfiles:
        a = file.split('_',10)
        if len(a) > 4:
            if a[5] == 'CUR.CSV':
                cur_file = os.path.join(filepath, file)
            if a[5] == 'PAC.CSV':
                pac_file = os.path.join(filepath, file)

    curfile = open(cur_file, "r", encoding='unicode_escape')
    curreader = csv.reader(curfile)
    pacfile = open(pac_file, "r", encoding='unicode_escape')
    pacreader = csv.reader(pacfile)

    total_matrix = np.zeros((141,141,4))
    mt_number = 3
    mt_row = 0
    mt = np.zeros((80,80,3))
    for item in curreader:

        if len(item) == 1 or len(item) == 3:
            data = np.array(item[0].split(';'))  
            if data[0] in ['TANGENTIAL FRONT', 'TANGENTIAL BACK', 'TOTAL CORNEAL REFRACTIVE POWER']:
                mt_row = 0
                if data[0] == 'TANGENTIAL FRONT': mt_number = 0 
                if data[0] == 'TANGENTIAL BACK': mt_number = 1
                if data[0] == 'TOTAL CORNEAL REFRACTIVE POWER': mt_number = 2
            if isFloat(data[0]) is True:
                data[data==''] = '0'
                total_matrix[int(-10*float(data[0])+70),:,mt_number] = data[1:142].astype(float)
                mt_row += 1
            if data[0] == '[SYSTEM]':  
                break
    for item in curreader:
        if len(item) == 1 or len(item) == 3:
            data = np.array(item[0].split(';')) 
            if data[0]  == 'AXL Axial Length [mm]':
                AL = data[1]
    
    mt = pickmatrix(total_matrix[:,:,0:3])
    mt[:,:,0] = (mt[:,:,0] - 7) / (13 - 7)
    # mt[:,:,1] = mt[:,:,1]
    # mt[:,:,2] = mt[:,:,2] 
    mt[:,:,1] = (mt[:,:,1] - 5) / (15 - 5)
    # mt[:,:,2] = (mt[:,:,2] - 500) / (750 - 500)
    mt[:,:,2] = (mt[:,:,2] - 41) / (45 - 41)
    mt = resize(mt , (128,128))
    return mt[:,:,0:3], AL

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

### start add_len_para

def processmergematrix(sixmonths, baseline, lens_in, mask):
    matirx = baseline

    idx_row = np.argwhere(np.all(matirx[...,:] == 0, axis = 0))

    sixmonths = np.delete(sixmonths, idx_row, axis = 1)
    baseline = np.delete(baseline, idx_row, axis = 1)
    lens_in = np.delete(lens_in, idx_row, axis = 1)
    mask = np.delete(mask, idx_row, axis = 1)

    idx2_col = np.argwhere(np.all(matirx[...,:] == 0, axis = 1))

    sixmonths = np.delete(sixmonths, idx2_col, axis = 0)
    baseline = np.delete(baseline, idx2_col, axis = 0)
    lens_in = np.delete(lens_in, idx2_col, axis = 0)
    mask = np.delete(mask, idx2_col, axis = 0)

    sixmonths = resize(sixmonths , (100,100))
    baseline = resize(baseline , (100,100))
    lens_in = resize(lens_in , (100,100)) 
    mask = resize(mask , (100,100)) 

    H, W, C = baseline.shape

    sixmonths = sixmonths[(H//2 - 40):(H//2 + 40), (W//2 - 40):(W//2 + 40), :]
    baseline = baseline[(H//2 - 40):(H//2 + 40), (W//2 - 40):(W//2 + 40), :]
    lens_in = lens_in[(H//2 - 40):(H//2 + 40), (W//2 - 40):(W//2 + 40), :]
    mask = mask[(H//2 - 40):(H//2 + 40), (W//2 - 40):(W//2 + 40), :]

    baseline[:,:,0] = (baseline[:,:,0] - 7) / (13 - 7)
    baseline[:,:,1] = (baseline[:,:,1] - 5) / (15 - 5)
    baseline[:,:,2] = (baseline[:,:,2] - 41) / (45 - 41)

    sixmonths[:,:,0] = (sixmonths[:,:,0] - 7) / (13 - 7)
    sixmonths[:,:,1] = (sixmonths[:,:,1] - 5) / (15 - 5)
    sixmonths[:,:,2] = (sixmonths[:,:,2] - 41) / (45 - 41)

    sixmonths = resize(sixmonths , (128,128))
    baseline = resize(baseline , (128,128))
    lens_in = resize(lens_in , (128,128))
    mask = resize(mask , (128,128))

    return sixmonths, baseline, lens_in, mask

def load_raw_csv(filepath):
    CSVfiles = sorted(os.listdir(filepath))
    # print(CSVfiles)
    for file in CSVfiles:
        a = file.split('_',10)
        if len(a) > 4:
            if a[5] == 'CUR.CSV':
                cur_file = os.path.join(filepath, file)
            if a[5] == 'PAC.CSV':
                pac_file = os.path.join(filepath, file)

    curfile = open(cur_file, "r", encoding='unicode_escape')
    curreader = csv.reader(curfile)
    pacfile = open(pac_file, "r", encoding='unicode_escape')
    pacreader = csv.reader(pacfile)

    total_matrix = np.zeros((141,141,4))
    mt_number = 3
    mt_row = 0
    # mt = np.zeros((80,80,3))
    for item in curreader:

        if len(item) == 1 or len(item) == 3:
            data = np.array(item[0].split(';'))  
            if data[0] in ['TANGENTIAL FRONT', 'TANGENTIAL BACK', 'TOTAL CORNEAL REFRACTIVE POWER']:
                mt_row = 0
                if data[0] == 'TANGENTIAL FRONT': mt_number = 0 
                if data[0] == 'TANGENTIAL BACK': mt_number = 1
                if data[0] == 'TOTAL CORNEAL REFRACTIVE POWER': mt_number = 2
            if isFloat(data[0]) is True:
                data[data==''] = '0'
                total_matrix[int(-10*float(data[0])+70),:,mt_number] = data[1:142].astype(float)
                mt_row += 1
            if data[0] == '[SYSTEM]':  
                break

    return total_matrix[:,:,0:3]

def read_len_para(filename,  mode = 'origin'):

    dict_pre = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            vari = line.split(' ')[0]
            value = line.split(' ')[1]
            dict_pre[vari] = float(value)

    dict_ = {}

    if mode == 'origin':

        dict_['r_basic'] = dict_pre['r_basic']
        dict_['T_basic'] = dict_pre['T_basic']
        dict_['r_reverse'] = dict_pre['r_reverse']
        dict_['T_reverse'] = dict_pre['T_reverse']
        dict_['r_pos_1'] = dict_pre['r_pos_1']
        dict_['T_pos_1'] = dict_pre['T_pos_1']
        dict_['r_pos_2'] = dict_pre['r_pos_2']
        dict_['T_pos_2'] = dict_pre['T_pos_2']
        dict_['r_side'] = dict_pre['r_side']
        dict_['T_side'] = dict_pre['T_side']
    
    elif mode == 'norm': 

        dict_['r_basic'] = dict_pre['r_basic']
        dict_['T_basic'] = (dict_pre['T_basic'] - 7) / (13 - 7)
        dict_['r_reverse'] = dict_pre['r_reverse']
        dict_['T_reverse'] = dict_['T_reverse']
        dict_['r_pos_1'] = dict_pre['r_pos_1']
        dict_['T_pos_1'] = (dict_pre['T_pos_1'] - 7) / (13 - 7)
        dict_['r_pos_2'] = dict_pre['r_pos_2']
        dict_['T_pos_2'] = (dict_pre['T_pos_2'] - 7) / (13 - 7)
        dict_['r_side'] = dict_pre['r_side']
        dict_['T_side'] = dict_pre['T_side']
    
    elif mode == 'neg': 

        dict_['r_basic'] = dict_pre['r_basic']
        dict_['T_basic'] = dict_pre['T_basic']
        dict_['r_reverse'] = dict_pre['r_reverse']
        dict_['T_reverse'] = -5
        dict_['r_pos_1'] = dict_pre['r_pos_1']
        dict_['T_pos_1'] = dict_pre['T_pos_1']
        dict_['r_pos_2'] = dict_pre['r_pos_2']
        dict_['T_pos_2'] = dict_pre['T_pos_2']
        dict_['r_side'] = dict_pre['r_side']
        dict_['T_side'] = dict_pre['T_side']

    elif mode == 'norm_neg': 

        dict_['r_basic'] = dict_pre['r_basic']
        dict_['T_basic'] = (dict_pre['T_basic'] - 7) / (13 - 7)
        dict_['r_reverse'] = dict_pre['r_reverse']
        dict_['T_reverse'] = -0.5
        dict_['r_pos_1'] = dict_pre['r_pos_1']
        dict_['T_pos_1'] = (dict_pre['T_pos_1'] - 7) / (13 - 7)
        dict_['r_pos_2'] = dict_pre['r_pos_2']
        dict_['T_pos_2'] = (dict_pre['T_pos_2'] - 7) / (13 - 7)
        dict_['r_side'] = dict_pre['r_side']
        dict_['T_side'] = dict_pre['T_side']
            

    return  dict_

def read_spar(filename):

    with open(filename, 'r') as f:
        line_ = f.readline()
        line = line_.strip()
        spl = line.split(',')
        arr = []
        for i in spl:
            if i == '' or None:
                i = 11.98
            elif i == 'nan':
                i = 24.95
            arr.append(float(i))
        
    arr = np.array(arr)
    arr[1] = arr[1] / 10
    arr[2] = arr[2] / 11.98
    arr[3] = arr[3] / 1000
    arr[4] = arr[4] / 50
    arr = arr[2:]

    return arr

def _gen_lens(r_basic, T_basic, r_reverse, T_reverse, r_pos_1, T_pos_1, r_pos_2, T_pos_2, r_side, T_side):

    pix = 1024
    total_length = 2*(r_basic + r_reverse + r_pos_1 + r_pos_2 + r_side)
    pix_of_per_length = pix / total_length
    lens = np.zeros((pix, pix, 4))

    pos_tmp = np.zeros((pix, pix, 2))

    order = np.arange(pix)[np.newaxis, :]

    i_index = np.repeat(order.T, 1024, axis=1)
    j_index = np.repeat(order, 1024, axis=0)
    index = np.concatenate((i_index[:,:,np.newaxis], j_index[:,:,np.newaxis]), axis=2)
    center_index = np.array([(pix - 1)/2, (pix - 1)/2])[np.newaxis, np.newaxis, :]

    r_table  = np.linalg.norm(index - center_index, ord=2, axis=2) / pix_of_per_length

    lens[r_table < r_basic,0] = T_basic
    lens[np.logical_and(r_table >= r_basic, r_table < r_basic + r_reverse), 1] = T_reverse

    pos_tmp[np.logical_and(r_table >= r_basic + r_reverse,
     r_table < r_basic + r_reverse + r_pos_1), 0] = T_pos_1
    pos_tmp[np.logical_and(r_table >= r_basic + r_reverse + r_pos_1,
     r_table < r_basic + r_reverse + r_pos_1 + r_pos_2), 1] = T_pos_2

    lens[:,:,2] = pos_tmp.sum(axis=2)

    lens[np.logical_and(r_table >= r_basic + r_reverse + r_pos_1 + r_pos_2,
     r_table < r_basic + r_reverse + r_pos_1 + r_pos_2 + r_side), 3] = T_side
    
    return lens, pix_of_per_length

def gen_lens(dict_):
    
    return _gen_lens(dict_['r_basic'], dict_['T_basic'], dict_['r_reverse'], 
    dict_['T_reverse'], dict_['r_pos_1'], dict_['T_pos_1'], dict_['r_pos_2'], dict_['T_pos_2'], 
    dict_['r_side'], dict_['T_side'])

def align_lens(lens, pix_of_per_length, raw_size = 141, pix_of_per_length_base = 141/14):

    H, W, C = lens.shape
    new_lens = np.zeros((raw_size, raw_size, C))
    resize_lens = resize(lens, (int(H/pix_of_per_length*pix_of_per_length_base), 
    int(W/pix_of_per_length*pix_of_per_length_base)))

    Hn, Wn, _ = resize_lens.shape

    if Hn <= raw_size:
        new_lens[(raw_size-Hn)//2:((raw_size-Hn)//2+Hn), (raw_size-Wn)//2:((raw_size-Wn)//2+Wn), :] = resize_lens
    else:
        new_lens[:, :, :] = resize_lens[(Hn-raw_size)//2:((Hn-raw_size)//2+raw_size), (Wn-raw_size)//2:((Wn-raw_size)//2+raw_size), :]

    return new_lens

### end add_len_para

##################################################################################################
class Dataset_eye(Dataset):
    def __init__(self, csv_dir, txt, split, len_mode):
        super(Dataset_eye, self).__init__()
        
        baseline_files = []

        f = open(os.path.join(csv_dir, txt) , 'r')
        for line in f:
            line = line.strip()
            line = line.split(',')
            baseline_files.append([line[0],line[1]])        

        self.baseline_filenames = [os.path.join(csv_dir, 'base', x[1]) for x in baseline_files]
        self.sixmonths_filenames = [os.path.join(csv_dir, 'after', x[0], x[1]) for x in baseline_files]
        self.para_filenames = [os.path.join(csv_dir, 'base', x[1], 'Len_paras', 'v1.txt') for x in baseline_files]
        self.how_long_times = [int(x[0]) for x in baseline_files]
        self.AL_filenames = [os.path.join(csv_dir, 'after', x[0], x[1], 'Len_paras', 'AL.txt') for x in baseline_files]
        self.spar_filenames = [os.path.join(csv_dir, 'spar_5', x[0], x[1]+'.txt') for x in baseline_files]

        self.tar_size = len(self.sixmonths_filenames)  # get the size of target
        self.split = split
        self.len_mode = len_mode

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        sixmonths = load_raw_csv(self.sixmonths_filenames[tar_index])
        baseline = load_raw_csv(self.baseline_filenames[tar_index])
        how_time_long = self.how_long_times[tar_index]

        AL_filename = self.AL_filenames[tar_index]

        # if os.path.exists(AL_filename):
        #     with open(self.AL_filenames[tar_index], 'r') as fin:
        #         sixAL = float(fin.readline())
        #         sixAL = torch.from_numpy(np.array(np.float32(sixAL))[np.newaxis])
        # else:
        #     sixAL = torch.zeros(1) - 1

        with open(AL_filename, 'r') as fin:
            sixAL = float(fin.readline())
            sixAL = torch.from_numpy(np.array(np.float32(sixAL))[np.newaxis])

        spar_filename = self.spar_filenames[tar_index]
        spar_arr = read_spar(spar_filename)
        # scale=StandardScaler()
        # spar_arr=scale.fit_transform(spar_arr)
        ### generate len and mask
        
        paras = read_len_para(self.para_filenames[tar_index], self.len_mode)
        if paras['r_reverse'] == 0.0:
            spar_arr[-3] = 0.68
        else:
            spar_arr[-3] = paras['r_reverse']
        if paras['r_pos_1'] + paras['r_pos_2'] == 0.0:
            spar_arr[-4] = 1.23
        else:
            spar_arr[-4] = paras['r_pos_1'] + paras['r_pos_2']
        # print(paras['r_reverse'])
        # print(spar_arr)
        # spar_arr[-3] = spar_arr[-3] +0.2
        lens, pix_of_per_length = gen_lens(paras)
        # lens_in = lens
        mask_pre = np.zeros_like(lens[:, :, 0:1])
        mask_pre[lens[:, :, 0:1]!=0] = 1
        lens_in = lens.sum(axis=2, keepdims=True)
        lens_in = align_lens(lens_in, pix_of_per_length)
        mask = align_lens(mask_pre, pix_of_per_length)

        ### alien len and mask with base and sixmonths

        sixmonths, baseline, lens_in, mask =\
            processmergematrix(sixmonths, baseline, lens_in, mask)

        sixmonths = torch.from_numpy(np.float32(sixmonths))        
        baseline = torch.from_numpy(np.float32(baseline))
        lens_in = torch.from_numpy(np.float32(lens_in))
        mask = torch.from_numpy(np.float32(mask))
        how_time_long = torch.from_numpy(np.array(np.float32(how_time_long))[np.newaxis])
        spar_arr = torch.from_numpy(np.array(np.float32(spar_arr)))
        spar_arr_with_time = torch.concat([how_time_long, spar_arr], dim=0)
        
        baseline = torch.concat([baseline, mask, lens_in],dim=2)
          
        sixmonths = sixmonths.permute(2,0,1)
        baseline = baseline.permute(2,0,1)

        sixmonths_filename = os.path.split(self.sixmonths_filenames[tar_index])[-1]
        baseline_filename = os.path.split(self.baseline_filenames[tar_index])[-1]

        if self.split == 'train':
            apply_trans = transforms_aug[random.getrandbits(3)]
            sixmonths = getattr(augment, apply_trans)(sixmonths)
            baseline = getattr(augment, apply_trans)(baseline)
        elif self.split == 'val':
            pass
        else:
            raise Exception("Split error!")
        
        return sixmonths[0:1,:,:], baseline, sixAL, spar_arr_with_time, sixmonths_filename, baseline_filename

def get_training_data(rgb_dir, txt, len_mode='origin'):
    print(rgb_dir)
    assert os.path.exists(rgb_dir)
    return Dataset_eye(rgb_dir, txt, 'train', len_mode)


def get_validation_data(rgb_dir, txt, len_mode='origin'):
    assert os.path.exists(rgb_dir)
    return Dataset_eye(rgb_dir, txt, 'val', len_mode)
        
