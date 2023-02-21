import openpyxl
import os
import numpy as np
from skimage.transform import resize
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

def isFloat(x):
    try:
        float(x)
        return True
    except:
        return False

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

### start add_len_para

def processmergematrix(sixmonths, baseline, lens_in):
    matirx = baseline

    idx_row = np.argwhere(np.all(matirx[...,:] == 0, axis = 0))

    sixmonths = np.delete(sixmonths, idx_row, axis = 1)
    baseline = np.delete(baseline, idx_row, axis = 1)
    lens_in = np.delete(lens_in, idx_row, axis = 1)

    idx2_col = np.argwhere(np.all(matirx[...,:] == 0, axis = 1))

    sixmonths = np.delete(sixmonths, idx2_col, axis = 0)
    baseline = np.delete(baseline, idx2_col, axis = 0)
    lens_in = np.delete(lens_in, idx2_col, axis = 0)

    sixmonths = resize(sixmonths , (100,100))
    baseline = resize(baseline , (100,100))
    lens_in = resize(lens_in , (100,100)) 

    H, W, C = baseline.shape

    sixmonths = sixmonths[(H//2 - 40):(H//2 + 40), (W//2 - 40):(W//2 + 40), :]
    baseline = baseline[(H//2 - 40):(H//2 + 40), (W//2 - 40):(W//2 + 40), :]
    lens_in = lens_in[(H//2 - 40):(H//2 + 40), (W//2 - 40):(W//2 + 40), :]

    # baseline[:,:,0] = (baseline[:,:,0] - 7) / (12 - 7)
    # baseline[:,:,1] = (baseline[:,:,1] - 5) / (15 - 5)
    # baseline[:,:,2] = (baseline[:,:,2] - 41) / (45 - 41)

    # sixmonths[:,:,0] = (sixmonths[:,:,0] - 7) / (12 - 7)
    # sixmonths[:,:,1] = (sixmonths[:,:,1] - 5) / (15 - 5)
    # sixmonths[:,:,2] = (sixmonths[:,:,2] - 41) / (45 - 41)

    sixmonths = resize(sixmonths , (128,128))
    baseline = resize(baseline , (128,128))
    lens_in = resize(lens_in , (128,128))

    return sixmonths, baseline, lens_in 

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
    for item in curreader:
        if len(item) == 1 or len(item) == 3:
            data = np.array(item[0].split(';')) 
            if data[0]  == 'AXL Axial Length [mm]':
                AL = data[1]
    
    '''
    mt = pickmatrix(total_matrix[:,:,0:3])
    mt[:,:,0] = (mt[:,:,0] - 7) / (12 - 7)
    # mt[:,:,1] = mt[:,:,1]
    # mt[:,:,2] = mt[:,:,2] 
    mt[:,:,1] = (mt[:,:,1] - 5) / (15 - 5)
    # mt[:,:,2] = (mt[:,:,2] - 500) / (750 - 500)
    mt[:,:,2] = (mt[:,:,2] - 41) / (45 - 41)
    mt = resize(mt , (128,128))
    '''

    return total_matrix[:,:,0:3], AL

def read_len_para(filename = None):

    dict_ = {}
    
    dict_['r_basic'] = 1.5
    dict_['T_basic'] = -0.3
    dict_['r_reverse'] = 0.5
    dict_['T_reverse'] = 0.6
    dict_['r_pos'] = 0.7
    dict_['T_pos'] = 0.1
    dict_['r_side'] = 0.3
    dict_['T_side'] = 0.05

    return  dict_

def _gen_lens(r_basic, T_basic, r_reverse, T_reverse, r_pos, T_pos, r_side, T_side):

    pix = 1024
    total_length = 2*(r_basic + r_reverse + r_pos + r_side)
    pix_of_per_length = pix / total_length
    lens = np.zeros((pix, pix, 4))

    order = np.arange(pix)[np.newaxis, :]

    i_index = np.repeat(order.T, 1024, axis=1)
    j_index = np.repeat(order, 1024, axis=0)
    index = np.concatenate((i_index[:,:,np.newaxis], j_index[:,:,np.newaxis]), axis=2)
    center_index = np.array([(pix - 1)/2, (pix - 1)/2])[np.newaxis, np.newaxis, :]

    r_table  = np.linalg.norm(index - center_index, ord=2, axis=2) / pix_of_per_length

    lens[r_table < r_basic,0] = T_basic
    lens[np.logical_and(r_table >= r_basic, r_table < r_basic + r_reverse), 1] = T_reverse
    lens[np.logical_and(r_table >= r_basic + r_reverse,
     r_table < r_basic + r_reverse + r_pos), 2] = T_pos
    lens[np.logical_and(r_table >= r_basic + r_reverse + r_pos,
     r_table < r_basic + r_reverse + r_pos + r_side), 3] = T_side
    
    return lens, pix_of_per_length

def gen_lens(dict_):
    
    return _gen_lens(dict_['r_basic'], dict_['T_basic'], dict_['r_reverse'], 
    dict_['T_reverse'], dict_['r_pos'], dict_['T_pos'], dict_['r_side'], dict_['T_side'])

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

def load_clean_len_excel(excel_name):
    workbook = openpyxl.load_workbook(excel_name)
    M_len_sheet = workbook['M_401']
    E_len_sheet = workbook['E_212']
    Index_sheet = workbook['Index']
    m_len = []
    m_index = []
    e_len = []
    e_index = []
    index = {}

    M_stop = 339

    for i in Index_sheet.values:
        index[i[0]] = i[1]
    # E, F, G, H, I, J, K
    # 定位弧曲率	降幅	总直径	光学区直径	反转弧宽度	定位弧宽度	周边弧宽度
    for _, i in enumerate(M_len_sheet.values):
        m_index.append('{}_OD'.format(index[i[0]]))
        m_index.append('{}_OS'.format(index[i[0]]))
        m_len.append(i[4:11])
        m_len.append(i[11:18])
    # E, F, G, H, I, J, K, L
    # 定位弧AC1曲率	降幅	总直径	光学区直径	反转弧宽度	定位弧AC1宽度	定位弧AC2宽度	周边弧宽度
    for _, i in enumerate(E_len_sheet.values):
        e_index.append('{}_OD'.format(index[i[0]]))
        e_index.append('{}_OS'.format(index[i[0]]))
        e_len.append(i[4:12])
        e_len.append(i[12:20])

    m_len_clean = []
    m_index_clean = []
    e_len_clean = []
    e_index_clean = []

    # E, F, G, H, I, J, K
    # 定位弧曲率	降幅	总直径	光学区直径	反转弧宽度	定位弧宽度	周边弧宽度
    for i, item in enumerate(m_len[0:(2*M_stop + 1)]):
        if (isinstance(item[-1], float) or isinstance(item[-1], int)) and (isinstance(item[-2], float) or isinstance(item[-2], int)):
            m_index_clean.append(m_index[i])
            m_len_clean.append(item)

    # E, F, G, H, I, J, K, L
    # 定位弧AC1曲率	降幅	总直径	光学区直径	反转弧宽度	定位弧AC1宽度	定位弧AC2宽度	周边弧宽度
    for i, item in enumerate(e_len):
        if (isinstance(item[-1], float) or isinstance(item[-1], int)) and (isinstance(item[1], float) or isinstance(item[1], int)):
            e_index_clean.append(e_index[i])
            e_len_clean.append(item)

    m_clean_np = np.array(m_len_clean, dtype=float)
    e_clean_np = np.array(e_len_clean, dtype=float)

    return m_clean_np, m_index_clean, e_clean_np, e_index_clean

def gen_mask(r_basic):

    return _gen_lens(r_basic, 1, 0.1, 0, 0.1, 0, 0.1, 0)

def write_len_paras_v1():

    excel_name = '/data1/rong/code/Eyes/data/lens_paras.xlsx'

    m_clean_np, m_index_clean, e_clean_np, e_index_clean = load_clean_len_excel(excel_name)
    
    base_path = '/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/base_with_len'
    base_filenames_m = [os.path.join(base_path, item) for item in m_index_clean]
    base_filenames_e = [os.path.join(base_path, item) for item in e_index_clean]

    record = os.path.join('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据', 'len_record.txt')
    f_record = open(record, 'w')

    for tar_index, filename in tqdm(enumerate(base_filenames_m)):
        
        try:

            baseline, _ = load_raw_csv(filename)
            lens, pix_of_per_length = gen_mask(m_clean_np[tar_index, 3]/2)
            lens_in = lens[:,:,0:1]
            # lens_in = lens.sum(axis=2, keepdims=True)
            lens_in = align_lens(lens_in, pix_of_per_length)
            _, baseline, lens_in =processmergematrix(baseline, baseline, lens_in)

            # plt.imsave('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/base.png', baseline[:, :, 0])
            # plt.imsave('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/len.png', lens_in[:, :, 0])

            # fig1 = plt.figure(1)
            # plt.imshow(baseline[:, :, 0])
            # plt.colorbar()
            # fig1.savefig('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/base.png')

            # fig2 = plt.figure(2)
            # plt.imshow(lens_in[:, :, 0])
            # plt.colorbar()
            # fig2.savefig('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/len.png')

            paras_dir = os.path.join(base_path, m_index_clean[tar_index], 'Len_paras')
            os.makedirs(paras_dir, exist_ok= True)
            paras_file = os.path.join(paras_dir, 'v1.txt')
            f_record.write(paras_file + '\n')

            # r_basic, T_basic, r_reverse, T_reverse, r_pos, T_pos, r_side, T_side

            with open(paras_file, 'w') as f:
                variable = 'r_basic'
                value = m_clean_np[tar_index, 3]/2
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_basic'
                value = np.sum(baseline[:,:,0:1] * lens_in) / np.sum(lens_in)
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_reverse'
                value = m_clean_np[tar_index, 4]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_reverse'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_1'
                value = m_clean_np[tar_index, 5]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_1'
                value = 43500 / m_clean_np[tar_index, 0]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_2'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_2'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_side'
                value = m_clean_np[tar_index, 6]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_side'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))
        
        except:

            print('File {} may not exist!'.format(filename))


    for tar_index, filename in tqdm(enumerate(base_filenames_e)):
        
        try:

            baseline, _ = load_raw_csv(filename)
            lens, pix_of_per_length = gen_mask(e_clean_np[tar_index, 3]/2)
            lens_in = lens[:,:,0:1]
            # lens_in = lens.sum(axis=2, keepdims=True)
            lens_in = align_lens(lens_in, pix_of_per_length)
            _, baseline, lens_in =processmergematrix(baseline, baseline, lens_in)

            paras_dir = os.path.join(base_path, e_index_clean[tar_index], 'Len_paras')
            os.makedirs(paras_dir, exist_ok= True)
            paras_file = os.path.join(paras_dir, 'v1.txt')
            f_record.write(paras_file + '\n')

            # r_basic, T_basic, r_reverse, T_reverse, r_pos, T_pos, r_side, T_side

            with open(paras_file, 'w') as f:
                variable = 'r_basic'
                value = e_clean_np[tar_index, 3]/2
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_basic'
                value = np.sum(baseline[:,:,0:1] * lens_in) / np.sum(lens_in)
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_reverse'
                value = e_clean_np[tar_index, 4]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_reverse'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_1'
                value = e_clean_np[tar_index, 5]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_1'
                value = 435 / e_clean_np[tar_index, 0]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_2'
                value = e_clean_np[tar_index, 6]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_2'
                value = 435 / e_clean_np[tar_index, 0] - 1.5
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_side'
                value = e_clean_np[tar_index, 7]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_side'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))
        
        except:

            print('File {} may not exist!'.format(filename))

    f_record.close()

def load_clean_len_excel_air(excel_name):
    workbook = openpyxl.load_workbook(excel_name)
    M_len_sheet = workbook['Sheet1']
    m_len = []
    m_index = []

    M_stop = 425

    # B, C, D, E
    # 定位弧曲率    总直径  光学区曲率半径	光学区直径
    for _, i in enumerate(M_len_sheet.values):
        m_index.append('{}_OD'.format(i[0]))
        m_index.append('{}_OS'.format(i[0]))
        m_len.append(i[1:5])
        m_len.append(i[5:9])

    m_len_clean = []
    m_index_clean = []

    # E, F, G, H, I, J, K
    # 定位弧曲率	降幅	总直径	光学区直径	反转弧宽度	定位弧宽度	周边弧宽度
    for i, item in enumerate(m_len[0:(2*M_stop + 1)]):
        if (isinstance(item[0], float) or isinstance(item[0], int)) and (isinstance(item[1], float) or isinstance(item[1], int)) and \
        (isinstance(item[2], float) or isinstance(item[2], int)) and (isinstance(item[3], float) or isinstance(item[3], int)):
            m_index_clean.append(m_index[i])
            m_len_clean.append(item)

    m_clean_np = np.array(m_len_clean, dtype=float)

    return m_clean_np, m_index_clean

def write_len_paras_all_v1():

    excel_name = '/data1/rong/code/Eyes/data/lens_paras.xlsx'
    air_excel_name = '/data1/rong/code/Eyes/data/air.xlsx'

    m_clean_np, m_index_clean, e_clean_np, e_index_clean = load_clean_len_excel(excel_name)
    a_clean_np, a_index_clean = load_clean_len_excel_air(air_excel_name)
    
    base_path = '/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/base_with_len'
    base_filenames_m = [os.path.join(base_path, item) for item in m_index_clean]
    base_filenames_e = [os.path.join(base_path, item) for item in e_index_clean]
    base_filenames_a = [os.path.join(base_path, item) for item in a_index_clean]

    record = os.path.join('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据', 'len_record_all.txt')
    f_record = open(record, 'w')

    for tar_index, filename in tqdm(enumerate(base_filenames_a)):
        
        try:

            baseline, _ = load_raw_csv(filename)

            paras_dir = os.path.join(base_path, a_index_clean[tar_index], 'Len_paras')
            os.makedirs(paras_dir, exist_ok= True)
            paras_file = os.path.join(paras_dir, 'v1.txt')
            f_record.write(paras_file + '\n')

            # r_basic, T_basic, r_reverse, T_reverse, r_pos_1, T_pos_1, r_pos_2, T_pos_2, r_side, T_side

            with open(paras_file, 'w') as f:
                variable = 'r_basic'
                value = a_clean_np[tar_index, 3]/2
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_basic'
                value = a_clean_np[tar_index, 2]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_reverse'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_reverse'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_1'
                value = a_clean_np[tar_index, 1]/2 - a_clean_np[tar_index, 3]/2
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_1'
                value = 435 / a_clean_np[tar_index, 0]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_2'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_2'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_side'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_side'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))
        
        except:

            print('File {} may not exist!'.format(filename))

    for tar_index, filename in tqdm(enumerate(base_filenames_m)):
        
        try:

            baseline, _ = load_raw_csv(filename)
            lens, pix_of_per_length = gen_mask(m_clean_np[tar_index, 3]/2)
            lens_in = lens[:,:,0:1]
            # lens_in = lens.sum(axis=2, keepdims=True)
            lens_in = align_lens(lens_in, pix_of_per_length)
            _, baseline, lens_in =processmergematrix(baseline, baseline, lens_in)

            # plt.imsave('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/base.png', baseline[:, :, 0])
            # plt.imsave('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/len.png', lens_in[:, :, 0])

            # fig1 = plt.figure(1)
            # plt.imshow(baseline[:, :, 0])
            # plt.colorbar()
            # fig1.savefig('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/base.png')

            # fig2 = plt.figure(2)
            # plt.imshow(lens_in[:, :, 0])
            # plt.colorbar()
            # fig2.savefig('/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/所有按月整理的数据/len.png')

            paras_dir = os.path.join(base_path, m_index_clean[tar_index], 'Len_paras')
            os.makedirs(paras_dir, exist_ok= True)
            paras_file = os.path.join(paras_dir, 'v1.txt')
            f_record.write(paras_file + '\n')

            # r_basic, T_basic, r_reverse, T_reverse, r_pos, T_pos, r_side, T_side

            with open(paras_file, 'w') as f:
                variable = 'r_basic'
                value = m_clean_np[tar_index, 3]/2
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_basic'
                value = np.sum(baseline[:,:,0:1] * lens_in) / np.sum(lens_in)
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_reverse'
                value = m_clean_np[tar_index, 4]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_reverse'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_1'
                value = m_clean_np[tar_index, 5]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_1'
                value = 43500 / m_clean_np[tar_index, 0]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_2'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_2'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_side'
                value = m_clean_np[tar_index, 6]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_side'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))
        
        except:

            print('File {} may not exist!'.format(filename))


    for tar_index, filename in tqdm(enumerate(base_filenames_e)):
        
        try:

            baseline, _ = load_raw_csv(filename)
            lens, pix_of_per_length = gen_mask(e_clean_np[tar_index, 3]/2)
            lens_in = lens[:,:,0:1]
            # lens_in = lens.sum(axis=2, keepdims=True)
            lens_in = align_lens(lens_in, pix_of_per_length)
            _, baseline, lens_in =processmergematrix(baseline, baseline, lens_in)

            paras_dir = os.path.join(base_path, e_index_clean[tar_index], 'Len_paras')
            os.makedirs(paras_dir, exist_ok= True)
            paras_file = os.path.join(paras_dir, 'v1.txt')
            f_record.write(paras_file + '\n')

            # r_basic, T_basic, r_reverse, T_reverse, r_pos, T_pos, r_side, T_side

            with open(paras_file, 'w') as f:
                variable = 'r_basic'
                value = e_clean_np[tar_index, 3]/2
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_basic'
                value = np.sum(baseline[:,:,0:1] * lens_in) / np.sum(lens_in)
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_reverse'
                value = e_clean_np[tar_index, 4]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_reverse'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_1'
                value = e_clean_np[tar_index, 5]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_1'
                value = 435 / e_clean_np[tar_index, 0]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_pos_2'
                value = e_clean_np[tar_index, 6]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_pos_2'
                value = 435 / e_clean_np[tar_index, 0] - 1.5
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'r_side'
                value = e_clean_np[tar_index, 7]
                f.write('{} {:.4f}\n'.format(variable, value))

                variable = 'T_side'
                value = 0
                f.write('{} {:.4f}\n'.format(variable, value))
        
        except:

            print('File {} may not exist!'.format(filename))

    f_record.close()

        
if __name__ == '__main__':

    # write_len_paras_v1()
    write_len_paras_all_v1()