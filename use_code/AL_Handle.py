import openpyxl
import os
import numpy as np
from skimage.transform import resize
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

def load_AL_air():
    excel_name = '/data1/rong/code/Eyes/data/_src/air_AL.xlsx'
    workbook = openpyxl.load_workbook(excel_name)
    M_len_sheet = workbook['Sheet1']
    base_rows = []
    after_rows = []

    base_writer = []
    after_writer = []

    # 
    # ID	取镜时间	R_眼轴	L_眼轴	眼轴检查日期2	R_眼轴	L_眼轴	眼轴检查日期3	R_眼轴	L_眼轴  眼轴检查日期4	R_眼轴	L_眼轴
    for _, line in enumerate(M_len_sheet.values):
        
        base_rows.append([line[0] + '_OD', line[2]])
        base_rows.append([line[0] + '_OS', line[3]])

        after_rows.append([line[0]+'_OD', line[1], line[4], line[5]])
        after_rows.append([line[0]+'_OS', line[1], line[4], line[6]])

        after_rows.append([line[0]+'_OD', line[1], line[7], line[8]])
        after_rows.append([line[0]+'_OS', line[1], line[7], line[9]])

        after_rows.append([line[0]+'_OD', line[1], line[10], line[11]])
        after_rows.append([line[0]+'_OS', line[1], line[10], line[12]])

    for line in base_rows[2:]:

        try:
            value = float(line[1])
            # print(value)
            if value > 1 and value < 100:
                base_writer.append([line[0], -1, value])
        except:
            base_writer.append([line[0], -1, 0])

    for line in after_rows[6:]:

        try:
            base = line[1].split('.')
            if int(base[0]) > 1900:
                basetime = datetime(int(base[0]), int(base[1]), int(base[2]))
            else:
                basetime = datetime(int(base[0])+2000, int(base[1]), int(base[2]))
            # check = line[2].split('/')
            # checktime = datetime(int(check[0]), int(check[1], int(check[2])))
            checktime = line[2]
            month2 = round((checktime - basetime).days / 30)
            value = float(line[3])
            if value > 1 and value < 100 and month2 <= 24:
                after_writer.append([line[0], month2, value])
                # if line[0] == 'OKLB0266_OS' or line[0] == 'OKLB0266_OD':
                #     print([month2, line[0], value])

        except:
            continue

    return base_writer, after_writer

def supplement():

    b1, a1 = load_AL_air()
    # print(len(b1))
    # print('\n')
    # print(len(a1))
    all_items = b1 + a1
    iv_index = {}
    for i, item in enumerate(b1):
        iv_index[item[0]] = i

    len_file = '/data1/rong/code/Eyes/data/txts/bak/src/1000_with_lens.txt'
    len_items = []
    with open(len_file, 'r') as f:
        for line in f:
            line_ = line.strip()
            line_s = line_.split(',')
            if 'OKLB' in line_s[1]:
                len_items.append([line_s[1], int(line_s[0])])
    
    all_table = np.zeros((len(b1), 26))
    res_table = np.zeros((len(b1), 26))
    
    for item in all_items:
        all_table[iv_index[item[0]], item[1] + 1] = item[2]
    for item in a1:
        res_table[iv_index[item[0]], item[1] + 1] = item[2] - all_table[iv_index[item[0]], 0]

    aver_res = res_table.sum(axis=0)/(all_table!=0).sum(axis=0)
    aver_res[2] = 0
    x1 = all_table[:, 0:1]
    x2 = aver_res[np.newaxis, :]
    result_table = x1 + x2

    for item in len_items:
        os.makedirs('/data1/rong/code/Eyes/data/base/{}/Len_paras'.format(item[0]), exist_ok=True)
        os.makedirs('/data1/rong/code/Eyes/data/after/{}/{}/Len_paras'.format(str(item[1]), item[0]), exist_ok=True)
        base_filename = '/data1/rong/code/Eyes/data/base/{}/Len_paras/AL.txt'.format(item[0])
        after_filename = '/data1/rong/code/Eyes/data/after/{}/{}/Len_paras/AL.txt'.format(str(item[1]), item[0])
        print(base_filename, result_table[iv_index[item[0]], 0])
        print(after_filename, result_table[iv_index[item[0]], item[1] + 1])
        with open(base_filename, 'w') as f:
            f.write(str(result_table[iv_index[item[0]], 0]))
        with open(after_filename, 'w') as f:
            f.write(str(result_table[iv_index[item[0]], item[1] + 1]))

def debug():
    filename = '/data1/rong/code/Eyes/data/txts/bak/src/1000_with_lens.txt'
    len_items = []
    with open(filename, 'r') as f:
        for line in f:
            line_ = line.strip()
            line_s = line_.split(',')
            if 'OKLB' in line_s[1]:
                len_items.append([line_s[1], int(line_s[0])])
    for item in len_items:
        os.makedirs('/data1/rong/code/Eyes/data/base/{}/Len_paras'.format(item[0]), exist_ok=True)
        os.makedirs('/data1/rong/code/Eyes/data/after/{}/{}/Len_paras'.format(str(item[1]), item[0]), exist_ok=True)
        # base_filename = '/data1/rong/code/Eyes/data/base/{}/Len_paras/AL.txt'.format(item[0])
        after_filename = '/data1/rong/code/Eyes/data/after/{}/{}/Len_paras/AL.txt'.format(str(item[1]), item[0])
        # with open(base_filename, 'r') as f:
        #     f.readline()
        with open(after_filename, 'r') as f:
            al = float(f.readline())
            if al<10:
                print(item, al)
    
        
if __name__ == '__main__':

    # write_len_paras_v1()
    # write_all()
    debug()