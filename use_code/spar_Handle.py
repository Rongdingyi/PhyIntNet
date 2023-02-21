import openpyxl
import os
import numpy as np
from skimage.transform import resize
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

def load():

    dir_name = '/data1/rong/code/Eyes/data/spar'
    after_writer = []
    
    for i in range(25):
        after_filenames = os.listdir(os.path.join(dir_name, str(i)))
        for j in after_filenames:
            x0 = j.split('.')[0]
            after_writer.append(str(i) + ',' + x0)

    return after_writer


def write_all():

    after = load()

    len_file = '/data1/rong/code/Eyes/data/txts/src/1000_with_lens.txt'
    len_filename = []
    with open(len_file, 'r') as f:
        for line in f:
            line_ = line.strip()
            len_filename.append(line_)

    len_filename = set(len_filename)

    record = os.path.join('/data1/rong/code/Eyes/data/txts/src/1000_with_lens_spar.txt')
    f_record = open(record, 'w')
    for j in after:
        if j in len_filename:
            f_record.write(j + '\n')
    f_record.close()
    
        
if __name__ == '__main__':

    write_all()