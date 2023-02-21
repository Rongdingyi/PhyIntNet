import os
import numpy as np
from glob import glob

np.random.seed(1234)
txts_dir = '/data1/rong/code/Eyes/data/all_arrange_by_month/txts/src'
all_names = glob(os.path.join(txts_dir, '*.txt'))
all_txts = [os.path.join(txts_dir, name)  for name in all_names]

train_txt_dir = os.path.join(txts_dir, 'train_txts')
val_txt_dir = os.path.join(txts_dir, 'val_txts')
os.makedirs(train_txt_dir, exist_ok=True)
os.makedirs(val_txt_dir, exist_ok=True)

for txt in all_txts:

    txt_name = os.path.basename(txt)
    f_all = open(txt, 'r')
    f_train = open(os.path.join(train_txt_dir, txt_name), 'w')
    f_val = open(os.path.join(val_txt_dir, txt_name), 'w')

    for item in f_all:
        a = np.random.rand()
        if a<0.8:
            f_train.write(item)
        else:
            f_val.write(item)

    print('Split {} Done'.format(txt))
