import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
# from skimage.transform import resize
# import argparse

# parser = argparse.ArgumentParser(description='eyes')
# parser.add_argument('--env', default='_base2',type=str, help='Directory of validation images')
# args = parser.parse_args()

def psnr(target, ref):
    target_data = np.array(target)
    ref_data = np.array(ref)
#  20*math.log10(1.0/rmse)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.) )
    return 20*math.log10(np.max(target_data)/rmse)

def psnr_mask(target, ref, mask):
    target_data = np.array(target)
    ref_data = np.array(ref)
#  20*math.log10(1.0/rmse)
    diff = (ref_data - target_data)*mask
    if diff.sum() == 0:
        return 100
    else:
        diff = diff.flatten('C')
        rmse = math.sqrt(np.sum(diff ** 2.)/np.sum(mask))
        return 20*math.log10(np.max(target_data[mask!=0])/rmse)

envs = os.listdir(os.path.join('/data1/rong/code/Eyes/exps'))

# envs = ['_base_no_LBPorLEN', 'full_model_train_mask']

for env in envs:

    if env != '_pre' and env != '_bak':

        result_logs = os.listdir(os.path.join('/data1/rong/code/Eyes/exps', env))

        for cnt, log in enumerate(result_logs):

            # if not os.path.exists(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis')):
            if True:

                filename = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'opt_record.txt')
                info={}
                with open(filename, 'r') as f:
                    for line in f:
                        line_ = line.strip()
                        tmp = line_.split(':')
                        if len(tmp) == 2:
                            info[tmp[0]] = tmp[1]

                path = os.listdir(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', 'img'))
                psnr_list = []
                for i in path:
                    result_file = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', 'img', i)
                    # target_file = '/data1/rong/code/Eyes/data/可视化/结果可视化/' + i
                    target_file = '/data1/rong/code/Eyes_12-08/data/按月整理的角膜地形图数据/可视化/戴镜后可视化/' + i
                    result_png = cv2.imread(result_file)
                    target_png = cv2.imread(target_file)
                    result = np.array((result_png[:,:,0]/255.) * 5 + 7)
                    target = np.array((target_png[:,:,0]/255.) * 5 + 7) 
                    if info['train_mask'] == ' False':
                        psnr_list.append(psnr(result, target))
                    elif info['train_mask'] == ' True':
                        mask = np.load(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', 'mask', i.replace('.png', '.npy')))
                        psnr_list.append(psnr_mask(result, target, mask[0,:,:]))
                
                os.makedirs(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis'), exist_ok=True)
                print('{} {} psnr: '.format(env, cnt), np.mean(psnr_list))
                with open(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis', 'psnr.txt'), 'w') as f:
                    f.write('psnr: {:.4f}'.format(np.mean(psnr_list)))
                os.makedirs(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis', 'merge'), exist_ok=True)
                for i in path:
                    result_file = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', 'img', i)
                    len_file = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', 'len', i)
                    bar_file = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', 'bar', i)
                    # bar_file = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', 'bar', i)
                    # target_file = '/data1/rong/code/Eyes/data/可视化/结果可视化/' + i
                    # source_file = '/data1/rong/code/Eyes/data/可视化/原可视化/' + i
                    target_file = '/data1/rong/code/Eyes_12-08/data/按月整理的角膜地形图数据/可视化/戴镜后可视化/' + i
                    source_file = '/data1/rong/code/Eyes_12-08/data/按月整理的角膜地形图数据/可视化/戴镜前可视化/' + i

                    source_png = cv2.imread(source_file)
                    result_png = cv2.imread(result_file)
                    target_png = cv2.imread(target_file)
                    len_png = cv2.imread(len_file)
                    bar_png = cv2.imread(bar_file)
                    bar_png = cv2.resize(bar_png, (128*4, int(bar_png.shape[0]/bar_png.shape[1]*128*4)))
                    # sub_png = np.asarray(np.abs(np.asarray(result_png, dtype=np.int16)  - np.asarray(target_png, dtype=np.int16)), dtype=np.uint8)
                    result = np.array((result_png[:,:,0]/255.) * 5 + 7)
                    target = np.array((target_png[:,:,0]/255.) * 5 + 7)
                    if info['train_mask'] == ' False':
                        metric = psnr(result, target)
                    elif info['train_mask'] == ' True':
                        mask = np.load(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', 'mask', i.replace('.png', '.npy')))
                        metric = psnr_mask(result, target, mask[0,:,:])

                    all_file2 = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis/merge', '{}_{}.png'.format(i.split('.')[0], metric))
                    # cv2.imwrite(all_file2, np.hstack((len_png, source_png, target_png, result_png, sub_png)))
                    cv2.imwrite(all_file2, np.vstack((np.hstack((len_png, source_png, target_png, result_png)), bar_png)))

            else:
                with open(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis', 'psnr.txt'), 'r') as f:
                    for line in f:
                        line_ = line.strip()
                        print('{} {} psnr: '.format(env, cnt), line_)


# for env in envs:

#     if env != 'pre':

#         result_logs = os.listdir(os.path.join('/data1/rong/code/Eyes/exps', env))

#         for log in result_logs:

#             if not os.path.exists(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis')):
#             # if True:

#                 path = os.listdir(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results'))
#                 psnr_list = []
#                 for i in path:
#                     result_file = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', i)
#                     # target_file = '/data1/rong/code/Eyes/data/可视化/结果可视化/' + i
#                     target_file = '/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/可视化/戴镜后可视化/' + i
#                     result_png = cv2.imread(result_file)
#                     target_png = cv2.imread(target_file)  
#                     result = np.array((result_png[:,:,0]/255.) * 5 + 7)
#                     target = np.array((target_png[:,:,0]/255.) * 5 + 7) 
#                     psnr_list.append(psnr(result, target))
                
#                 os.makedirs(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis'), exist_ok=True)
#                 print('{} {} psnr: '.format(env, log), np.mean(psnr_list))
#                 with open(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis', 'psnr.txt'), 'w') as f:
#                     f.write('psnr: {:.4f}'.format(np.mean(psnr_list)))
#                 os.makedirs(os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis', 'merge'), exist_ok=True)
#                 for i in path:
#                     result_file = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', i)
#                     len_file = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results', i.replace('.png', '_len.png'))
#                     # target_file = '/data1/rong/code/Eyes/data/可视化/结果可视化/' + i
#                     # source_file = '/data1/rong/code/Eyes/data/可视化/原可视化/' + i
#                     target_file = '/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/可视化/戴镜后可视化/' + i
#                     source_file = '/data1/rong/code/Eyes/data/按月整理的角膜地形图数据/可视化/戴镜前可视化/' + i

#                     source_png = cv2.imread(source_file)
#                     result_png = cv2.imread(result_file)
#                     target_png = cv2.imread(target_file)
#                     sub_png = np.asarray(np.abs(np.asarray(result_png, dtype=np.int16)  - np.asarray(target_png, dtype=np.int16)), dtype=np.uint8)
#                     result = np.array((result_png[:,:,0]/255.) * 5 + 7)
#                     target = np.array((target_png[:,:,0]/255.) * 5 + 7) 
#                     metric = psnr(result, target)

#                     all_file2 = os.path.join('/data1/rong/code/Eyes/exps', env, log, 'results_analysis/merge', '{}_{}.png'.format(i.split('.')[0], metric))
#                     cv2.imwrite(all_file2, np.hstack((source_png, target_png, result_png, sub_png)))
            