import torch
import torch.nn as nn
import os
from collections import OrderedDict
import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_name)
# from model_exps import Uformer


def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

# def load_checkpoint(model, weights):
#     checkpoint = torch.load(weights)
#     try:
#         model.load_state_dict(checkpoint["state_dict"])
#     except:
#         state_dict = checkpoint["state_dict"]
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:] if 'module.' in k else k
#             new_state_dict[name] = v
#         model_dict = model.state_dict()
#         pretrained_dict = {k:v for k,v in state_dict.items() if k in model_dict}
#         # pretrained_dict = {k: v for k, v in state_dict.items() if (k in model_dict and 'module.LEN_proj.proj.0' not in k)}
#         state_dict.update(pretrained_dict)
#         model.load_state_dict(pretrained_dict,False)
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):

    arch = opt.arch
    print('You choose '+arch+'...')
    
    if arch == 'Uformer_E':
        from model_main import Uformer
        model_restoration = Uformer(img_size=opt.train_ps, out_chans=1,
                embed_dim=opt.embed_dim, depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                win_size=opt.win_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                patch_norm=True, use_checkpoint=False, token_projection='linear', token_mlp='leff',
                shift_flag=True, modulator=True, cross_modulator=False, use_LENmu=opt.use_LENmu, 
                time_encode=opt.time_encode, qk_mode=opt.qk_mode, predict_AL=opt.predict_AL)
    elif arch == 'UNet':
        from model_exp import UNet
        model_restoration = UNet(dim=32)
    elif arch == 'Full-Convolution':
        from model_exp import FCN32s
        model_restoration = FCN32s()
    elif arch == 'Stacked-hourglass':
        from model_exp import PoseNet
        model_restoration = PoseNet()
    else:
        raise Exception("Arch error!")

    return model_restoration