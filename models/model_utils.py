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
        model_dict = model.state_dict()
        # pretrained_dict = {k:v for k,v in state_dict.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in state_dict.items() if (k in model_dict and 'module.LEN_proj.proj.0' not in k)}
        # state_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict,False)
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
#         model.load_state_dict(new_state_dict)

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
