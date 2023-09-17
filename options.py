# import os
# import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--nepoch', type=int, default=2500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='Eyes')
        parser.add_argument('--pretrain_weights',type=str, default='./Uformer_B.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='6,7', help='GPUs')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./exps',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        # args for Uformer

        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        
        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--train_dir', type=str, default ='./data',  help='dir of train data')
        parser.add_argument('--train_txt', type=str, default ='./data/txts/train.txt',  help='txt of train data')
        parser.add_argument('--val_dir', type=str, default ='./data',  help='dir of val data')
        parser.add_argument('--val_txt', type=str, default ='./data/txts/val.txt',  help='txt of val data')
        parser.add_argument('--test_dir', type=str, default ='./data',  help='dir of test data')
        parser.add_argument('--test_txt', type=str, default ='./data/txts/test.txt',  help='txt of val data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 

        # args for experiment
        parser.add_argument('--arch', type=str, default ='Uformer_E',  help='archtechture')
        parser.add_argument('--use_LBPmu', action='store_true',default=False)
        parser.add_argument('--use_LENmu', action='store_true',default=False)
        parser.add_argument('--train_mask', action='store_true',default=False)
        parser.add_argument('--len_mode', type=str, default ='origin')
        parser.add_argument('--time_encode', action='store_true',default=False)
        parser.add_argument('--qk_mode', type=str, default ='dot')
        parser.add_argument('--w1', type=float, default=1.0, help='w1')
        parser.add_argument('--w2', type=float, default=0.0, help='w2')
        parser.add_argument('--predict_AL', action='store_true',default=False)

        # extra
        # parser.add_argument('--val_init', action='store_true',default=False)

        return parser
