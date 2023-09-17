import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn

def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.fl = PerceptualLoss([2,2,2], [0.6,0.3,0.1])
        self.l1 = torch.nn.L1Loss()
    def forward(self, x, y, mask):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        # loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps))) + self.fl(x,y)
        loss = self.l1(x*mask,y*mask)
        return loss

class MaskLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(MaskLoss, self).__init__()
        self.eps = eps
        self.fl = PerceptualLoss([2,2,2], [0.6,0.3,0.1])
        self.mse = torch.nn.MSELoss()
    def forward(self, x, y, mask):
        diff = (x - y)*mask
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        x = x.repeat([1,3,1,1])
        y = y.repeat([1,3,1,1])
        # loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))+ self.fl(x*mask,y*mask)
        loss = self.mse(x*mask,y*mask)+ self.fl(x*mask,y*mask) + torch.mean(torch.sqrt((diff * diff)+ (self.eps*self.eps)))
        return loss

class ValidLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self):
        super(ValidLoss, self).__init__()
        self.basic = torch.nn.MSELoss()

    def forward(self, gt, pred):
        gt_ = gt.clone()
        mask = (gt_ > 0).float()
        loss = self.basic(mask*gt, mask*pred)
        return loss

class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).cuda() 
        #VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end
        assert len(blocks) <= 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        vgg = vgg.cuda() 

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]
        # self.device = device
    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # inputs = F.normalize(inputs, mean, std)
        # targets = F.normalize(targets, mean, std)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = torch.tensor(0.0).cuda() 
        
        # compare their weighted loss
        for lhs, rhs, w in zip(input_features, target_features, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            # print(lhs, rhs, w)
            # print(F.mse_loss(lhs, rhs) * w)
            loss += F.mse_loss(lhs, rhs) * w


        return loss

class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()
        
def perceptual_loss(x, y):
    F.mse_loss(x, y)
    
def PerceptualLoss(blocks, weights):
    return FeatureLoss(perceptual_loss, blocks, weights)