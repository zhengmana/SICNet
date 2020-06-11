import torch.utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage,storage_offset,size,stride,requires_grad,backward_hooks):
        tensor=torch._utils._rebuild_tensor(storage,storage_offset,size,stride)
        tensor.requires_grad=requires_grad
        tensor._backward_hooks=backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2=_rebuild_tensor_v2
#与 torch 相关的头文件
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
from torchvision.utils import save_image
#其他头文件
import sys
import os
import time
import numpy as np
import cv2
import argparse
import yaml
import json
import random
import math
import copy
from tqdm import tqdm
from easydict import EasyDict as edict
import scipy.io as sio
#可视化相关文件
from logger import Logger
logger = Logger('./tensorboardlogs_stagethree/train')
loggertest = Logger('./tensorboardlogs_stagethree/test')

parser = argparse.ArgumentParser(description='Training code')
parser.add_argument('--config', default='nanconfig.yaml', type=str, help='yaml config file')
args = parser.parse_args()
CONFIG = edict(yaml.load(open(args.config, 'r')))
print('==> CONFIG is: \n', CONFIG, '\n')

if CONFIG.IS_TRAIN:
    LOGDIR = '%s/%s_%d' % (CONFIG.LOGS.LOG_DIR, CONFIG.NAME, int(time.time()))
    LOGDIR_TEST = '%s/%s_%d' % (CONFIG.LOGS.LOG_DIR_TEST, CONFIG.NAME, int(time.time()))
    SNAPSHOTDIR = '%s/%s_%d' % (CONFIG.LOGS.SNAPSHOT_DIR, CONFIG.NAME, int(time.time()))
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(LOGDIR_TEST):
        os.makedirs(LOGDIR_TEST)
    if not os.path.exists(SNAPSHOTDIR):
        os.makedirs(SNAPSHOTDIR)

def to_varabile(arr, requires_grad=False, is_cuda=True):
    if type(arr) == np.ndarray:
        tensor = torch.from_numpy(arr)
    else:
        tensor = arr
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

#Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#左右数据集的均值rgb格式
MEAN_LEFT_var = to_varabile(np.array(CONFIG.DATASET.MEAN_LEFT, dtype=np.float32)[:, np.newaxis, np.newaxis], requires_grad=False,is_cuda=True)
MEAN_RIGHT_var = to_varabile(np.array(CONFIG.DATASET.MEAN_RIGHT, dtype=np.float32)[:, np.newaxis, np.newaxis], requires_grad=False,is_cuda=True)
#一些超参的设置
class Param:
    image_size=256
    content_weight=1
    gan_weight=0.0004
    tv_weight=1.0
    weight_decay=0.00
    G_learning_rate=0.0002
    D_learing_rate=0.00002
#数据格式转换
def to_np(x):
    return x.cpu().data.numpy()
#动态调节生成器和鉴别器的学习率
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 20:
        lr = 0.0005
    elif epoch > 20 and epoch <= 40:
        lr = 0.0001
    elif epoch > 40 and epoch <= 60:
        lr = 0.00005
    elif epoch > 60 and epoch <= 80:
        lr = 0.00001
    else:
        lr = 0.000005
    log = " ** new learning rate: %.9f (for gene)" % (lr)
    print(log)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def adjust_learning_rate_D(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 20:
        lr = 0.00001
    elif epoch > 20 and epoch <= 40:
        lr = 0.000005
    elif epoch > 40 and epoch <= 60:
        lr = 0.000001
    elif epoch > 60 and epoch <= 80:
        lr = 0.0000005
    else:
        lr = 0.0000001
    log = " ** new learning rate: %.9f (for dis)" % (lr)
    print(log)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def adjust_learning_rate_D_right(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 20:
        lr = 0.00001
    elif epoch > 20 and epoch <= 40:
        lr = 0.000005
    elif epoch > 40 and epoch <= 60:
        lr = 0.000001
    elif epoch > 60 and epoch <= 80:
        lr = 0.0000005
    else:
        lr = 0.0000001
    log = " ** new learning rate: %.9f (for disright)" % (lr)
    print(log)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def print_gpu_status(name=None):
    mem_cache = torch.cuda.memory_cached()
    mem_cache_max = torch.cuda.max_memory_cached()
    mem_alloc = torch.cuda.memory_allocated()
    mem_alloc_max = torch.cuda.max_memory_allocated()
    print("%sGPU memory cached: %.3fMB / %.3fMB , allocated: %.3fMB / %.3fMB"
          % (name+": " if name is not None else name,
             mem_cache / 1024 / 1024, mem_cache_max / 1024 / 1024,
             mem_alloc / 1024 / 1024, mem_alloc_max / 1024 / 1024)
          )

def print_model_size(model, input1=None, input2=None, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params size in memory: {:4f}MB'.format(model._get_name(), para * type_size / 1000 / 1000))

    if input1 is None and input2 is None:
        return
    out_sizes = []
    out1, out2 = model(input1, input2)
    out_sizes.append(np.array(out1.size()))
    out_sizes.append(np.array(out2.size()))
    # input1_ = input1.clone()
    # input1_.requires_grad_(requires_grad=False)
    #
    # mods = list(model.modules())
    #
    # for i in range(1, len(mods)):
    #     m = mods[i]
    #     if isinstance(m, nn.ReLU):
    #         if m.inplace:
    #             continue
    #     out = m(input1_)
    #     out_sizes.append(np.array(out.size()))
    #     input1_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))

######################################################################################################################
# "Stereo Globally and Locally Consistent Image Completion" Model
######################################################################################################################
#loss的定义
#loss tv子函数定义主要想平滑像素值
def cal_tv(image):
    temp=image.clone()
    temp[:,:,:Param.image_size-1,:]=image[:,:,1:,:]
    re=((image-temp)**2).mean()
    temp=image.clone()
    temp[:,:,:,Param.image_size-1]=image[:,:,:,1:]
    re +=((image-temp)**2).mean()
    return re

def l1_loss(self,pred,gt):
    loss = nn.L1Loss()(pred, gt)
    return loss

def smooth_l1_loss(self,pred,gt):
    loss=nn.SmoothL1Loss()(pred,gt)
    return loss

def bce_loss(self, pred, gt):
    loss = nn.BCEWithLogitsLoss()(pred, gt)
    return loss



def warp_layer(inputs, offset):
    shape = inputs.shape
    batch_size = shape[0]
    channels = shape[1]
    height = shape[2]
    width = shape[3]
    # print batch_size, channels, height, width
    inputs = torch.flip(inputs, [3])
    offset = torch.flip(offset, [3])
    width_f = float(width)
    height_f = float(height)

    # making a grid first

    index_x = torch.linspace(0.0, width_f - 1.0, width).cuda()
    index_x = index_x.repeat(height, 1)
    # print index_x.size(), offset.size()
    index_x = index_x + offset
    index_x = index_x.clamp(min=0.0, max=width_f - 1.0)
    x_floor_f = index_x.floor().repeat(1, channels, 1, 1)
    x_ceil_f = x_floor_f + 1
    x_ceil_f = x_ceil_f
    x_floor = x_floor_f.int()
    x_ceil = x_ceil_f.clamp(min=0.0, max=width_f - 1.0).int()
    # print inputs.size(), x_floor.size(), index_x.size()
    pix_floor = torch.gather(inputs.float(), 3, x_floor.long())
    pix_ceil = torch.gather(inputs.float(), 3, x_ceil.long())
    weight_floor = x_ceil_f - index_x
    weight_ceil = index_x - x_floor_f
    # print weight_ceil.dtype, pix_ceil.dtype
    output = weight_ceil * pix_ceil + weight_floor * pix_floor
    output= torch.flip(output, [3])
    return output

import numpy as np
from PIL import Image
from scipy.signal import convolve2d



#
def calc_loss(completion,relative, block_num, mask_bbox, offset):
    m_h = mask_bbox[:, 3] - mask_bbox[:, 1]
    m_w = mask_bbox[:, 2] - mask_bbox[:, 0]
    batch_size = relative.shape[0]
    im_h = relative.shape[2]
    im_w = relative.shape[3]
    range_h = torch.from_numpy(np.array(m_h / block_num, dtype=np.int32)).float().cuda()
    range_w = torch.from_numpy(np.array(m_w / block_num, dtype=np.int32)).float().cuda()


    constant1 = torch.from_numpy(np.array([1], dtype=np.int32)).float().cuda()
    costs_sum = torch.from_numpy(np.array([0], dtype=np.int32)).float().cuda()

    for k in range(batch_size):
        masks = []
        w_i = torch.from_numpy(np.array([0], dtype=np.int32)).float().cuda()
        for i in range(block_num):
            h_j = torch.from_numpy(np.array([0], dtype=np.int32)).float().cuda()
            for j in range(block_num):
                mask_part = np.zeros((im_h, im_w), dtype=np.float32)
                x1 = mask_bbox[k, 0] + w_i * range_w[k]
                y1 = mask_bbox[k, 1] + h_j * range_h[k]
                x2 = x1 + range_w[k]
                y2 = y1 + range_h[k]
                x1_ = int(x1.cpu().numpy()[0])
                y1_ = int(y1.cpu().numpy()[0])
                x2_ = int(x2.cpu().numpy()[0])
                y2_ = int(y2.cpu().numpy()[0])
                mask_part[y1_: y2_+1, x1_:x2_+1] = 1.
                #import cv2
                #cv2.imwrite(str(i) + str(j) + '.png', mask_part * 255)
                mask_part = mask_part[np.newaxis, :, :]
                masks.append(mask_part)
                h_j = h_j + constant1
            w_i = w_i + constant1
        masks = torch.from_numpy(np.array(masks)).float().cuda()
        comp = torch.unsqueeze(completion[k], dim=0).repeat(block_num * block_num, 1, 1, 1)
        rela = torch.unsqueeze(relative[k], dim=0).repeat(block_num * block_num, 1, 1, 1)
        off = torch.unsqueeze(offset[k], dim=0).repeat(block_num * block_num, 1, 1, 1)
        fake_left = warp_layer(rela, off)
        save_img = fake_left.cpu().numpy()
        img = save_img[0].transpose(1, 2, 0)
        cost = calc_dis_loss(comp,fake_left, masks)
        costs_sum = costs_sum + cost
    chushu = batch_size * block_num * block_num

    return costs_sum / torch.from_numpy(np.array(chushu, dtype=np.int32)).float().cuda(), img

def calc_dis_loss(leftcompletion, fake_left, mask):
    '''covariance and similirity'''
    # cov loss

    leftcompletion_part = torch.mul(leftcompletion, mask)
    x_c = torch.unsqueeze(torch.mean(leftcompletion_part, dim=3), -1)
    output_part = torch.mul(fake_left, mask)
    y_r = torch.unsqueeze(torch.mean(output_part, dim=3), -1)
    menumerator = torch.sum(torch.mul(leftcompletion_part - x_c, output_part - y_r))
    denominator = torch.sqrt(torch.sum(torch.mul(torch.pow(leftcompletion_part - x_c, 2), torch.pow(output_part - y_r, 2))))
    cost = torch.div(menumerator, denominator)
    # similirity loss
    # sim_loss = torch.mean(compute_ssim(completion, output))
    return cost#, sim_loss

def calc_dis_loss_r(completion_r, x, mask_r, offset_r):
    '''covariance and similirity'''
    # fli2p
    relative = x[:, :, :, ::-1]
    offset = offset_r[:, :, :, ::-1]
    # cov loss
    output = warp_layer(relative, offset)
    output = output[:,:,::-1]
    completion_part = torch.mul(completion_r, mask_r)
    x_c = torch.unsqueeze(torch.mean(completion_part, dim=3), -1)
    output_part = torch.mul(output, mask_r)
    y_r = torch.unsqueeze(torch.mean(output_part, dim=3), -1)
    menumerator = torch.sum(torch.mul(completion_part - x_c, output_part - y_r))
    denominator = torch.sqrt(torch.sum(torch.multiply(torch.pow(completion_part - x_c, 2), torch.pow(output_part - y_r, 2))))
    cost = torch.div(menumerator, denominator)
    return cost

def AffineAlignOp(features, idxs, aligned_height, aligned_width, Hs):
    def _transform_matrix(Hs, w, h):
        _Hs = np.zeros(Hs.shape, dtype=np.float32)
        for i, H in enumerate(Hs):
            H0 = np.concatenate((H, np.array([[0, 0, 1]])), axis=0)
            A = np.array([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]])
            A_inv = np.array([[w / 2.0, 0, w / 2.0], [0, h / 2.0, h / 2.0], [0, 0, 1]])
            H0 = A.dot(H0).dot(A_inv)
            H0 = np.linalg.inv(H0)
            _Hs[i] = H0[:-1]
        return _Hs

    bz, C_feat, H_feat, W_feat = features.size()
    N = len(idxs)
    feature_select = features[idxs]  # (N, feature_channel, feature_size, feature_size)
    Hs_new = _transform_matrix(Hs, w=W_feat, h=H_feat)  # return (N, 2, 3)
    Hs_var = Variable(torch.from_numpy(Hs_new), requires_grad=False).cuda()
    flow = F.affine_grid(theta=Hs_var, size=(N, C_feat, H_feat, W_feat)).float().cuda()
    flow = flow[:, :aligned_height, :aligned_width, :]
    rois = F.grid_sample(feature_select, flow, mode='bilinear', padding_mode='border')  # 'zeros' | 'border'
    return rois

def CropAlignOp(feature_var, rois_var, aligned_height, aligned_width, spatial_scale):
    rois_np = rois_var.data.cpu().numpy()
    affinematrixs_feat = []
    for roi in rois_np:
        x1, y1, x2, y2 = roi * float(spatial_scale)
        matrix = np.array([[aligned_width / (x2 - x1), 0, -aligned_width / (x2 - x1) * x1],
                           [0, aligned_height / (y2 - y1), -aligned_height / (y2 - y1) * y1]
                           ])
        affinematrixs_feat.append(matrix)
    affinematrixs_feat = np.array(affinematrixs_feat)
    feature_rois = AffineAlignOp(feature_var, np.array(range(rois_var.size(0))),
                                 aligned_height, aligned_width, affinematrixs_feat)
    return feature_rois
# 定义卷积层
class ConvBnRelu(nn.Module):
    def __init__(self, inp_dim, out_dim,
                 kernel_size=3, stride=1, dilation=1, group=1,
                 bias=True, bn=True, relu=True):
        super(ConvBnRelu, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, (kernel_size - 1) // 2 + (dilation - 1), dilation,
                              group, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
#定义反卷积层
class DeconvBnRelu(nn.Module):
    def __init__(self, inp_dim, out_dim,
                 kernel_size=4, stride=2,
                 bias=True, bn=True, relu=True):
        super(DeconvBnRelu, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.ConvTranspose2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#生成器模型
class GLCIC_G(nn.Module):
    def __init__(self, bias_in_conv=True, pretrainfile=None):
        super(GLCIC_G, self).__init__()
        self.conv1_1 = ConvBnRelu(4, 64, kernel_size=5, stride=1, bias=bias_in_conv)
        self.conv1_2 = ConvBnRelu(64, 128, kernel_size=3, stride=2, bias=bias_in_conv)
        self.conv1_3 = ConvBnRelu(128, 128, kernel_size=3, stride=1, bias=bias_in_conv)

        self.conv2_1 = ConvBnRelu(128, 256, kernel_size=3, stride=2, bias=bias_in_conv)
        self.conv2_2 = ConvBnRelu(256, 256, kernel_size=3, stride=1, bias=bias_in_conv)
        self.conv2_3 = ConvBnRelu(256, 256, kernel_size=3, stride=1, bias=bias_in_conv)

        self.conv3_1 = ConvBnRelu(256, 256, kernel_size=3, dilation=2, stride=1, bias=bias_in_conv)
        self.conv3_2 = ConvBnRelu(256, 256, kernel_size=3, dilation=4, stride=1, bias=bias_in_conv)
        self.conv3_3 = ConvBnRelu(256, 256, kernel_size=3, dilation=8, stride=1, bias=bias_in_conv)
        self.conv3_4 = ConvBnRelu(256, 256, kernel_size=3, dilation=16, stride=1, bias=bias_in_conv)

        self.conv4_1 = ConvBnRelu(256, 256, kernel_size=3, stride=1, bias=bias_in_conv)
        self.conv4_2 = ConvBnRelu(256, 256, kernel_size=3, stride=1, bias=bias_in_conv)

        self.conv_merge_l = ConvBnRelu(512, 256, kernel_size=1, stride=1)
        self.conv_merge_r = ConvBnRelu(512, 256, kernel_size=1, stride=1)

        self.decoder1_1 = DeconvBnRelu(256, 128, kernel_size=4, stride=2, bias=bias_in_conv)
        self.decoder1_2 = ConvBnRelu(128, 128, kernel_size=3, stride=1, bias=bias_in_conv)

        self.decoder2_1 = DeconvBnRelu(128, 64, kernel_size=4, stride=2, bias=bias_in_conv)
        self.decoder2_2 = ConvBnRelu(64, 32, kernel_size=3, stride=1, bias=bias_in_conv)
        self.decoder2_3 = ConvBnRelu(32, 3, kernel_size=3, stride=1, bias=bias_in_conv, bn=False, relu=False)

        self.decoder1_1r = DeconvBnRelu(256, 128, kernel_size=4, stride=2, bias=bias_in_conv)
        self.decoder1_2r = ConvBnRelu(128, 128, kernel_size=3, stride=1, bias=bias_in_conv)

        self.decoder2_1r = DeconvBnRelu(128, 64, kernel_size=4, stride=2, bias=bias_in_conv)
        self.decoder2_2r = ConvBnRelu(64, 32, kernel_size=3, stride=1, bias=bias_in_conv)
        self.decoder2_3r = ConvBnRelu(32, 3, kernel_size=3, stride=1, bias=bias_in_conv, bn=False, relu=False)
        self.init(pretrainfile)

    def init(self, pretrainfile=None):
        if pretrainfile is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, .1)
                    m.bias.data.zero_()

        elif 'completionnet_places2.t7' in pretrainfile:
            mapping = {'conv1_1.conv': 0, 'conv1_1.bn': 1, 'conv1_2.conv': 3, 'conv1_2.bn': 4, 'conv1_3.conv': 6,
                       'conv1_3.bn': 7, 'conv2_1.conv': 9, 'conv2_1.bn': 10, 'conv2_2.conv': 12, 'conv2_2.bn': 13,
                       'conv2_3.conv': 15, 'conv2_3.bn': 16, 'conv3_1.conv': 18, 'conv3_1.bn': 19, 'conv3_2.conv': 21,
                       'conv3_2.bn': 22, 'conv3_3.conv': 24, 'conv3_3.bn': 25, 'conv3_4.conv': 27, 'conv3_4.bn': 28,
                       'conv4_1.conv': 30, 'conv4_1.bn': 31, 'conv4_2.conv': 33, 'conv4_2.bn': 34,
                       'decoder1_1.conv': 36, 'decoder1_1.bn': 37, 'decoder1_2.conv': 39, 'decoder1_2.bn': 40,
                       'decoder2_1.conv': 42, 'decoder2_1.bn': 43, 'decoder2_2.conv': 45, 'decoder2_2.bn': 46,
                       'decoder2_3.conv': 48,  'decoder1_1r.conv': 36, 'decoder1_1r.bn': 37, 'decoder1_2r.conv': 39, 'decoder1_2r.bn': 40,
                       'decoder2_1r.conv': 42, 'decoder2_1r.bn': 43, 'decoder2_2r.conv': 45, 'decoder2_2r.bn': 46,
                       'decoder2_3r.conv': 48}
            from torch.utils.serialization import load_lua
            pretrain = load_lua(pretrainfile).model
            pretrained_dict = {}
            for key, mapidx in mapping.items():
                if '.conv' in key:
                    pretrained_dict[key + '.weight'] = pretrain.modules[mapidx].weight
                    pretrained_dict[key + '.bias'] = pretrain.modules[mapidx].bias
                elif '.bn' in key:
                    pretrained_dict[key + '.weight'] = pretrain.modules[mapidx].weight
                    pretrained_dict[key + '.bias'] = pretrain.modules[mapidx].bias
                    pretrained_dict[key + '.running_var'] = pretrain.modules[mapidx].running_var
                    pretrained_dict[key + '.running_mean'] = pretrain.modules[mapidx].running_mean
            model_dict = self.state_dict()
            print('==> [netG] load official weight as pretrain. init %d/%d layers.' % (
            len(pretrained_dict), len(model_dict)))
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            # model_dict.update(pretrained_dict)
            # self.load_state_dict(pretrained_dict)
            model_dict.update(new_dict)
            self.load_state_dict(model_dict)

        else:
            pretrain_param = torch.load(pretrainfile, map_location=lambda storage, loc: storage)
            model_dict = self.state_dict()
            model_dict.update(pretrain_param)
            self.load_state_dict(model_dict)
            print('==> [netG] load self-train weight as pretrain.')

    def forward(self, inputl,inputr):
        x = self.conv1_1(inputl)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        feature_left_init = x

        mn = self.conv1_1(inputr)
        mn = self.conv1_2(mn)
        mn = self.conv1_3(mn)
        mn = self.conv2_1(mn)
        mn = self.conv2_2(mn)
        mn = self.conv2_3(mn)
        mn = self.conv3_1(mn)
        mn = self.conv3_2(mn)
        mn = self.conv3_3(mn)
        mn = self.conv3_4(mn)
        mn = self.conv4_1(mn)
        mn = self.conv4_2(mn)
        feature_right_init = mn
        mn = torch.cat([feature_left_init, feature_right_init], 1)
        x_merge_l = self.conv_merge_l(mn)
        mn_merge_r = self.conv_merge_r(mn)

        x = self.decoder1_1(x_merge_l)
        x = self.decoder1_2(x)
        x = self.decoder2_1(x)
        x = self.decoder2_2(x)
        x = self.decoder2_3(x)
        leftcompletion = torch.sigmoid(x)

        mn = self.decoder1_1r(mn_merge_r)
        mn = self.decoder1_2r(mn)
        mn = self.decoder2_1r(mn)
        mn = self.decoder2_2r(mn)
        mn = self.decoder2_3r(mn)
        rightcompletion = torch.sigmoid(mn)
        return leftcompletion, rightcompletion
#左图鉴别器模型
class GLCIC_D_left(nn.Module):
    def __init__(self, bias_in_conv=True, pretrainfile=None):
        super(GLCIC_D_left, self).__init__()
        # local D
        self.local_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_fc = nn.Linear(8192, 1024)
        # global D
        self.global_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv6 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_fc = nn.Linear(8192, 1024)
        # after concat
        self.fc = nn.Linear(2048, 1)

        self.init(pretrainfile)

    def init(self, pretrainfile=None):
        if pretrainfile is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, .1)
                    m.bias.data.zero_()
        else:
            self.load_state_dict(torch.load(pretrainfile, map_location=lambda storage, loc: storage))
            print('==> [netD_left] load self-train weight as pretrain.')

    def _forward_local(self, input):
        x = self.local_conv1(input)
        x = self.local_conv2(x)
        x = self.local_conv3(x)
        x = self.local_conv4(x)
        x = self.local_conv5(x)
        x = x.view(x.size(0), -1)
        x = self.local_fc(x)
        return x

    def _forward_global(self, input):
        x = self.global_conv1(input)
        x = self.global_conv2(x)
        x = self.global_conv3(x)
        x = self.global_conv4(x)
        x = self.global_conv5(x)
        x = self.global_conv6(x)
        x = x.view(x.size(0), -1)
        x = self.global_fc(x)
        return x

    def forward(self, input_local, input_global):
        x_local = self._forward_local(input_local)
        x_global = self._forward_global(input_global)
        x = torch.cat([x_local, x_global], 1)
        x = self.fc(x)
        return x

#右图鉴别器模型
class GLCIC_D_right(nn.Module):
    def __init__(self, bias_in_conv=True, pretrainfile=None):
        super(GLCIC_D_right, self).__init__()
        # local D
        self.local_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_fc = nn.Linear(8192, 1024)
        # global D
        self.global_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv6 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_fc = nn.Linear(8192, 1024)
        # after concat
        self.fc = nn.Linear(2048, 1)

        self.init(pretrainfile)

    def init(self, pretrainfile=None):
        if pretrainfile is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, .1)
                    m.bias.data.zero_()
        else:
            self.load_state_dict(torch.load(pretrainfile, map_location=lambda storage, loc: storage))
            print('==> [netD_right] load self-train weight as pretrain.')

    def _forward_local(self, input):
        x = self.local_conv1(input)
        x = self.local_conv2(x)
        x = self.local_conv3(x)
        x = self.local_conv4(x)
        x = self.local_conv5(x)
        x = x.view(x.size(0), -1)
        x = self.local_fc(x)
        return x

    def _forward_global(self, input):
        x = self.global_conv1(input)
        x = self.global_conv2(x)
        x = self.global_conv3(x)
        x = self.global_conv4(x)
        x = self.global_conv5(x)
        x = self.global_conv6(x)
        x = x.view(x.size(0), -1)
        x = self.global_fc(x)
        return x

    def forward(self, input_local, input_global):
        x_local = self._forward_local(input_local)
        x_global = self._forward_global(input_global)
        x = torch.cat([x_local, x_global], 1)
        x = self.fc(x)
        return x



######################################################################################################################
# Dataset: kitti
######################################################################################################################

class MyDataset(object):
    def __init__(self, ImageDir_left,istrain=True):
        self.istrain = istrain
        self.imgdir_left = ImageDir_left
        self.imglist_left = os.listdir(ImageDir_left)
        print('==> Load left Dataset: \n', {'left dataset': ImageDir_left, 'istrain:': istrain , 'len': self.__len__()}, '\n')
        assert istrain == CONFIG.IS_TRAIN

    def __len__(self):
        return len(self.imglist_left)
    def __getitem__(self, idx):
                return self.loadImage(idx)

    def loadImage(self, idx):
        if self.imgdir_left == CONFIG.DATASET.VALDIR_LEFT :
            path_test = os.path.join(self.imgdir_left, self.imglist_left[idx])
            image_test = cv2.imread(path_test)
            image_test = image_test[:, :, ::-1]
            image_test = cv2.resize(image_test, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES),
                               interpolation=cv2.INTER_LINEAR)
            input_test = (image_test.astype(np.float32) / 255.0 - CONFIG.DATASET.MEAN_LEFT)
            input_test = input_test.transpose(2, 0, 1)
            # bbox_c_test, mask_c_test, _ = self.randommaskhaha(image_test.shape[0], image_test.shape[1])
            mask_c_test = cv2.imread('%s/%s' % ('/home/zhengmana/Desktop/testaaa/maskleft/', 'mask' + self.imglist_left[idx].replace('jpg', 'png')),cv2.IMREAD_GRAYSCALE).astype(np.float32)
            # mask_c_test = cv2.imread('%s/%s' % ('/media/zhengmana/mn/zero/maskleft/', self.imglist_left[idx].replace('jpg', 'png')),cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask_c_test = mask_c_test[np.newaxis, :, :]
            mask_c_test[mask_c_test >= 1] = 1.0
            mask_c_test[mask_c_test < 1] = 0.0

            # 加载右图,为了后续左右特征图的融合，所以加载的右图是和左图相同名字的一个batch
            path_right_test = os.path.join('/media/zhengmana/mn/KITTI_nana/val/right/', os.listdir('/media/zhengmana/mn/KITTI_nana/val/right/')[idx])
            # path_right_test = os.path.join('/media/zhengmana/mn/KITTI_nana/train/right/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/right/')[idx])
            image_right_test = cv2.imread(path_right_test)
            image_right_test = image_right_test[:, :, ::-1]
            image_right_test = cv2.resize(image_right_test, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES),
                               interpolation=cv2.INTER_LINEAR)
            input_right_test = (image_right_test.astype(np.float32) / 255.0 - CONFIG.DATASET.MEAN_RIGHT)
            input_right_test = input_right_test.transpose(2, 0, 1)
            # bbox_c_right_test, mask_c_right_test, _ = self.randommaskhaha(image_right_test.shape[0], image_right_test.shape[1])
            mask_c_right_test = cv2.imread('%s/%s' % ('/home/zhengmana/Desktop/testaaa/maskright/', 'mask'+self.imglist_left[idx].replace('jpg', 'png')),cv2.IMREAD_GRAYSCALE).astype(np.float32)
            # mask_c_right_test = cv2.imread('%s/%s' % ('/media/zhengmana/mn/zero/maskright/', self.imglist_left[idx].replace('jpg', 'png')),cv2.IMREAD_GRAYSCALE).astype(np.float32)
            #mask_c = cv2.resize(mask_c, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES),
                                #interpolation=cv2.INTER_NEAREST)
            mask_c_right_test = mask_c_right_test[np.newaxis, :, :]
            mask_c_right_test[mask_c_right_test >= 1] = 1.0
            mask_c_right_test[mask_c_right_test < 1] = 0.0
            return np.float32(input_test), np.float32(mask_c_test), np.int32(idx), np.float32(input_right_test), np.float32(mask_c_right_test)
        else:
            path = os.path.join(self.imgdir_left, self.imglist_left[idx])
            image = cv2.imread(path)
            image = image[:, :, ::-1]
            image = cv2.resize(image, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES),
                               interpolation=cv2.INTER_LINEAR)
            input = (image.astype(np.float32) / 255.0 - CONFIG.DATASET.MEAN_LEFT)
            input = input.transpose(2, 0, 1)

            path_mask_left = os.path.join('/media/zhengmana/mn/zero/maskleft/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/right/')[idx])
            mask_c_2 = cv2.imread(path_mask_left, 0)
            mask_c_2 = np.array(mask_c_2)
            mask_c = np.array(mask_c_2 / 255)
            mask_c = mask_c[np.newaxis, :, :]
            ret, binary = cv2.threshold(mask_c_2, 127, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            p1 = min(contours[0][:, :, 0])[0]       # x1
            q1 = min(contours[0][:, :, 1])[0]       # y1
            p2 = max(contours[0][:, :, 0])[0]       # x2
            q2 = max(contours[0][:, :, 1])[0]       # y2
            # 中心点坐标
            x_center = int((p1 + p2) / 2)
            y_center = int((q1 + q2) / 2)
            x1 = x_center - CONFIG.DATASET.LOCAL_RES / 2
            y1 = y_center - CONFIG.DATASET.LOCAL_RES / 2

            bbox_c = np.array([x1, y1, x1 + CONFIG.DATASET.LOCAL_RES, y1 + CONFIG.DATASET.LOCAL_RES],
                                    dtype=np.int32)
            mask_bbox_c = np.array([[p1, q1, p2 + 1, q2 + 1]], dtype=np.int32)

            # mask_c = cv2.imread('%s/%s' % ('', self.imglist_left[idx].replace('jpg', 'png')),
            #                          cv2.IMREAD_GRAYSCALE).astype(np.float32)
            # mask_c = mask_c[np.newaxis, :, :]
            # mask_c[mask_c >= 1] = 1.0
            # mask_c[mask_c < 1] = 0.0
            # bbox_c, mask_c, mask_bbox_c = self.randommaskhaha(image.shape[0], image.shape[1])
            # bbox_d, mask_d, mask_bbox_d = self.randommaskhaha(image.shape[0], image.shape[1])

            # 加载右图,为了后续左右特征图的融合，所以加载的右图是和左图相同名字的一个batch
            path_right = os.path.join('/media/zhengmana/mn/KITTI_nana/train/right/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/right/')[idx])
            image_right = cv2.imread(path_right)
            image_right = image_right[:, :, ::-1]
            image_right = cv2.resize(image_right, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES),
                                     interpolation=cv2.INTER_LINEAR)
            input_right = (image_right.astype(np.float32) / 255.0 - CONFIG.DATASET.MEAN_RIGHT)
            input_right = input_right.transpose(2, 0, 1)

            # bbox_c_right, mask_c_right, mask_bbox_c_right = self.randommaskhaha(image_right.shape[0], image_right.shape[1])
            # bbox_d_right, mask_d_right, _ = self.randommaskhaha(image_right.shape[0], image_right.shape[1])
            path_mask_right = os.path.join('/media/zhengmana/mn/zero/maskright/', os.listdir('/media/zhengmana/mn/KITTI_nana/train/right/')[idx])
            mask_c_right_2 = cv2.imread(path_mask_right, 0)
            mask_c_right_2 = np.array(mask_c_right_2)
            mask_c_right = np.array(mask_c_right_2 / 255)
            mask_c_right = mask_c_right[np.newaxis, :, :]
            ret, binary = cv2.threshold(mask_c_right_2, 127, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            p1 = min(contours[0][:, :, 0])
            q1 = min(contours[0][:, :, 1])
            p2 = max(contours[0][:, :, 0])
            q2 = max(contours[0][:, :, 1])
            # 中心点坐标
            x_center = int((p1 + p2)/ 2)
            y_center = int((q1 + q2)/ 2)
            x1 = x_center - CONFIG.DATASET.LOCAL_RES / 2
            y1 = y_center - CONFIG.DATASET.LOCAL_RES / 2

            bbox_c_right = np.array([x1, y1, x1 + CONFIG.DATASET.LOCAL_RES, y1 + CONFIG.DATASET.LOCAL_RES], dtype=np.int32)
            mask_bbox_c_right = np.array([[p1, q1, p2+1, q2+1]], dtype=np.int32)

            # 加载视差图真值
            path_dis = os.path.join('/media/zhengmana/mn/zero/leftdisp/',os.listdir('/media/zhengmana/mn/zero/leftdisp/')[idx])
            offset = cv2.imread(path_dis, 0)
            offset = np.array([offset])


            # img = img * 255.0
            # img = cv2.imwrite('./fakeleft.png', img)
            # offset = cv2.resize(image_dis, (256, 256),interpolation=cv2.INTER_LINEAR)
            # offset = offset * 256 / 640
            # path_dis_save = os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/offset/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx])
            # cv2.imwrite(path_dis_save, offset)
            return np.float32(input), np.float32(mask_c), bbox_c, mask_bbox_c, np.float32(input_right), np.float32(mask_c_right), bbox_c_right, mask_bbox_c_right, idx, offset
    def randommaskhaha(self, height, width):
        x1, y1 = np.random.randint(0, CONFIG.DATASET.INPUT_RES - CONFIG.DATASET.LOCAL_RES + 1, 2)
        x2, y2 = np.array([x1, y1]) + CONFIG.DATASET.LOCAL_RES
        w, h = np.random.randint(CONFIG.DATASET.HOLE_MIN, CONFIG.DATASET.HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, CONFIG.DATASET.LOCAL_RES - w)
        q1 = y1 + np.random.randint(0, CONFIG.DATASET.LOCAL_RES - h)
        p2 = p1 + w
        q2 = q1 + h
        mask = np.zeros((height, width), dtype=np.float32)
        mask[q1:q2 + 1, p1:p2 + 1] = 1.0
        bbox = np.array([x1, y1, x1 + CONFIG.DATASET.LOCAL_RES, y1 + CONFIG.DATASET.LOCAL_RES], dtype=np.int32)
        mask_bbox = [p1, q1, p2+1, q2+1]  # format: x1, y1, x2, y2
        return bbox, mask[np.newaxis, :, :], mask_bbox


######################################################################################################################
#Training
######################################################################################################################
def train(dataLoader, testdataLoader,model_G, model_D, model_D_right, optimizer_G, optimizer_D, optimizer_D_right, epoch):
    batch_time = AverageMeter('batch_time')
    losses_G = AverageMeter('losses_G')
    losses_G_right = AverageMeter('losses_G_right')
    losses_D = AverageMeter('losses_D')
    losses_D_right = AverageMeter('losses_D_right')

    losses_G_L2 = AverageMeter('losses_G_L2')
    losses_G_disp = AverageMeter('losses_G_disp')
    losses_G_real = AverageMeter('losses_G_real')
    losses_D_real = AverageMeter('losses_D_real')
    losses_D_fake = AverageMeter('losses_D_fake')

    losses_G_L2_right = AverageMeter('losses_G_L2_right')
    losses_G_real_right = AverageMeter('losses_G_real_right')
    losses_D_real_right = AverageMeter('losses_D_real_right')
    losses_D_fake_right = AverageMeter('losses_D_fake_right')

    # switch to train mode
    model_G.train()
    model_D.train()

    if epoch <= CONFIG.TRAIN_G_EPOCHES:

        end = time.time()
        for i, data in enumerate(dataLoader):

            input3ch, mask_c, bbox_c, input3ch_right, mask_c_right, bbox_c_right  = data
            input4ch = torch.cat([input3ch * (1 - mask_c), mask_c], dim=1)
            input3ch_var = to_varabile(input3ch, requires_grad=False, is_cuda=True) + MEAN_LEFT_var
            input4ch_var = to_varabile(input4ch, requires_grad=True, is_cuda=True)
            bbox_c_var = to_varabile(bbox_c, requires_grad=False, is_cuda=True)
            mask_c_var = to_varabile(mask_c, requires_grad=True, is_cuda=True)

            input4ch_right = torch.cat([input3ch_right * (1 - mask_c_right), mask_c_right], dim=1)
            input3ch_right_var = to_varabile(input3ch_right, requires_grad=False, is_cuda=True) + MEAN_RIGHT_var
            input4ch_right_var = to_varabile(input4ch_right, requires_grad=True, is_cuda=True)
            bbox_c_right_var = to_varabile(bbox_c_right, requires_grad=False, is_cuda=True)
            mask_c_right_var = to_varabile(mask_c_right, requires_grad=True, is_cuda=True)

            out_G,out_G_right,feature_left,feature_right = model_G(input4ch_var, input4ch_right_var)
            loss_G_L2 = nn.MSELoss()(out_G, input3ch_var)
            losses_G_L2.update(loss_G_L2.item(), input3ch.size(0))
            loss_G_L2_right = nn.MSELoss()(out_G_right, input3ch_right_var)


            losses_G_L2_right.update(loss_G_L2_right.item(), input3ch_right.size(0))

            completion = (input3ch_var) * (1 - mask_c_var) + out_G * mask_c_var
            completion_right=(input3ch_right_var) * (1 - mask_c_right_var) + out_G_right * mask_c_right_var
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # youhua
            loss_G = loss_G_L2
            loss_G_right = loss_G_L2_right
            losses_G.update(loss_G.item(), input3ch.size(0))
            losses_G_right.update(loss_G_right.item(), input3ch_right.size(0))
            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            loss_G_right.backward(retain_graph=True)
            optimizer_G.step()

            # ========================= Log ================================================
            step = epoch * len(dataLoader) + i
            # (1) Log the scalar values
            info = {'loss_G_L2': loss_G_L2.item(), 'loss_G_L2_right': loss_G_L2_right.item()}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)

            # feature_left0 = feature_left
            # feature_left1 = feature_left
            # feature_left2 = feature_left
            # feature_left_gray = feature_left
            # feature_left_rgb = feature_left
            #
            # info = {'feature_left_0': to_np(feature_left0.view(-1, 256, 64, 64)[:5])}
            # for tag, feature_left0 in info.items():
            #     logger.image_summary(tag, feature_left0, step, 0)
            #
            # info = {'feature_left_1': to_np(feature_left1.view(-1, 256, 64, 64)[:5])}
            # for tag, feature_left1 in info.items():
            #     logger.image_summary(tag, feature_left1, step, 1)
            #
            # info = {'feature_left_2': to_np(feature_left2.view(-1, 256, 64, 64)[:5])}
            # for tag, feature_left2 in info.items():
            #     logger.image_summary(tag, feature_left2, step, 2)
            #
            # info = {'feature_left_gray': to_np(feature_left_gray.view(-1, 1, 64, 64)[:5])}
            # for tag, feature_left_gray in info.items():
            #     logger.image_summary(tag, feature_left_gray, step)
            #
            # info = {'feature_left_rgb': to_np(feature_left_rgb.view(-1, 3, 64, 64)[:5])}
            # for tag, feature_left_rgb in info.items():
            #     logger.image_summary(tag, feature_left_rgb, step)

        print('Epoch: [{0}][{1}/{2}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''G {loss_G.val:.4f} ({loss_G.avg:.4f})\t'
            'G_right {loss_G_right.val:.4f} ({loss_G_right.avg:.4f})\t'
            'D {loss_D.val:.4f} ({loss_D.avg:.4f})\t''G_L2 {G_L2.val:.4f} ({G_L2.avg:.4f})\t''G_real {G_real.val:.4f} ({G_real.avg:.4f})\t'
            'D_fake {D_fake.val:.4f} ({D_fake.avg:.4f})\t''D_real {D_real.val:.4f} ({D_real.avg:.4f})\t'.format(
                epoch, i, len(dataLoader), batch_time=batch_time, loss_G=losses_G, loss_G_right=losses_G_right,
                loss_D=losses_D, G_L2=losses_G_L2, G_real=losses_G_real,D_fake=losses_D_fake, D_real=losses_D_real))
        vis = torch.cat([input3ch_var * (1 - mask_c_var), completion, input3ch_right_var * (1 - mask_c_right_var),completion_right], dim=0)
        save_image(vis.data, os.path.join(LOGDIR, 'epoch%d_vis.png' % (epoch)), nrow=input3ch.size(0),
                   padding=2, normalize=True, range=None, scale_each=True, pad_value=0)

        # switch to eval mode
        model_G.eval()
        sum = 0.0
        l1sum = 0.0
        iter = 0
        sum_right = 0.0
        l1sum_right = 0.0
        iter_right = 0
        for j, data_test in enumerate(testdataLoader):
            input3ch_test, mask_c_test, idxs_test, input3ch_right_test, mask_c_right_test = data_test
            input4ch_test = torch.cat([input3ch_test * (1 - mask_c_test), mask_c_test], dim=1)
            input3ch_test_var = to_varabile(input3ch_test, requires_grad=False, is_cuda=True) + MEAN_LEFT_var
            input4ch_test_var = to_varabile(input4ch_test, requires_grad=False, is_cuda=True)
            mask_c_test_var = to_varabile(mask_c_test, requires_grad=False, is_cuda=True)

            input4ch_right_test = torch.cat([input3ch_right_test * (1 - mask_c_right_test), mask_c_right_test], dim=1)
            input3ch_right_test_var = to_varabile(input3ch_right_test, requires_grad=False,
                                                  is_cuda=True) + MEAN_RIGHT_var
            input4ch_right_test_var = to_varabile(input4ch_right_test, requires_grad=True, is_cuda=True)
            mask_c_right_test_var = to_varabile(mask_c_right_test, requires_grad=True, is_cuda=True)

            out_G_test, out_G_right_test,feature_left_test,feature_right_test = model_G(input4ch_test_var, input4ch_right_test_var)


            loss_G_L2_test = torch.nn.MSELoss()(out_G_test, input3ch_test_var)
            loss_G_l1_test = torch.nn.L1Loss()(out_G_test, input3ch_test_var)
            iter = iter + 1
            sum = sum + float(loss_G_L2_test)
            l1sum = l1sum + float(loss_G_l1_test)
            meanl2error = sum / iter
            meanl1error = l1sum / iter

            loss_G_L2_right_test = torch.nn.MSELoss()(out_G_right_test, input3ch_right_test_var)
            loss_G_l1_right_test = torch.nn.L1Loss()(out_G_right_test, input3ch_right_test_var)
            iter_right = iter_right + 1
            sum_right = sum_right + float(loss_G_L2_right_test)
            l1sum_right = l1sum_right + float(loss_G_l1_right_test)
            meanl2error_right = sum_right / iter_right
            meanl1error_right = l1sum_right / iter_right

            completion_test = (input3ch_test_var) * (1 - mask_c_test_var) + out_G_test * mask_c_test_var
            completion_right_test = (input3ch_right_test_var) * (1 - mask_c_right_test_var) + out_G_right_test * mask_c_right_test_var
            # ========================= Log_test ============================================
            step_test = epoch * len(testdataLoader) + j
            # (1) Log the test scalar values
            info_test = {'loss_G_L2': loss_G_L2_test.item(), 'loss_G_L2_right': loss_G_L2_right_test.item()}

            for tag_test, value_test in info_test.items():
                loggertest.scalar_summary(tag_test, value_test, step_test)

            # (2) Log the test images
            # info_test = {'images_left_test': to_np(out_G_test.view(-1, 3, 256, 256)[:3])}
            # for tag_test, images_left_test in info_test.items():
            #     loggertest.image_summary(tag_test, images_left_test, step_test)
            #
            # info_test = {'images_right_test': to_np(out_G_right_test.view(-1, 3, 256, 256)[:3])}
            # for tag_test, images_right_test in info_test.items():
            #     loggertest.image_summary(tag_test, images_right_test, step_test)

            # info_test = {'feature_left_test': to_np(feature_left_test.view(-1, 3, 64, 64)[:3])}
            # for tag_test, feature_left_test in info_test.items():
            #     loggertest.image_summary(tag_test, feature_left_test, step)
            #
            # info_test = {'feature_right_test': to_np(feature_right_test.view(-1, 3, 64, 64)[:3])}
            # for tag_test, feature_right_test in info_test.items():
            #     loggertest.image_summary(tag_test, feature_right_test, step_test)

        print("l2: %.8f, l1: %.8f, l2_right: %.8f, l1_right: %.8f " % (meanl2error, meanl1error, meanl2error_right, meanl1error_right))
        vis_test = torch.cat([input3ch_test_var * (1 - mask_c_test_var), completion_test,
                                  input3ch_right_test_var * (1 - mask_c_right_test_var),completion_right_test], dim=0)
        save_image(vis_test.data, os.path.join(LOGDIR_TEST, 'epoch%d_vis.png' % (epoch)),
                       nrow=input3ch_test.size(0), padding=2, normalize=True, range=None, scale_each=True, pad_value=0)


    elif epoch <= CONFIG.TRAIN_D_EPOCHES:
        end = time.time()
        for i, data in enumerate(dataLoader):
            input3ch, mask_c, bbox_c, input3ch_right, mask_c_right, bbox_c_right  = data
            input4ch = torch.cat([input3ch * (1 - mask_c), mask_c], dim=1)
            input3ch_var = to_varabile(input3ch, requires_grad=False, is_cuda=True) + MEAN_LEFT_var
            input4ch_var = to_varabile(input4ch, requires_grad=True, is_cuda=True)
            bbox_c_var = to_varabile(bbox_c, requires_grad=False, is_cuda=True)
            mask_c_var = to_varabile(mask_c, requires_grad=True, is_cuda=True)

            input4ch_right = torch.cat([input3ch_right * (1 - mask_c_right), mask_c_right], dim=1)
            input3ch_right_var = to_varabile(input3ch_right, requires_grad=False, is_cuda=True) + MEAN_RIGHT_var
            input4ch_right_var = to_varabile(input4ch_right, requires_grad=True, is_cuda=True)
            bbox_c_right_var = to_varabile(bbox_c_right, requires_grad=False, is_cuda=True)
            mask_c_right_var = to_varabile(mask_c_right, requires_grad=True, is_cuda=True)

            out_G, out_G_right, feature_left, feature_right = model_G(input4ch_var, input4ch_right_var)
            # loss_G_L2 = nn.MSELoss()(out_G, input3ch_var)
            # losses_G_L2.update(loss_G_L2.item(), input3ch.size(0))
            # loss_G_L2_right = nn.MSELoss()(out_G_right, input3ch_right_var)
            # losses_G_L2_right.update(loss_G_L2_right.item(), input3ch_right.size(0))

            completion = (input3ch_var) * (1 - mask_c_var) + out_G * mask_c_var
            completion_right = (input3ch_right_var) * (1 - mask_c_right_var) + out_G_right * mask_c_right_var

            local_completion = CropAlignOp(completion, bbox_c_var,
                                           CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)
            local_input3ch = CropAlignOp(input3ch_var, bbox_c_var,
                                         CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)

            out_D_fake = model_D(local_completion, completion)
            loss_D_fake = nn.BCEWithLogitsLoss()(out_D_fake, torch.zeros_like(out_D_fake))
            losses_D_fake.update(loss_D_fake.item(), input3ch.size(0))

            out_D_real = model_D(local_input3ch, input3ch_var)
            loss_D_real = nn.BCEWithLogitsLoss()(out_D_real, torch.ones_like(out_D_real))
            losses_D_real.update(loss_D_real.item(), input3ch.size(0))

            # out_G_real = out_D_fake
            # loss_G_real = nn.BCEWithLogitsLoss()(out_G_real, torch.ones_like(out_G_real))
            # losses_G_real.update(loss_G_real.item(), input3ch.size(0))

            # right view
            local_completion_right = CropAlignOp(completion_right, bbox_c_right_var,
                                           CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)
            local_input3ch_right = CropAlignOp(input3ch_right_var, bbox_c_right_var,
                                         CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)

            out_D_fake_right = model_D_right(local_completion_right, completion_right)
            loss_D_fake_right = nn.BCEWithLogitsLoss()(out_D_fake_right, torch.zeros_like(out_D_fake_right))
            losses_D_fake_right.update(loss_D_fake_right.item(), input3ch_right.size(0))

            out_D_real_right = model_D_right(local_input3ch_right, input3ch_right_var)
            loss_D_real_right = nn.BCEWithLogitsLoss()(out_D_real_right, torch.ones_like(out_D_real_right))
            losses_D_real_right.update(loss_D_real_right.item(), input3ch_right.size(0))

            # out_G_real_right = out_D_fake_right
            # loss_G_real_right = nn.BCEWithLogitsLoss()(out_G_real_right, torch.ones_like(out_G_real_right))
            # losses_G_real.update(loss_G_real_right.item(), input3ch_right.size(0))

            # youhua
            loss_D = loss_D_fake + loss_D_real
            losses_D.update(loss_D.item(), input3ch.size(0))
            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            loss_D_right = loss_D_fake_right + loss_D_real_right
            losses_D_right.update(loss_D_right.item(), input3ch_right.size(0))
            optimizer_D_right.zero_grad()
            loss_D_right.backward(retain_graph=True)
            optimizer_D_right.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            step_two = epoch * len(dataLoader) + i
            # (1) Log the scalar values
            info_two = {'loss_D': loss_D.item(), 'loss_D_right': loss_D_right.item()}

            for tagtwo, valuetwo in info_two.items():
                logger.scalar_summary(tagtwo, valuetwo, step_two)

        print(
            'Epoch: [{0}][{1}/{2}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''G {loss_G.val:.4f} ({loss_G.avg:.4f})\t'
            'G_right {loss_G_right.val:.4f} ({loss_G_right.avg:.4f})\t''D {loss_D.val:.4f} ({loss_D.avg:.4f})\t''D_right {loss_D_right.val:.4f} ({loss_D_right.avg:.4f})\t'
            'G_L2 {G_L2.val:.4f} ({G_L2.avg:.4f})\t''G_real {G_real.val:.4f} ({G_real.avg:.4f})\t'
            'D_fake {D_fake.val:.4f} ({D_fake.avg:.4f})\t''D_real {D_real.val:.4f} ({D_real.avg:.4f})\t''D_fake_right {D_fake_right.val:.4f} ({D_fake_right.avg:.4f})\t''D_real_right {D_real_right.val:.4f} ({D_real_right.avg:.4f})\t'.format(
                epoch, i, len(dataLoader), batch_time=batch_time, loss_G=losses_G, loss_G_right=losses_G_right,
                loss_D=losses_D, loss_D_right=losses_D_right, G_L2=losses_G_L2, G_real=losses_G_real, D_fake=losses_D_fake, D_real=losses_D_real,
                D_fake_right=losses_D_fake_right, D_real_right=losses_D_real_right))
        vis = torch.cat([input3ch_var * (1 - mask_c_var), completion, input3ch_right_var * (1 - mask_c_right_var),
                         completion_right], dim=0)
        save_image(vis.data, os.path.join(LOGDIR, 'epoch%d_vis.png' % (epoch)), nrow=input3ch.size(0),
                   padding=2, normalize=True, range=None, scale_each=True, pad_value=0)


    else:
        print("Training......")
        end = time.time()
        save_data_old = None
        save_data = None
        for i, data in enumerate(dataLoader):
            save_data_old = save_data
            save_data = data
            input3ch, mask_c, bbox_c, mask_bbox_c, input3ch_right, mask_c_right, bbox_c_right, mask_bbox_c_right, idx, offset = data
            input4ch = torch.cat([input3ch * (1 - mask_c), mask_c], dim=1)
            input3ch_var = to_varabile(input3ch, requires_grad=False, is_cuda=True) + MEAN_LEFT_var
            input4ch_var = to_varabile(input4ch, requires_grad=True, is_cuda=True)
            bbox_c_var = to_varabile(bbox_c, requires_grad=False, is_cuda=True)
            mask_c_var = to_varabile(mask_c, requires_grad=True, is_cuda=True)
            mask_bbox_c_var= to_varabile(mask_bbox_c, requires_grad=False, is_cuda=True)

            input4ch_right = torch.cat([input3ch_right * (1 - mask_c_right), mask_c_right], dim=1)
            input3ch_right_var = to_varabile(input3ch_right, requires_grad=False, is_cuda=True) + MEAN_RIGHT_var
            input4ch_right_var = to_varabile(input4ch_right, requires_grad=True, is_cuda=True)
            bbox_c_right_var = to_varabile(bbox_c_right, requires_grad=False, is_cuda=True)
            mask_c_right_var = to_varabile(mask_c_right, requires_grad=True, is_cuda=True)

            out_G, out_G_right = model_G(input4ch_var, input4ch_right_var)
            #generator loss
            loss_G_L2 = nn.MSELoss()(out_G * mask_c_var, input3ch_var * mask_c_var)
            losses_G_L2.update(loss_G_L2.item(), input3ch.size(0))
            loss_G_L2_right = nn.MSELoss()(out_G_right, input3ch_right_var)
            losses_G_L2_right.update(loss_G_L2_right.item(), input3ch_right.size(0))

            completion = (input3ch_var) * (1 - mask_c_var) + out_G * mask_c_var
            completion_right = (input3ch_right_var) * (1 - mask_c_right_var) + out_G_right * mask_c_right_var
            save_data.append(completion)
            save_data.append(completion_right)

            local_completion = CropAlignOp(completion, bbox_c_var,
                                           CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)
            local_input3ch = CropAlignOp(input3ch_var, bbox_c_var,
                                         CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)

            out_D_fake = model_D(local_completion, completion)
            loss_D_fake = nn.BCEWithLogitsLoss()(out_D_fake, torch.zeros_like(out_D_fake))
            losses_D_fake.update(loss_D_fake.item(), input3ch.size(0))

            out_D_real = model_D(local_input3ch, input3ch_var)
            loss_D_real = nn.BCEWithLogitsLoss()(out_D_real, torch.ones_like(out_D_real))
            losses_D_real.update(loss_D_real.item(), input3ch.size(0))

            out_G_real = out_D_fake
            loss_G_real = nn.BCEWithLogitsLoss()(out_G_real, torch.ones_like(out_G_real))
            losses_G_real.update(loss_G_real.item(), input3ch.size(0))


            #求取视差
            # mask_bbox = torch.from_numpy(mask_bbox_c).float().cuda() #
            completion_cuda = torch.from_numpy(completion.cpu().detach().numpy()).float().cuda()                # [5, 3, 256, 256]
            completion_right_cuda = torch.from_numpy(completion_right.cpu().detach().numpy()).float().cuda()    # [5, 3, 256, 256]
            mask_bbox_cuda = torch.from_numpy(np.squeeze(mask_bbox_c.numpy(), axis=1)).float().cuda()    # [5, 4]
            offset_cuda = torch.from_numpy(np.squeeze(offset.numpy(), axis=1)).float().cuda()       # [5, 256, 256]

            loss_G_disp, _ = calc_loss(completion_cuda, completion_right_cuda, 7, mask_bbox_cuda, offset_cuda)
            losses_G_disp.update(loss_G_disp.item(), input3ch.size(0))

            # right view
            local_completion_right = CropAlignOp(completion_right, bbox_c_right_var,
                                                 CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)
            local_input3ch_right = CropAlignOp(input3ch_right_var, bbox_c_right_var,
                                               CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)

            out_D_fake_right = model_D_right(local_completion_right, completion_right)
            loss_D_fake_right = nn.BCEWithLogitsLoss()(out_D_fake_right, torch.zeros_like(out_D_fake_right))
            losses_D_fake_right.update(loss_D_fake_right.item(), input3ch_right.size(0))

            out_D_real_right = model_D_right(local_input3ch_right, input3ch_right_var)
            loss_D_real_right = nn.BCEWithLogitsLoss()(out_D_real_right, torch.ones_like(out_D_real_right))
            losses_D_real_right.update(loss_D_real_right.item(), input3ch_right.size(0))

            out_G_real_right = out_D_fake_right
            loss_G_real_right = nn.BCEWithLogitsLoss()(out_G_real_right, torch.ones_like(out_G_real_right))
            losses_G_real_right.update(loss_G_real_right.item(), input3ch_right.size(0))

            # youhua
            loss_G = loss_G_L2 + 0.0004  * loss_G_real + loss_G_disp
            loss_G_right = loss_G_L2_right + 0.0004  * loss_G_real_right + loss_G_disp
            # loss_G = loss_G_L2 + 0.0004  * loss_G_real
            # loss_G_right = loss_G_L2_right + 0.0004  * loss_G_real_right
            losses_G.update(loss_G.item(), input3ch.size(0))
            losses_G_right.update(loss_G_right.item(), input3ch_right.size(0))
            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            loss_G_right.backward(retain_graph=True)
            optimizer_G.step()

            loss_D = loss_D_fake + loss_D_real
            losses_D.update(loss_D.item(), input3ch.size(0))
            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            loss_D_right = loss_D_fake_right + loss_D_real_right
            losses_D_right.update(loss_D_right.item(), input3ch_right.size(0))
            optimizer_D_right.zero_grad()
            loss_D_right.backward(retain_graph=True)
            optimizer_D_right.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # vis = torch.cat([input3ch_var * (1 - mask_c_var), completion, input3ch_right_var * (1 - mask_c_right_var),
            #                  completion_right], dim=0)
            # save_image(vis.data, os.path.join(LOGDIR, 'epoch%d_vis_%d.png' % (epoch,i)), nrow=input3ch.size(0),
            #            padding=2, normalize=True, range=None, scale_each=True, pad_value=0)
            # save_image(completion.data, os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/leftcompletion/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx]))
            # save_image(completion_right.data, os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/rightcompletion/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx]))
            # save_image(mask_c_var.data, os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/maskleft/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx]))
            # save_image(mask_c_right_var.data,os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/maskright/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx]))
            # save_image(local_completion.data,os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/localleft/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx]))
            # save_image(local_completion_right.data,os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/localright/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx]))
            # save_image(input3ch_var.data,os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/gtleft/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx]))
            # save_image(input3ch_right_var.data,os.path.join('/home/zhengmana/Desktop/stereo image inpainting/logs/gtright/',os.listdir('/media/zhengmana/mn/KITTI_nana/train/dis/')[idx]))
            if i % 17 == 0:
                step_three = epoch * len(dataLoader) + i
                step_three = int(step_three / 17)
                # (1) Log the generator scalar values
                info_three = {'loss_G': loss_G.item(), 'loss_G_right': loss_G_right.item()}

                for tag_three, value_three in info_three.items():
                    logger.scalar_summary(tag_three, value_three, step_three)
                # (2) Log the discriminator scalar values
                info_three = {'loss_D': loss_D.item(), 'loss_D_right': loss_D_right.item()}

                for tagthree, valuethree in info_three.items():
                    logger.scalar_summary(tagthree, valuethree, step_three)

                # info_three = {'feature_left_0': to_np(feature_left.view(-1, 1, 64, 64)[:5])}
                # for tag, feature_left0 in info_three.items():
                #     logger.image_summary(tag, feature_left0, step_three)
                # info_three = {'feature_right_0': to_np(feature_right.view(-1, 1, 64, 64)[:5])}
                # for tag, feature_right in info_three.items():
                #     logger.image_summary(tag, feature_right, step_three)

                # info_three = {'feature_merge_l_0': to_np(feature_merge_l.view(-1, 1, 64, 64)[:5])}
                # for tag, feature_left_merge in info_three.items():
                #     logger.image_summary(tag, feature_left_merge, step_three)
                #
                # info_three = {'feature_merger_r_0': to_np(feature_merge_r.view(-1, 1, 64, 64)[:5])}
                # for tag, feature_right_merge in info_three.items():
                #     logger.image_summary(tag, feature_right_merge, step_three)
            # vis = torch.cat([input3ch_var * (1 - mask_c_var), completion, input3ch_right_var * (1 - mask_c_right_var),
            #                  completion_right], dim=0)
            # save_image(vis.data, os.path.join(LOGDIR, 'epoch%d_vis_%d.png' % (epoch,i)), nrow=input3ch.size(0),
            #            padding=2, normalize=True, range=None, scale_each=True, pad_value=0)
            torch.cuda.empty_cache()
        print_gpu_status("Train")
        print(
            'Epoch: [{0}][{1}/{2}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t''G {loss_G.val:.4f} ({loss_G.avg:.4f})\t'
            'G_right {loss_G_right.val:.4f} ({loss_G_right.avg:.4f})\t''D {loss_D.val:.4f} ({loss_D.avg:.4f})\t''D_right {loss_D_right.val:.4f} ({loss_D_right.avg:.4f})\t'
            'G_L2 {G_L2.val:.4f} ({G_L2.avg:.4f})\t''G_real {G_real.val:.4f} ({G_real.avg:.4f})\t''G_disp {G_disp.val:.4f} ({G_disp.avg:.4f})\t''G_L2_right {G_L2_right.val:.4f} ({G_L2_right.avg:.4f})\t''G_real_right {G_real_right.val:.4f} ({G_real_right.avg:.4f})\t'
            'G_disp_right {G_disp_right.val:.4f} ({G_disp_right.avg:.4f})\t''D_fake {D_fake.val:.4f} ({D_fake.avg:.4f})\t''D_real {D_real.val:.4f} ({D_real.avg:.4f})\t''D_fake_right {D_fake_right.val:.4f} ({D_fake_right.avg:.4f})\t''D_real_right {D_real_right.val:.4f} ({D_real_right.avg:.4f})\t'.format(
                epoch, i, len(dataLoader), batch_time=batch_time, loss_G=losses_G, loss_G_right=losses_G_right,
                loss_D=losses_D, loss_D_right=losses_D_right, G_L2=losses_G_L2, G_real=losses_G_real, G_disp = losses_G_disp, G_L2_right=losses_G_L2_right,
                G_real_right=losses_G_real_right,G_disp_right = losses_G_disp , D_fake=losses_D_fake, D_real=losses_D_real,
                 D_fake_right=losses_D_fake_right, D_real_right=losses_D_real_right))

        if save_data is not None and save_data[0].shape[0] < dataLoader.batch_size:
            input3ch, mask_c, bbox_c, mask_bbox_c, input3ch_right, mask_c_right, bbox_c_right, mask_bbox_c_right, idx, offset, completion, completion_right = save_data_old
            input3ch_var = to_varabile(input3ch, requires_grad=False, is_cuda=True) + MEAN_LEFT_var
            mask_c_var = to_varabile(mask_c, requires_grad=True, is_cuda=True)
            input3ch_right_var = to_varabile(input3ch_right, requires_grad=False, is_cuda=True) + MEAN_RIGHT_var
            mask_c_right_var = to_varabile(mask_c_right, requires_grad=True, is_cuda=True)

            vis = torch.cat([input3ch_var * (1 - mask_c_var), completion, input3ch_right_var * (1 - mask_c_right_var),
                         completion_right], dim=0)
            save_image(vis.data, os.path.join(LOGDIR, 'epoch%d_vis.png' % (epoch)), nrow=input3ch.size(0),
                   padding=2, normalize=True, range=None, scale_each=True, pad_value=0)
        elif save_data is not None:
            vis = torch.cat([input3ch_var * (1 - mask_c_var), completion, input3ch_right_var * (1 - mask_c_right_var),
                             completion_right], dim=0)
            save_image(vis.data, os.path.join(LOGDIR, 'epoch%d_vis.png' % (epoch)), nrow=input3ch.size(0),
                       padding=2, normalize=True, range=None, scale_each=True, pad_value=0)
        else:
            print("No data to save!!!!!!!!!!!!!!!")

        # switch to eval mode
        print("Testing......")
        model_G.eval()
        sum = 0.0
        l1sum = 0.0
        iter = 0
        sum_right = 0.0
        l1sum_right = 0.0
        iter_right = 0
        save_data_old = None
        save_data = None
        for j, data_test in enumerate(testdataLoader):
            save_data_old = save_data
            save_data = data_test
            input3ch_test, mask_c_test, idxs_test, input3ch_right_test, mask_c_right_test = data_test
            input4ch_test = torch.cat([input3ch_test * (1 - mask_c_test), mask_c_test], dim=1)
            input3ch_test_var = to_varabile(input3ch_test, requires_grad=False, is_cuda=True) + MEAN_LEFT_var
            input4ch_test_var = to_varabile(input4ch_test, requires_grad=False, is_cuda=True)
            mask_c_test_var = to_varabile(mask_c_test, requires_grad=False, is_cuda=True)

            input4ch_right_test = torch.cat([input3ch_right_test * (1 - mask_c_right_test), mask_c_right_test], dim=1)
            input3ch_right_test_var = to_varabile(input3ch_right_test, requires_grad=False,is_cuda=True) + MEAN_RIGHT_var
            input4ch_right_test_var = to_varabile(input4ch_right_test, requires_grad=True, is_cuda=True)
            mask_c_right_test_var = to_varabile(mask_c_right_test, requires_grad=True, is_cuda=True)

            out_G_test, out_G_right_test = model_G(input4ch_test_var,input4ch_right_test_var)

            loss_G_L2_test = torch.nn.MSELoss()(out_G_test, input3ch_test_var)
            loss_G_l1_test = torch.nn.L1Loss()(out_G_test, input3ch_test_var)
            iter = iter + 1
            sum = sum + float(loss_G_L2_test)
            l1sum = l1sum + float(loss_G_l1_test)
            meanl2error = sum / iter
            meanl1error = l1sum / iter

            loss_G_L2_right_test = torch.nn.MSELoss()(out_G_right_test, input3ch_right_test_var)
            loss_G_l1_right_test = torch.nn.L1Loss()(out_G_right_test, input3ch_right_test_var)
            iter_right = iter_right + 1
            sum_right = sum_right + float(loss_G_L2_right_test)
            l1sum_right = l1sum_right + float(loss_G_l1_right_test)
            meanl2error_right = sum_right / iter_right
            meanl1error_right = l1sum_right / iter_right

            completion_test = (input3ch_test_var) * (1 - mask_c_test_var) + out_G_test * mask_c_test_var
            completion_right_test = (input3ch_right_test_var) * (1 - mask_c_right_test_var) + out_G_right_test * mask_c_right_test_var
            save_data.append(completion_test)
            save_data.append(completion_right_test)
            # ========================= Log_test ============================================
            step_test = epoch * len(testdataLoader) + j
            # (1) Log the test scalar values
            info_test = {'loss_G': loss_G_L2_test.item(), 'loss_G_right': loss_G_L2_right_test.item()}

            for tag_test, value_test in info_test.items():
                loggertest.scalar_summary(tag_test, value_test, step_test)

            info_testl1 = {'loss_G_L1_test': loss_G_l1_test.item(), 'loss_G_L1_right_test': loss_G_l1_right_test.item()}

            for tag_testl1, value_testl1 in info_testl1.items():
                loggertest.scalar_summary(tag_testl1, value_testl1, step_test)

        #     info_three_test = {'feature_left_test': to_np(feature_left_test.view(-1, 1, 64, 64)[:5])}
        #     for tag, feature_left_test in info_three_test.items():
        #         loggertest.image_summary(tag, feature_left_test, step_test)
        #
        #     info_three_test = {'feature_right_test': to_np(feature_right_test.view(-1, 1, 64, 64)[:5])}
        #     for tag, feature_right_test in info_three_test.items():
        #         loggertest.image_summary(tag, feature_right_test, step_test)
        #
        #     info_three_test = {'feature_merge_l_test': to_np(feature_merge_l_test.view(-1, 1, 64, 64)[:5])}
        #     for tag, feature_left_merge_test in info_three_test.items():
        #         loggertest.image_summary(tag, feature_left_merge_test, step_test)
        #
        #     info_three_test = {'feature_merger_r_test': to_np(feature_merge_r_test.view(-1, 1, 64, 64)[:5])}
        #     for tag, feature_right_merge_test in info_three_test.items():
        #         loggertest.image_summary(tag, feature_right_merge_test, step_test)
            torch.cuda.empty_cache()
        print_gpu_status("Test")
        print("l2: %.8f, l1: %.8f, l2_right: %.8f, l1_right: %.8f " % (
            meanl2error, meanl1error, meanl2error_right, meanl1error_right))
        if save_data is not None and save_data[0].shape[0] < testdataLoader.batch_size:
            input3ch_test, mask_c_test, idxs_test, input3ch_right_test, mask_c_right_test, completion_test, completion_right_test = save_data_old
            input3ch_test_var = to_varabile(input3ch_test, requires_grad=False, is_cuda=True) + MEAN_LEFT_var
            mask_c_test_var = to_varabile(mask_c_test, requires_grad=False, is_cuda=True)
            input3ch_right_test_var = to_varabile(input3ch_right_test, requires_grad=False,
                                                  is_cuda=True) + MEAN_RIGHT_var
            mask_c_right_test_var = to_varabile(mask_c_right_test, requires_grad=True, is_cuda=True)

            vis_test = torch.cat([input3ch_test_var * (1 - mask_c_test_var), completion_test,
                                  input3ch_right_test_var * (1 - mask_c_right_test_var), completion_right_test], dim=0)
            save_image(vis_test.data, os.path.join(LOGDIR_TEST, 'epoch%d_vis.png' % (epoch)),
                       nrow=input3ch_test.size(0), padding=2, normalize=True, range=None, scale_each=True, pad_value=0)
        elif save_data is not None:
            vis_test = torch.cat([input3ch_test_var * (1 - mask_c_test_var), completion_test,
                                  input3ch_right_test_var * (1 - mask_c_right_test_var), completion_right_test], dim=0)
            save_image(vis_test.data, os.path.join(LOGDIR_TEST, 'epoch%d_vis.png' % (epoch)),
                   nrow=input3ch_test.size(0), padding=2, normalize=True, range=None, scale_each=True, pad_value=0)
        else:
            print("No test data to save!!!!!!!!!!!!!!!")



    if epoch == CONFIG.TRAIN_G_EPOCHES:
        torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, 'preG_%d.pkl' % (epoch)))

    if epoch == CONFIG.TRAIN_D_EPOCHES:
        torch.save(model_D.state_dict(), os.path.join(SNAPSHOTDIR, 'preD_%d.pkl' % (epoch)))
        torch.save(model_D_right.state_dict(), os.path.join(SNAPSHOTDIR, 'preD_RIGHT_%d.pkl' % (epoch)))

def main():
    dataset = MyDataset(ImageDir_left = CONFIG.DATASET.TRAINDIR_LEFT,istrain=True)
    testdataset = MyDataset(ImageDir_left=CONFIG.DATASET.VALDIR_LEFT, istrain=True)
    BATCHSIZE = CONFIG.SOLVER.IMG_PER_GPU * len(CONFIG.SOLVER.GPU_IDS)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True,
                                             num_workers=CONFIG.SOLVER.WORKERS, pin_memory=False)
    testdataLoader=torch.utils.data.DataLoader(testdataset, batch_size= 3, shuffle=True,
                                             num_workers=CONFIG.SOLVER.WORKERS, pin_memory=False)

    model_G = GLCIC_G(bias_in_conv=True, pretrainfile=CONFIG.INIT_G).cuda()
    model_D = GLCIC_D_left(bias_in_conv=True, pretrainfile=CONFIG.INIT_D).cuda()
    model_D_right = GLCIC_D_right(bias_in_conv=True, pretrainfile=CONFIG.INIT_D_RIGHT).cuda()

    # optimizer_G = torch.optim.Adam(model_G.parameters(), lr=5e-4,weight_decay=CONFIG.SOLVER.WEIGHTDECAY)
    # optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-5,weight_decay=CONFIG.SOLVER.WEIGHTDECAY)
    # optimizer_D_right = torch.optim.Adam(model_D_right.parameters(), lr=1e-5, weight_decay=CONFIG.SOLVER.WEIGHTDECAY)
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=5e-4)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-5)
    optimizer_D_right = torch.optim.Adam(model_D_right.parameters(), lr=1e-5)
    epoches = CONFIG.TOTAL_EPOCHES

    print_model_size(model_G)
    print_model_size(model_D)
    print_model_size(model_D_right)

    for epoch in range(epoches):
        print('===========>   [Epoch %d] training    <===========' % epoch)
        train(dataLoader,testdataLoader,model_G, model_D, model_D_right, optimizer_G, optimizer_D, optimizer_D_right,epoch)
        adjust_learning_rate(optimizer_G, epoch)
        adjust_learning_rate_D(optimizer_D, epoch)
        adjust_learning_rate_D_right(optimizer_D_right, epoch)
        if epoch % CONFIG.LOGS.SNAPSHOT_FREQ == 0:
            torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, 'G_%d.pkl' % (epoch)))
            torch.save(model_D.state_dict(), os.path.join(SNAPSHOTDIR, 'D_%d.pkl' % (epoch)))
            torch.save(model_D_right.state_dict(), os.path.join(SNAPSHOTDIR, 'D_RIGHT_%d.pkl' % (epoch)))
    torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, 'G_%d.pkl' % (epoch)))
    torch.save(model_D.state_dict(), os.path.join(SNAPSHOTDIR, 'D_%d.pkl' % (epoch)))
    torch.save(model_D_right.state_dict(), os.path.join(SNAPSHOTDIR, 'D_RIGHT_%d.pkl' % (epoch)))

def evaluate():
    testdataset = MyDataset(ImageDir_left=CONFIG.DATASET.VALDIR_LEFT, istrain=True)
    testdataLoader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False,
                                                 num_workers=CONFIG.SOLVER.WORKERS, pin_memory=False)
    model_G = GLCIC_G(bias_in_conv=True, pretrainfile=CONFIG.VAL.INIT_G).cuda()
    # switch to eval mode
    model_G.eval()
    sum = 0.0
    l1sum = 0.0
    iter = 0
    sum_right = 0.0
    l1sum_right = 0.0
    iter_right = 0
    for j, data_test in enumerate(testdataLoader):
        input3ch_test, mask_c_test, idxs_test, input3ch_right_test, mask_c_right_test = data_test
        filename = testdataset.imglist_left[idxs_test.numpy()[0]]
        input4ch_test = torch.cat([input3ch_test * (1 - mask_c_test), mask_c_test], dim=1)
        input3ch_test_var = to_varabile(input3ch_test, requires_grad=False, is_cuda=True) + MEAN_LEFT_var
        input4ch_test_var = to_varabile(input4ch_test, requires_grad=False, is_cuda=True)
        mask_c_test_var = to_varabile(mask_c_test, requires_grad=False, is_cuda=True)

        input4ch_right_test = torch.cat([input3ch_right_test * (1 - mask_c_right_test), mask_c_right_test], dim=1)
        input3ch_right_test_var = to_varabile(input3ch_right_test, requires_grad=False, is_cuda=True) + MEAN_RIGHT_var
        input4ch_right_test_var = to_varabile(input4ch_right_test, requires_grad=True, is_cuda=True)
        mask_c_right_test_var = to_varabile(mask_c_right_test, requires_grad=False, is_cuda=True)


        out_G_test, out_G_right_test = model_G(input4ch_test_var,input4ch_right_test_var)

        loss_G_L2_test = torch.nn.MSELoss()(out_G_test * mask_c_test_var, input3ch_test_var * mask_c_test_var)
        loss_G_l1_test = torch.nn.L1Loss()(out_G_test * mask_c_test_var, input3ch_test_var * mask_c_test_var)
        iter = iter + 1
        sum = sum + float(loss_G_L2_test)
        l1sum = l1sum + float(loss_G_l1_test)



        out_G_right_test_mask = out_G_right_test * mask_c_right_test_var
        input3ch_right_test_var_mask = input3ch_right_test_var * mask_c_right_test_var

        loss_G_L2_right_test = torch.nn.MSELoss()(out_G_right_test_mask, input3ch_right_test_var_mask)
        loss_G_l1_right_test = torch.nn.L1Loss()(out_G_right_test_mask, input3ch_right_test_var_mask)
        iter_right = iter_right + 1
        sum_right = sum_right + float(loss_G_L2_right_test)
        l1sum_right = l1sum_right + float(loss_G_l1_right_test)


        # completion_test = (input3ch_test_var) * (1 - mask_c_test_var) + out_G_test * mask_c_test_var
        # completion_right_test = (input3ch_right_test_var) * (1 - mask_c_right_test_var) + out_G_right_test * mask_c_right_test_var
        #
        # completion_test_np = completion_test.data.cpu().numpy().transpose((0, 2, 3, 1))[0] * 255.0
        # cv2.imwrite('%s/left%s' % ('/home/zhengmana/Desktop/testaaa/dispfusionleft/', filename), np.uint8(completion_test_np[:, :, ::-1]))
        #
        # completion_right_test_np = completion_right_test.data.cpu().numpy().transpose((0, 2, 3, 1))[0] * 255.0
        # cv2.imwrite('%s/right%s' % ('/home/zhengmana/Desktop/testaaa/dispfusionright/', filename), np.uint8(completion_right_test_np[:, :, ::-1]))
    meanl2error = sum / iter
    meanl1error = l1sum / iter
    meanl2error_right = sum_right / iter_right
    meanl1error_right = l1sum_right / iter_right
    print("l2: %.8f, l1: %.8f, l2_right: %.8f, l1_right: %.8f " % (meanl2error, meanl1error, meanl2error_right, meanl1error_right))




if __name__ == '__main__':
    if CONFIG.NAME in ['NANA', 'NANA2', 'NANA3']:
        evaluate()
    else:
        main()