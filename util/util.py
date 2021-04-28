from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import random
import inspect, re
import numpy as np
import os
import collections
import math
from torch.autograd import Variable
import torch.nn as nn
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3,1,1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count

def binary_mask(in_mask, threshold):
    assert in_mask.dim() == 2, "mask must be 2 dimensions"

    output = torch.ByteTensor(in_mask.size())
    output = (output > threshold).float().mul_(1)

    return output

def create_gMask(gMask_opts):
    pattern = gMask_opts['pattern']
    mask_global = gMask_opts['mask_global']
    MAX_SIZE = gMask_opts['MAX_SIZE']
    fineSize = gMask_opts['fineSize']
    maxPartition=gMask_opts['maxPartition']
    if pattern is None:
        raise ValueError
    wastedIter = 0
    while True:
        x = random.randint(1, MAX_SIZE-fineSize)
        y = random.randint(1, MAX_SIZE-fineSize)
        mask = pattern[y:y+fineSize, x:x+fineSize] # need check
        area = mask.sum()*100./(fineSize*fineSize)
        if area>20 and area<maxPartition:
            break
        wastedIter += 1
    if mask_global.dim() == 3:
        mask_global = mask.expand(1, mask.size(0), mask.size(1))
    else:
        mask_global = mask.expand(1, 1, mask.size(0), mask.size(1))
    return mask_global

# inMask is tensor should be 1*1*256*256 float
# Return: ByteTensor

# get feature of mask
def cal_feat_mask(inMask, conv_layers, threshold):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    assert inMask.size(0) == 1, "the first dimension must be 1 for mask"
    inMask = inMask.float()
    convs = []
    inMask = Variable(inMask, requires_grad = False)
    for id_net in range(conv_layers):
        conv = nn.Conv2d(1,1,4,2,1, bias=False)
        conv.weight.data.fill_(1/16)
        convs.append(conv)
    lnet = nn.Sequential(*convs)
    if inMask.is_cuda:
        lnet = lnet.cuda()
    output = lnet(inMask)
    output = (output > threshold).float().mul_(1)
    output=Variable(output, requires_grad = False)
    return output.detach().byte()


# index
def cal_mask_given_mask_thred(img, mask, patch_size, stride, mask_thred):
    assert img.dim() == 3, 'img has to be 3 dimenison!'
    assert mask.dim() == 2, 'mask has to be 2 dimenison!'
    dim = img.dim()

    _, H, W = img.size(dim-3), img.size(dim-2), img.size(dim-1)   # c x 32 x 32
    
    nH = int(math.floor((H-patch_size)/stride + 1))  # 
    nW = int(math.floor((W-patch_size)/stride + 1))  # 
    
    N = nH*nW    # 32x32 -> 1024

    flag = torch.zeros(N).long()    # flag == 1 -> unknown region
    offsets_tmp_vec = torch.zeros(N).long()  # 


    nonmask_point_idx_all = torch.zeros(N).long()   # nonmask 32x32

    tmp_non_mask_idx = 0


    mask_point_idx_all = torch.zeros(N).long()      # mask 32x32

    tmp_mask_idx = 0

    for i in range(N):
        h = int(math.floor(i/nW))    # 1D -> 2D
        w = int(math.floor(i%nW))    # 1D -> 2D

        mask_tmp = mask[h*stride:h*stride + patch_size,
                        w*stride:w*stride + patch_size]    # patch size = 1 -> 1 by 1

        
#         # determine whether mask or not
#         if torch.sum(mask_tmp) < mask_thred:    # mask_thread = 5/16
#             nonmask_point_idx_all[tmp_non_mask_idx] = i     # nonmask_point_idx_all = non_mask location
#             tmp_non_mask_idx += 1                           # tmp_non_maks_idx = non_mask count
#         else:
#             mask_point_idx_all[tmp_mask_idx] = i            # mask_location
#             tmp_mask_idx += 1                               # mask count
#             flag[i] = 1                                     # flag == 0,1 mask 1-> mask, 0->nonmask
#             offsets_tmp_vec[i] = -1   
    
        # get mask region
        if torch.sum(mask_tmp) >= mask_thred:
            mask_point_idx_all[tmp_mask_idx] = i            # mask_location
            tmp_mask_idx += 1                               # mask count
            flag[i] = 1                                     # flag == 0,1 mask 1-> mask, 0->nonmask
            offsets_tmp_vec[i] = -1
        # non-mask region -> ref == 1024
        nonmask_point_idx_all[tmp_non_mask_idx] = i
        tmp_non_mask_idx += 1   # 1024
    


    non_mask_num = tmp_non_mask_idx
    mask_num = tmp_mask_idx

    nonmask_point_idx = nonmask_point_idx_all.narrow(0, 0, non_mask_num)  # [1, 2, 3, 4, 5, 6, 7.. 10]
    mask_point_idx=mask_point_idx_all.narrow(0, 0, mask_num)              # [2, 4, 7, 8, 9]

    # get flatten_offsets, N == 1024
    flatten_offsets_all = torch.LongTensor(N).zero_()     # ack mask count
    for i in range(N):
        offset_value = torch.sum(offsets_tmp_vec[0:i+1])
        if flag[i] == 1:
            offset_value = offset_value + 1
        flatten_offsets_all[i+offset_value] = -offset_value

    flatten_offsets = flatten_offsets_all.narrow(0, 0, non_mask_num)   # 772 <- value?

    
    # flag -> [0, 0, ... 1, 1, 1,... 0, 0] ,  nonmask -> [0,1, .... // .. 1023, 1024], mask_point_idx -> [256, 257, ...], 
    return flag, nonmask_point_idx, flatten_offsets, mask_point_idx


# sp_x: LongTensor
# sp_y: LongTensor
def cal_sps_for_Advanced_Indexing(h, w):
    sp_y = torch.arange(0, w).long()
    sp_y = torch.cat([sp_y]*h)

    lst = []
    for i in range(h):
        lst.extend([i]*w)
    sp_x = torch.from_numpy(np.array(lst))
    return sp_x, sp_y


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )


