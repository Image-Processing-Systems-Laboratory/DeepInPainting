import random
from PIL import Image
from glob import glob
import time
import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np

# -*-coding:utf-8-*-
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import models
from collections import namedtuple

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

import functools
from torch.optim import lr_scheduler

from models.InnerCos import InnerCos
from models.InnerCos2 import InnerCos2

import util.util as util

from util.NonparametricShift import NonparametricShift
from util.MaxCoord import MaxCoord


# IPSRFunction.py
class IPSRFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, ref, shift_sz, stride, triple_w, flag, nonmask_point_idx, mask_point_idx, flatten_offsets, sp_x, sp_y):
        # ctx = context <- pytorch: forward, backward, weight temp
        assert input.dim() == 4, "Input Dim has to be 4"  # [1, 512, 32, 32]

        # ctx.ref = ref
        ctx.triple_w = triple_w  # ???
        ctx.flag = flag  # [mask index]
        ctx.flatten_offsets = flatten_offsets  #

        ctx.bz, c_real, ctx.h, ctx.w = input.size()  # [1, 512, 32, 32]

        c = c_real  # 512 channel == c

        ctx.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

        assert mask.dim() == 2, "Mask dimension must be 2"

        # bz is the batchsize of this GPU
        output_lst = ctx.Tensor(ctx.bz, c, ctx.h, ctx.w)  # output
        ind_lst = torch.LongTensor(ctx.bz, ctx.h * ctx.w, ctx.h, ctx.w)  # [1, 1024, 32, 32]

        if torch.cuda.is_available:
            ind_lst = ind_lst.cuda()
            nonmask_point_idx = nonmask_point_idx.cuda()
            mask_point_idx = mask_point_idx.cuda()
            sp_x = sp_x.cuda()
            sp_y = sp_y.cuda()

        for idx in range(ctx.bz):  # 1

            inpatch = input.narrow(0, idx, 1)  # [1, 512, 32, 32] -> feature map -> source feature
            output = ref.relu4_3.narrow(0, idx, 1)  # [1, 512, 32, 32] -> feature map -> ref feature

            Nonparm = NonparametricShift()

            _, conv_enc, conv_new_dec, _, known_patch, unknown_patch = Nonparm.buildAutoencoder(inpatch.squeeze(), False, False, nonmask_point_idx, mask_point_idx, shift_sz, stride)  # inpatch.squeeze() -> [512, 32, 32]

            output_var = Variable(output)

            # cross-correlation vector == tmp1
            tmp1 = conv_enc(output_var)  # [1, 512, 32, 32] -> conv -> [1, 772, 32, 32]

            maxcoor = MaxCoord()

            # ind -> ch , vmax -> value
            kbar, ind, vmax = maxcoor.update_output(tmp1.data, sp_x, sp_y)

            real_patches = kbar.size(1)

            vamx_mask = vmax.index_select(0, mask_point_idx)
            _, _, kbar_h, kbar_w = kbar.size()
            out_new = unknown_patch.clone()
            out_new = out_new.zero_()
            mask_num = torch.sum(ctx.flag)

            in_attention = ctx.Tensor(mask_num, real_patches).zero_()

            kbar = ctx.Tensor(1, real_patches, kbar_h, kbar_w).zero_()
            ind_laten = 0

            # Dmax, Dad
            for i in range(kbar_h):
                for j in range(kbar_w):
                    indx = i * kbar_w + j  # 1024
                    check = torch.eq(mask_point_idx, indx)  # equal, if available -> 1
                    non_r_ch = ind[indx]  #

                    #                     offset = ctx.flatten_offsets[non_r_ch]  # index -> mask patch count
                    #                     correct_ch = int(non_r_ch + offset)     #

                    correct_ch = int(non_r_ch)

                    if (check.sum() >= 1):  # if mask ?
                        known_region = known_patch[non_r_ch]  # correlation max channel index == non_r_ch
                        unknown_region = unknown_patch[ind_laten]

                        # first patch -> copy
                        if ind_laten == 0:
                            out_new[ind_laten] = known_region
                            in_attention[ind_laten, correct_ch] = 1
                            kbar[:, :, i, j] = torch.unsqueeze(in_attention[ind_laten], 0)

                        # not first -> calculate -> paper
                        elif ind_laten != 0:
                            little_value = unknown_region.clone()
                            ininconv = out_new[ind_laten - 1].clone()
                            ininconv = torch.unsqueeze(ininconv, 0)

                            value_2 = little_value * (1 / (little_value.norm(2) + 1e-8))
                            conv_enc_2 = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
                            value_2 = torch.unsqueeze(value_2, 0)
                            conv_enc_2.weight.data = value_2

                            ininconv_var = Variable(ininconv)

                            at_value = conv_enc_2(ininconv_var)
                            at_value_m = at_value.data
                            at_value_m = at_value_m.squeeze()

                            at_final_new = at_value_m / (at_value_m + vamx_mask[ind_laten])
                            at_final_ori = vamx_mask[ind_laten] / (at_value_m + vamx_mask[ind_laten])
                            out_new[ind_laten] = (at_final_new) * out_new[ind_laten - 1] + (at_final_ori) * known_region
                            in_attention[ind_laten] = in_attention[ind_laten - 1] * at_final_new.item()
                            in_attention[ind_laten, correct_ch] = in_attention[ind_laten, correct_ch] + at_final_ori.item()
                            kbar[:, :, i, j] = torch.unsqueeze(in_attention[ind_laten], 0)
                        ind_laten += 1
                    else:
                        kbar[:, correct_ch, i, j] = 1
            kbar_var = Variable(kbar)
            result_tmp_var = conv_new_dec(kbar_var)
            result_tmp = result_tmp_var.data
            output_lst[idx] = result_tmp
            ind_lst[idx] = kbar.squeeze()

        output = output_lst

        ctx.ind_lst = ind_lst
        return output

    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst

        c = grad_output.size(1)

        grad_swapped_all = grad_output.clone()

        spatial_size = ctx.h * ctx.w

        W_mat_all = Variable(ctx.Tensor(ctx.bz, spatial_size, spatial_size).zero_())
        for idx in range(ctx.bz):
            W_mat = W_mat_all.select(0, idx).clone()
            back_attention = ind_lst[idx].clone()
            for i in range(ctx.h):
                for j in range(ctx.w):
                    indx = i * ctx.h + j
                    W_mat[indx] = back_attention[:, i, j]

            W_mat_t = W_mat.t()

            # view(c/3,-1):t() makes each line be a gradient of certain position which is c/3 channels.
            grad_swapped_weighted = torch.mm(W_mat_t, grad_swapped_all[idx].view(c, -1).t())

            # Then transpose it back
            grad_swapped_weighted = grad_swapped_weighted.t().contiguous().view(1, c, ctx.h, ctx.w)
            grad_swapped_all[idx] = torch.add(grad_swapped_all[idx], grad_swapped_weighted.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input = grad_swapped_all

        return grad_input, None, None, None, None, None, None, None, None, None, None, None


# IPSR_model.py
class IPSR_model(nn.Module):
    # constructor
    def __init__(self, threshold, fixed_mask, shift_sz=1, stride=1, mask_thred=1, triple_weight=1):
        super(IPSR_model, self).__init__()
        # threshold=5/16
        # shift-sz=1
        # fixed_mask=1
        self.threshold = threshold
        self.fixed_mask = fixed_mask
        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.cal_fixed_flag = True  # whether we need to calculate the temp varaiables this time. 临时变量
        # 这是恒定的张量，与空间相关，与马赛克范围无关
        # these two variables are for accerlating MaxCoord, it is constant tensors,
        # related with the spatialsize, unrelated with mask.
        self.sp_x = None
        self.sp_y = None

    def set_mask(self, mask_global, layer_to_last, threshold):
        mask = util.cal_feat_mask(mask_global, layer_to_last, threshold)
        self.mask = mask.squeeze()
        return self.mask

    def set_ref(self, latent_ref):
        self.ref = latent_ref

    # If mask changes, then need to set cal_fix_flag true each iteration
    def forward(self, input):  # input : 32x32 feature
        _, self.c, self.h, self.w = input.size()  # [1, 512, 32, 32]

        if self.fixed_mask and self.cal_fixed_flag == False:
            assert torch.is_tensor(self.flag), 'flag must have been figured out and has to be a tensor!'
        else:
            latter = input.narrow(0, 0, 1).data

            # index needs to be modified  [1024], x, [252]
            self.flag, self.nonmask_point_idx, self.flatten_offsets, self.mask_point_idx = util.cal_mask_given_mask_thred(latter.squeeze(), self.mask, self.shift_sz, self.stride, self.mask_thred)
            self.cal_fixed_flag = True

        if not (torch.is_tensor(self.sp_x) or torch.is_tensor(self.sp_y)):
            # 返回
            self.sp_x, self.sp_y = util.cal_sps_for_Advanced_Indexing(self.h, self.w)

        # input == rough output -> refinement -> feature  ,,, ref_latent
        return IPSRFunction.apply(input, self.mask, self.ref, self.shift_sz, self.stride, self.triple_weight, self.flag, self.nonmask_point_idx, self.mask_point_idx, self.flatten_offsets, self.sp_x, self.sp_y)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'threshold: ' + str(self.threshold) + ' ,triple_weight ' + str(self.triple_weight) + ')'


##############################################################################
# Classes (networks.py)
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan_gp', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if gan_type == 'wgan_gp':
            self.loss = nn.MSELoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.loss = nn.BCELoss()
        else:
            raise ValueError("GAN type [%s] not recognized." % gan_type)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        if target_is_real:
            errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - target_tensor) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + target_tensor) ** 2)) / 2
            return errD
        else:
            errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + target_tensor) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - target_tensor) ** 2)) / 2
            return errG


# ipsr layer
class IPSR(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, ipsr_model, cosis_list, cosis_list2, mask_global, input_nc, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(IPSR, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv_3 = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1)
        downrelu_3 = nn.LeakyReLU(0.2, True)
        downnorm_3 = norm_layer(inner_nc, affine=True)
        uprelu_3 = nn.ReLU(True)
        upnorm_3 = norm_layer(outer_nc, affine=True)

        downconv = nn.Conv2d(input_nc, input_nc, kernel_size=4, stride=2, padding=3, dilation=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(input_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # ipsr layer
        ipsr = IPSR_model(opt.threshold, opt.fixed_mask, opt.shift_sz, opt.stride, opt.mask_thred, opt.triple_weight)  # constructor
        ipsr.set_mask(mask_global, 3, opt.threshold)
        ipsr_model.append(ipsr)

        # downsampling
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip)
        innerCos.set_mask(mask_global, opt)  # Here we need to set mask for innerCos layer too.
        cosis_list.append(innerCos)

        # upsampling
        innerCos2 = InnerCos2(strength=opt.strength, skip=opt.skip)
        innerCos2.set_mask(mask_global, opt)  # Here we need to set mask for innerCos layer too.
        cosis_list2.append(innerCos2)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
            down = [downconv_3]
            up = [uprelu, upconv_3]
            model = down + [submodule] + up
        # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
        # else, the normal
        else:
            upconv = nn.ConvTranspose2d(outer_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
            down = [downrelu, downconv, downnorm, downrelu_3, downconv_3, ipsr, innerCos, downnorm_3]
            up = [innerCos2, uprelu_3, upconv_3, upnorm_3, uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    # x == Middle
    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  # cat in the C channel


# refinement
class UnetGeneratorIPSR(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, mask_global, ipsr_model, cosis_list, cosis_list2, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGeneratorIPSR, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock_3(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock_3(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock_3(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        # ipsr layer
        unet_ipsr = IPSR(ngf * 4, ngf * 8, opt, ipsr_model, cosis_list, cosis_list2, mask_global, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock_3(ngf * 2, ngf * 4, input_nc=None, submodule=unet_ipsr, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock_3(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock_3, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv_3 = nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1)
        downrelu_3 = nn.LeakyReLU(0.2, True)
        downnorm_3 = norm_layer(inner_nc, affine=True)
        uprelu_3 = nn.ReLU(True)
        upnorm_3 = norm_layer(outer_nc, affine=True)

        downconv = nn.Conv2d(input_nc, input_nc, kernel_size=4, stride=2, padding=3, dilation=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(input_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
            down = [downconv_3]
            up = [uprelu, upconv_3]
            model = down + [submodule] + up
        # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
        # else, the normal
        else:
            upconv = nn.ConvTranspose2d(outer_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
            down = [downrelu, downconv, downnorm, downrelu_3, downconv_3, downnorm_3]
            up = [uprelu_3, upconv_3, upnorm_3, uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  # cat in the C channel


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# It construct network from the inside to the outside.
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
        # else, the normal
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()

            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  # cat in the C channel


################################### This is for D ###################################
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PFDiscriminator(nn.Module):
    def __init__(self):
        super(PFDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)

        )

    def forward(self, input):
        return self.model(input)

###############################################################################
# Functions (networks.py)
###############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, opt, mask_global, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    cosis_list = []
    cosis_list2 = []
    ipsr_model = []
    # rough
    if which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # refinement
    elif which_model_netG == 'unet_ipsr':
        netG = UnetGeneratorIPSR(input_nc, output_nc, 8, opt, mask_global, ipsr_model, cosis_list, cosis_list2, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    return init_net(netG, init_type, init_gain, gpu_ids), cosis_list, cosis_list2, ipsr_model


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'feature':
        netD = PFDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)

    return init_net(netD, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(5):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(5, 10):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 17):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)
        save_filename = '%s_net_%s.pt' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pt' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class IPSRModel(BaseModel):
    def name(self):
        return 'IPSRModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda')
        self.opt = opt
        self.isTrain = opt.isTrain
        self.vgg = Vgg16(requires_grad=False)
        self.vgg = self.vgg.cuda()
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.input_ref = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        # batchsize should be 1 for mask_global
        self.mask_global = torch.BoolTensor(1, 1, opt.fineSize, opt.fineSize)
        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize / 4) + self.opt.overlap: int(self.opt.fineSize / 2) + int(self.opt.fineSize / 4) - self.opt.overlap, \
        int(self.opt.fineSize / 4) + self.opt.overlap: int(self.opt.fineSize / 2) + int(self.opt.fineSize / 4) - self.opt.overlap] = 1
        self.mask_type = opt.mask_type
        self.gMask_opts = {}

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        # refinement
        self.netG, self.Cosis_list, self.Cosis_list2, self.IPSR_model = define_G(opt.input_nc_g, opt.output_nc, opt.ngf, opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)
        # rough
        self.netP, _, _, _ = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netP, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)

        # discriminator
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion

            self.netD = define_D(opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netF = define_D(opt.input_nc, opt.ndf, opt.which_model_netF, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netP, 'P', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netF, 'F', opt.which_epoch)
        self.criterionGAN = GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
        self.criterionL1 = torch.nn.L1Loss()

        # loss
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_P = torch.optim.Adam(self.netP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_P)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            print_network(self.netG)
            print_network(self.netP)
            if self.isTrain:
                print_network(self.netD)
                print_network(self.netF)
            print('-----------------------------------------------')

    # training
    def set_isTrain(self):
        self.isTrain = True

    # validation
    def set_isVal(self):
        self.isTrain = False

    def set_input(self, input, mask, ref):
        input_A = input  # Ground Truth(For masking)
        input_B = input  # Ground Truth(For L1 loss)
        input_mask = mask  # Mask
        input_ref = ref  # Augmented Image

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_ref.resize_(input_ref.size()).copy_(input_ref)
        self.image_paths = 0
        if self.opt.mask_type == 'center':
            self.mask_global = self.mask_global
        elif self.opt.mask_type == 'random':
            self.mask_global.zero_()
            self.mask_global = input_mask
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)
        self.ex_mask = self.mask_global.expand(1, 3, self.mask_global.size(2), self.mask_global.size(3))  # 1*c*h*w
        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).bool()
        self.input_A.narrow(1, 0, 1).masked_fill_(self.mask_global, 2 * 123.0 / 255.0 - 1.0)  # -0.03
        self.input_A.narrow(1, 1, 1).masked_fill_(self.mask_global, 2 * 104.0 / 255.0 - 1.0)  # -0.18
        self.input_A.narrow(1, 2, 1).masked_fill_(self.mask_global, 2 * 117.0 / 255.0 - 1.0)  # -0.08
        self.set_latent_mask(self.mask_global, 3, self.opt.threshold)

    # It is quite convinient, as one forward-pass, all the innerCos will get the GT_latent!
    def set_latent_mask(self, mask_global, layer_to_last, threshold):
        self.IPSR_model[0].set_mask(mask_global, layer_to_last, threshold)
        self.Cosis_list[0].set_mask(mask_global, self.opt)
        self.Cosis_list2[0].set_mask(mask_global, self.opt)

    # training
    def set_ref_latent(self):
        self.ref_latent = self.vgg(Variable(self.input_ref, requires_grad=False))
        self.IPSR_model[0].set_ref(self.ref_latent)

    # propagation init
    def forward(self):
        self.real_A = self.input_A.to(self.device)
        self.fake_P = self.netP(self.real_A)
        self.un = self.fake_P.clone()
        self.Unknowregion = self.un.data.masked_fill_(self.inv_ex_mask, 0)
        self.knownregion = self.real_A.data.masked_fill_(self.ex_mask, 0)
        self.Syn = self.Unknowregion + self.knownregion
        self.Middle = torch.cat((self.Syn, self.input_A), 1)

        # refinement
        self.fake_B = self.netG(self.Middle)
        self.real_B = self.input_B.to(self.device)
        self.real_Ref = self.input_ref.to(self.device)

    # consistency loss
    def set_gt_latent(self):
        gt_latent = self.vgg(Variable(self.input_B, requires_grad=False))
        self.Cosis_list[0].set_target(gt_latent.relu4_3)
        self.Cosis_list2[0].set_target(gt_latent.relu4_3)

    def test(self):
        self.real_A = self.input_A.to(self.device)
        self.fake_P = self.netP(self.real_A)
        self.un = self.fake_P.clone()
        self.Unknowregion = self.un.data.masked_fill_(self.inv_ex_mask, 0)
        self.knownregion = self.real_A.data.masked_fill_(self.ex_mask, 0)
        self.Syn = self.Unknowregion + self.knownregion
        self.Middle = torch.cat((self.Syn, self.input_A), 1)
        self.fake_B = self.netG(self.Middle)
        self.real_B = self.input_B.to(self.device)
        self.real_Ref = self.input_ref.to(self.device)
        self.loss_IPSR = self.criterionGAN(self.real_B, self.fake_B, False)

    def get_loss(self):
        self.loss_valid = (self.criterionL1(self.fake_B, self.real_B) + self.criterionL1(self.fake_P, self.real_B)) * self.opt.lambda_A
        return OrderedDict([('GAN', self.loss_valid.data.item())])

    def backward_D(self):
        fake_AB = self.fake_B
        # Real
        self.gt_latent_fake = self.vgg(Variable(self.fake_B.data, requires_grad=False))
        self.gt_latent_real = self.vgg(Variable(self.input_B, requires_grad=False))
        real_AB = self.real_B  # GroundTruth
        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)
        self.pred_fake_F = self.netF(self.gt_latent_fake.relu3_3.detach())
        self.pred_real_F = self.netF(self.gt_latent_real.relu3_3)
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F, self.pred_real_F, True)
        self.loss_D = self.loss_D_fake * 0.5 + self.loss_F_fake * 0.5
        # When two losses are ready, together backward.
        # It is op, so the backward will be called from a leaf.(quite different from LuaTorch)
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B
        fake_f = self.gt_latent_fake
        pred_fake = self.netD(fake_AB)
        pred_fake_f = self.netF(fake_f.relu3_3)
        pred_real = self.netD(self.real_B)
        pred_real_F = self.netF(self.gt_latent_real.relu3_3)

        self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_F, False)
        # Second, G(A) = B
        self.loss_G_L1 = (self.criterionL1(self.fake_B, self.real_B) + self.criterionL1(self.fake_P, self.real_B)) * self.opt.lambda_A
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight

        # Third add additional netG contraint loss!
        self.ng_loss_value = 0
        self.ng_loss_value2 = 0
        if self.opt.cosis:
            for gl in self.Cosis_list:
                # self.ng_loss_value += gl.backward()
                self.ng_loss_value += Variable(gl.loss.data, requires_grad=True)
            self.loss_G += self.ng_loss_value
            for gl in self.Cosis_list2:
                # self.ng_loss_value += gl.backward()
                self.ng_loss_value2 += Variable(gl.loss.data, requires_grad=True)
            self.loss_G += self.ng_loss_value2
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()
        self.optimizer_G.zero_grad()
        self.optimizer_P.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_P.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('D', self.loss_D_fake.data.item()),
                            ('F', self.loss_F_fake.data.item())
                            ])

    def get_current_visuals(self):
        real_A = self.real_A.data
        fake_B = self.fake_B.data
        real_B = self.real_B.data
        real_Ref = self.real_Ref.data
        fake_P = self.fake_P.data
        return real_A, real_Ref, fake_B, fake_P, real_B

    def get_error(self):
        return self.loss_IPSR

    def save(self, epoch):
        self.save_network(self.netG, 'G', epoch, self.gpu_ids)
        self.save_network(self.netP, 'P', epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', epoch, self.gpu_ids)
        self.save_network(self.netF, 'F', epoch, self.gpu_ids)

    def load(self, epoch):
        self.load_network(self.netG, 'G', epoch)
        self.load_network(self.netP, 'P', epoch)


# models.py
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'ipsr_net':
        model = IPSRModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model


class Data_load(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, ref_root, img_transform, mask_transform, ref_transform):
        super(Data_load, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.ref_transform = ref_transform

        self.paths = glob('{:s}/*.jpg'.format(img_root), recursive=False)
        self.ref_paths = glob('{:s}/*.jpg'.format(ref_root), recursive=False)
        self.mask_paths = glob('{:s}/*.png'.format(mask_root))

        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))

        ref = Image.open(self.ref_paths[index])
        ref = self.ref_transform(ref.convert('RGB'))
        return gt_img, mask, ref

    def __len__(self):
        return len(self.paths)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=8):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Option():
    def __init__(self):
        self.dataroot = r'/home/jara/dataset/Paris StreetView/train'
        self.refroot = r'/home/jara/dataset/Paris StreetView/train_ref'
        self.validroot = r'/home/jara/dataset/Paris StreetView/valid'
        self.validrefroot = r'/home/jara/dataset/Paris StreetView/valid_ref'
        self.maskroot = r'/home/jara/dataset/mask'
        self.batchSize = 1  # Need to be set to 1
        self.fineSize = 256  # image size
        self.input_nc = 3  # input channel size for first stage
        self.input_nc_g = 6  # input channel size for second stage
        self.output_nc = 3  # output channel size
        self.ngf = 64  # inner channel
        self.ndf = 64  # inner channel
        self.which_model_netD = 'basic'  # patch discriminator
        self.which_model_netF = 'feature'  # feature patch discriminator
        self.which_model_netG = 'unet_ipsr'  # second stage network
        self.which_model_netP = 'unet_256'  # first stage network
        self.triple_weight = 1
        self.name = 'IPSR_inpainting'
        self.n_layers_D = '3'  # network depth
        self.gpu_ids = [0]  # use gpu_id 1
        self.model = 'ipsr_net'
        self.checkpoints_dir = r'/home/jara/DeepInPainting/checkpoints'
        self.norm = 'instance'
        self.fixed_mask = 1
        self.use_dropout = True
        self.init_type = 'normal'
        self.mask_type = 'random'
        self.lambda_A = 100
        self.threshold = 5 / 16.0
        self.stride = 1
        self.shift_sz = 1  # size of feature patch
        self.mask_thred = 1
        self.bottleneck = 512
        self.gp_lambda = 10.0
        self.ncritic = 5
        self.constrain = 'MSE'
        self.strength = 1
        self.init_gain = 0.02
        self.cosis = 1
        self.gan_type = 'lsgan'
        self.gan_weight = 0.2
        self.overlap = 4
        self.skip = 0
        self.display_freq = 1000
        self.print_freq = 50
        self.save_latest_freq = 5000
        self.save_epoch_freq = 1
        self.continue_train = False
        self.epoch_count = 1
        self.phase = 'train'
        self.which_epoch = ''
        self.niter = 20
        self.niter_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0002
        self.lr_policy = 'lambda'
        self.lr_decay_iters = 50
        self.isTrain = True


opt = Option()
transform_mask = transforms.Compose(
    [transforms.Resize((opt.fineSize, opt.fineSize)),
     transforms.ToTensor(),
     ])
transform_ref = transforms.Compose(
    [transforms.Resize((opt.fineSize, opt.fineSize)),
     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
transform = transforms.Compose(
    [transforms.RandomResizedCrop((opt.fineSize, opt.fineSize), scale=(0.8, 1.0), ratio=(1, 1)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

# Load train data
dataset_train = Data_load(opt.dataroot, opt.maskroot, opt.refroot, transform, transform_mask, transform_ref)
iterator_train = (data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True))
# Load validation data
dataset_valid = Data_load(opt.validroot, opt.maskroot, opt.validrefroot, transform, transform_mask, transform_ref)
iterator_valid = (data.DataLoader(dataset_valid, batch_size=opt.batchSize, shuffle=True))

# Create model
model = create_model(opt)
total_steps = 0

iter_start_time = time.time()
early = EarlyStopping(20)

train_loss = []
valid_loss = []
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    t_loss = []
    v_loss = []

    epoch_start_time = time.time()
    epoch_iter = 0

    for image, mask, ref in iterator_train:
        image = image.cuda()
        mask = mask.cuda()
        mask = mask[0][0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.bool()

        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        model.set_input(image, mask, ref)  # it not only sets the input data with mask, but also sets the latent mask.
        model.set_ref_latent()
        model.set_gt_latent()
        model.optimize_parameters()  # forward

        tl = model.get_loss().get('GAN')
        t_loss.append(tl)

        if total_steps % opt.display_freq == 0:
            real_A, real_Ref, fake_B, fake_P, real_B = model.get_current_visuals()
            pic = (torch.cat([real_A, real_Ref, fake_P, fake_B], dim=0) + 1) / 2.0
            torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (r'/home/jara/DeepInPainting/saveimg', epoch, total_steps + 1, len(dataset_train)), nrow=2)

    model.save(epoch)

    for image, mask, ref in iterator_valid:
        image = image.cuda()
        mask = mask.cuda()
        mask = mask[0][0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.bool()

        model.set_input(image, mask, ref)  # it not only sets the input data with mask, but also sets the latent mask.
        model.set_ref_latent()
        model.set_gt_latent()
        model.test()
        vl = model.get_loss().get('GAN')
        v_loss.append(vl)

    train_result_loss = np.average(t_loss)
    valid_result_loss = np.average(v_loss)

    train_loss.append(train_result_loss)
    valid_loss.append(valid_result_loss)

    print('Epoch : %d -> Train loss : %f, Valid loss : %f' % (epoch, train_result_loss, valid_result_loss))

    early(valid_result_loss)
    if early.early_stop:
        print("Early stopping")
        break

    model.update_learning_rate()
