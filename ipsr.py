import random
from PIL import Image
from glob import glob
import time
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np

# -*-coding:utf-8-*-
from collections import OrderedDict
from torch.autograd import Variable

import torch.nn.functional as F
from models.base_model import BaseModel
from models import networks
from models.vgg16 import Vgg16


class IPSR(BaseModel):
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
        self.netG, self.Cosis_list, self.Cosis_list2, self.IPSR_model = networks.define_G(opt.input_nc_g, opt.output_nc, opt.ngf, opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)
        # rough
        self.netP, _, _, _ = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netP, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)

        # discriminator
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion

            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netF = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netF, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netP, 'P', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netF, 'F', opt.which_epoch)
        self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
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
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            networks.print_network(self.netP)
            if self.isTrain:
                networks.print_network(self.netD)
                networks.print_network(self.netF)
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


def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'ipsr_net':
        model = IPSR()
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
        self.which_model_netG = 'unet_csa'  # second stage network
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
transform = transforms.Compose(
    [transforms.Resize((opt.fineSize, opt.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
transform_ref = transforms.Compose(
    [transforms.RandomResizedCrop((opt.fineSize, opt.fineSize), scale=(0.8, 1.0), ratio=(1, 1)),
     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
