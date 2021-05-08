class Opion():
    
    def __init__(self):
        
        #self.dataroot= r'/home/jara/DeepInPainting_3/test1' #image dataroot
        #self.maskroot= r'/home/jara/DeepInPainting_3/test2'
        self.batchSize= 1   # Need to be set to 1
        self.fineSize=256 # image size
        self.input_nc=3  # input channel size for first stage
        self.input_nc_g=6 # input channel size for second stage
        self.output_nc=3# output channel size
        self.ngf=64 # inner channel
        self.ndf=64# inner channel
        self.which_model_netD='basic' # patch discriminator
        
        self.which_model_netF='feature'# feature patch discriminator
        self.which_model_netG='unet_csa'# seconde stage network
        self.which_model_netP='unet_256'# first stage network
        self.triple_weight=1
        self.name='CSA_inpainting'
        self.n_layers_D='3' # network depth
        self.gpu_ids=[0]
        self.model='csa_net'
        self.checkpoints_dir=r'/home/jara/DeepInPainting_3/checkpoints' #
        self.norm='instance'
        self.fixed_mask=1
        self.use_dropout=False
        self.init_type='normal'
        self.mask_type='random'
        self.lambda_A=100
        self.threshold=5/16.0
        self.stride=1
        self.shift_sz=1 # size of feature patch
        self.mask_thred=1
        self.bottleneck=512
        self.gp_lambda=10.0
        self.ncritic=5
        self.constrain='MSE'
        self.strength=1
        self.init_gain=0.02
        self.cosis=1
        self.gan_type='lsgan'
        self.gan_weight=0.2
        self.overlap=4
        self.skip=0
        self.display_freq=1000
        self.print_freq=50
        self.save_latest_freq=5000
        self.save_epoch_freq=2
        self.continue_train=False
        self.epoch_count=1
        self.phase='train'
        self.which_epoch=46
        self.niter=20
        self.niter_decay=100
        self.beta1=0.5
        self.lr=0.0002
        self.lr_policy='lambda'
        self.lr_decay_iters=50
        self.isTrain=False

import os
import shutil
from util.data_load import Data_load
from models.models import create_model
import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
#from IQA_pytorch import SSIM
import numpy as np
from flask import Flask, render_template, request, url_for, redirect
from werkzeug import secure_filename
from PIL import Image

app = Flask(__name__)

opt = Opion()
model = create_model(opt)

load_epoch=46
model.load(load_epoch)
cnt=1

image_path1=r'/home/jara/DeepInPainting_3/test1'
#image_path2=r'/home/jara/download'
image_path2=r'/home/jara/DeepInPainting_3/test2'
image_path3=r'/home/jara/DeepInPainting_3/test3'

@app.route("/") 
def index():
	# return render_template('home.html')
	return render_template('index.html')	

@app.route('/getImage',methods=['GET','POST'])
def get_image():
	if request.method=='POST':
		save_dir=r'/home/jara/DeepInPainting_3/static/img'
		
		if(os.path.isdir(image_path1)==True):	
			shutil.rmtree(image_path1)
			os.mkdir(image_path1)
		if(os.path.isdir(image_path2)==True):	
			shutil.rmtree(image_path2)
			os.mkdir(image_path2)
		if(os.path.isdir(image_path3)==True):	
			shutil.rmtree(image_path3)
			os.mkdir(image_path3)
		
		print(request.files)

		f=request.files['srcImage']
		filename=secure_filename(f.filename)
		f.save(os.path.join(image_path1,filename))

		f=request.files['binaryMask']
		filename=secure_filename(f.filename)
		f.save(os.path.join(image_path2,filename))

		f=request.files['refImage']
		filename=secure_filename(f.filename)
		f.save(os.path.join(image_path3,filename))

		# load mask
		transform_mask = transforms.Compose(
		[transforms.Resize((opt.fineSize,opt.fineSize)),
		 transforms.ToTensor(),
		])
		# load image
		transform = transforms.Compose(
		[transforms.Resize((opt.fineSize,opt.fineSize)),
		 transforms.ToTensor(),
		 transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
		dataroot=r'/home/jara/DeepInPainting_3/test1'
		#maskroot=r'/home/jara/download'
		maskroot=r'/home/jara/DeepInPainting_3/test2'
		refroot=r'/home/jara/DeepInPainting_3/test3'
		dataset_test = Data_load(dataroot, maskroot, refroot, transform, transform_mask, transform)
		iterator_test = (data.DataLoader(dataset_test, batch_size=opt.batchSize,shuffle=False))

		# 3 pictures <- request parsing
		for image, mask, ref in (iterator_test):
			image=image.cuda()
			mask=mask.cuda()
			mask=mask[0][0]
			mask=torch.unsqueeze(mask,0)
			mask=torch.unsqueeze(mask,1)
			mask=mask.bool()

			#pic1 = ((torch.cat([mask], dim=0) + 1) / 2.0)
			#torchvision.utils.save_image(pic1, r'/home/jara/DeepInPainting_3/result/test.jpg')
			
			model.set_input(image,mask,ref) # it not only sets the input data with mask, but also sets the latent mask.
			model.set_ref_latent()
			model.set_gt_latent()
			model.test()
			real_A,ref_B,fake_B,fake_P,real_B=model.get_current_visuals()
			# pic = ((torch.cat([real_A,ref_B,fake_B,fake_P], dim=0) + 1) / 2.0)
			pic = ((torch.cat([fake_B], dim=0) + 1) / 2.0)
			torchvision.utils.save_image(pic, r'/home/jara/DeepInPainting_3/static/img/test.jpg')
	return redirect(url_for('showResult'))



@app.route("/result")
def showResult():
	return render_template('result.html')


if __name__ == '__main__':
	app.run(debug = True, host = '127.0.0.1', port = 5000)

