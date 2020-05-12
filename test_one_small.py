from __future__ import print_function
from __future__ import division
from past.utils import old_div
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd.functional import jacobian

from model_small import _netG

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--test_image', required=True, help='path to dataset')
parser.add_argument('--output_directory',default='.',help='path to output directory. (it needs to be made already!!!)')
parser.add_argument('--output_name_prefix',required=True,help='prefix name for the outputfiles that will be saved')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=1,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)



# Load the model
netG = _netG(opt)
# netG = TransformerNet()
netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
# netG.requires_grad = False
netG.eval()



#transform is applied to image below.  image is class PIL.Image
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Note: load_image resizes the image to (opt.imageSize,opt.imageSize)
image = utils.load_image(opt.test_image, opt.imageSize)
image = transform(image)
image = image.repeat(1, 1, 1, 1)

# Create emtpy tensors for each
input_real = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
real_center = torch.FloatTensor(1, 3, old_div(opt.imageSize,2), old_div(opt.imageSize,2))

criterionMSE = nn.MSELoss()

# if opt.cuda:
#     netG.cuda()
#     input_real, input_cropped = input_real.cuda(),input_cropped.cuda()
#     criterionMSE.cuda()
#     real_center = real_center.cuda()

# I don't think this is needed anymore, but Variable is from autograd package
input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
real_center = Variable(real_center)

# input_cropped and input_real are still full size (128x128), but copies of image? 

input_real.data.resize_(image.size()).copy_(image)
input_cropped.data.resize_(image.size()).copy_(image)

# real_center_cpu is the center of the image (defaults to half of each dimension, so 64x64)
real_center_cpu = image[:,:,56:72,56:72]
# real_center_cpu = image[:,:,old_div(opt.imageSize,4):old_div(opt.imageSize,4)+old_div(opt.imageSize,2),old_div(opt.imageSize,4):old_div(opt.imageSize,4)+old_div(opt.imageSize,2)]


# real_center.data is resized and copied over with real_center_cpu (actual image center)
with torch.no_grad():
	real_center.resize_(real_center_cpu.size()).copy_(real_center_cpu)

# This is masking the center of the image (not sure why the three channels have different values, maybe rgb each have different zeros?)
input_cropped.data[:,0,int(56+opt.overlapPred):int(72-opt.overlapPred),int(56+opt.overlapPred):int(72-opt.overlapPred)] = 2*117.0/255.0 - 1.0
input_cropped.data[:,1,int(56+opt.overlapPred):int(72-opt.overlapPred),int(56+opt.overlapPred):int(72-opt.overlapPred)] = 2*104.0/255.0 - 1.0
input_cropped.data[:,2,int(56+opt.overlapPred):int(72-opt.overlapPred),int(56+opt.overlapPred):int(72-opt.overlapPred)] = 2*123.0/255.0 - 1.0

# input_cropped.data[:,0,old_div(opt.imageSize,4)+opt.overlapPred:old_div(opt.imageSize,4)+old_div(opt.imageSize,2)-opt.overlapPred,old_div(opt.imageSize,4)+opt.overlapPred:old_div(opt.imageSize,4)+old_div(opt.imageSize,2)-opt.overlapPred] = 2*117.0/255.0 - 1.0
# input_cropped.data[:,1,old_div(opt.imageSize,4)+opt.overlapPred:old_div(opt.imageSize,4)+old_div(opt.imageSize,2)-opt.overlapPred,old_div(opt.imageSize,4)+opt.overlapPred:old_div(opt.imageSize,4)+old_div(opt.imageSize,2)-opt.overlapPred] = 2*104.0/255.0 - 1.0
# input_cropped.data[:,2,old_div(opt.imageSize,4)+opt.overlapPred:old_div(opt.imageSize,4)+old_div(opt.imageSize,2)-opt.overlapPred,old_div(opt.imageSize,4)+opt.overlapPred:old_div(opt.imageSize,4)+old_div(opt.imageSize,2)-opt.overlapPred] = 2*123.0/255.0 - 1.0
fake = netG(input_cropped)
errG = criterionMSE(fake,real_center)

# zerov = input_cropped*0
# print("zerov max: {}".format(zerov.max()))
# print("zerov min: {}".format(zerov.min()))
# zerov[0,0,0,0] = 0.1
# zero = netG(zerov)
# print("zero max: {}".format(zero.max()))
# print("zero min: {}".format(zero.min()))
# print(type(zero))
# print(zero.__dir__())
# Add fake back into center of input_cropped
recon_image = input_cropped.clone()
recon_image.data[:,:,56:72,56:72] = fake.data

utils.save_image(opt.output_directory  + opt.output_name_prefix + '_orig.png',image[0])
utils.save_image(opt.output_directory  + opt.output_name_prefix + '_cropped.png',input_cropped.data[0])
utils.save_image(opt.output_directory  + opt.output_name_prefix + '_recons.png',recon_image.data[0])

utils.save_image_color(opt.output_directory  + opt.output_name_prefix + '_color1_orig.png',image[0][1])
utils.save_image_color(opt.output_directory  + opt.output_name_prefix + '_color1_cropped.png',input_cropped.data[0][1])
utils.save_image_color(opt.output_directory  + opt.output_name_prefix + '_color1_recons.png',recon_image.data[0][1])

print('\nMSE Loss: %.4f\n' % errG.item())
# The Jacobian stuff
# print("Shape of input_cropped: ",input_cropped.shape)
# print("Shape of fake: ",fake.shape)
# # torch.save(zero,"jacobians/zero_202535.pkl")
# torch.save(fake,"jacobians/fake_202535.pkl")
# jacob = jacobian(netG,input_cropped)

# jacob_dir = 'jacobians/'
# jacob_filename = 'jacob_202535.pkl'
# torch.save(jacob,jacob_dir+jacob_filename)
# print("Shape of jacob: ",jacob.shape)
# torch.save(input_cropped,"jacobians/input_cropped_202535.pkl")