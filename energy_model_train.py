import sys
from google.colab import drive
drive.mount('/ME')
sys.path.append('/ME/My Drive/energy/')

import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import json
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import download_flowers_data, plot_ims, plot_diagnostics
import math
import numpy as np
import gc

#next steps - plot gradient if previous loss is positive, plot if previous loss is negative
#check size of norm progression for coopnets
#convert to TPU

config = {
  "seed": 123, #for replicability
  "z_size": 100, #number of latent variables in latent z for coopnets
  'img_size': 32, #width/length of image
  'num_channels': 1, #number of channels of image, 1 or 3
  'langevin_step_num_gen': 0, #number of langevin/sgd steps for generator, usually 0
  'batch_size': 100, #number of samples per iteration
  'lr_gen': 0.0001, #learning rate of generator
  'beta1_gen': 0.5, #beta1 for optimizer of generator
  'sigma_gen': 0.3, #sigma for langevin dynamics of generator
  'lr_des': 0.0001, #learning rate of descriptor
  'beta1_des': 0.9, #beta1 for optimizer of descriptor
  'with_noise': 'no', #use sgd with/without noise for descriptor
  'langevin_step_num_des': 100, #number of minimizing steps from initial points for descriptor  
  'num_iter':  10000, #number of total training iterations
  'init': 'uniform', #initialize samples - uniform, gaussian, generator, persistent
  'langevin_step_size_des': 1, #step size for sgd/adam/langevin of descriptor
  'noise_divide': 1, #for strict langevin, divide variance of noise
  'data_epsilon': 0.03, #add noise to observed samples
  'l2_energy_reg': 0, #penalty on energy in loss
  'weight_decay': 0, #weight decay of optimizer
  'minimizer': 'sgd', #minimzier of samples for descriptor - adam or sgd
  'sgd_beta': 0, #momentum for sgd of synthesized samples
  'test_size': 10000, #test size output of finished models for FID scores
  'data': 'FashionMNIST', #dataset - usually mnist or cifar10
  'spec_norm': 'no', #add spectral normialization
  'n_f' : 64, #complexity of net
  'leak' : 0.2, #leaky_relu parameter
  'lr_decrease': 'no' #manually decrease learning rate
}

ebm_dir='/ME/My Drive/energy/'

device = t.device('cuda' if t.cuda.is_available() else 'cpu') #use gpu

#these are optimum beta for different initializations
if config['init']=='uniform':
  config['beta1_des'] = 0.9
elif config['init']=='generator':
  config['beta1_des'] = 0.5

digit = None #if not none, select digit for unconditional modeling. if none, all 10 digits are modeled

# directory for experiment result
EXP_DIR = f"{ebm_dir}out_data/{str(config['data'])}_{config['init']}_{config['langevin_step_size_des']}_step_{config['data_epsilon']}_noise_{config['lr_des']}_lr_{config['langevin_step_num_des']}_{config['minimizer']}_beta_{config['sgd_beta']}_{config['spec_norm']}_energy_{config['l2_energy_reg']}_{config['with_noise']}/"
print(EXP_DIR)

# make directory for saving results
#WARNING: currently will delete current directory. Be careful here.
if os.path.exists(EXP_DIR):
  print('file already exists. overwrite')
  shutil.rmtree(EXP_DIR)           # Removes all the subdirectories!
  os.makedirs(EXP_DIR)
  for folder in ['checkpoints', 'shortrun', 'longrun']:
    os.mkdir(EXP_DIR + folder)
else:
  os.makedirs(EXP_DIR)
  for folder in ['checkpoints', 'shortrun', 'longrun']:
    os.mkdir(EXP_DIR + folder)

#record configuration for easier reference from past experiments
config_json = json.dumps(config)
f = open(EXP_DIR+'config.json',"w")
f.write(config_json)
f.close()

# set seed for cpu and CUDA, get device
t.manual_seed(config['seed'])
if t.cuda.is_available():
  t.cuda.manual_seed_all(config['seed'])

print(t.cuda.is_available())
print(device) #confirm cuda

################################
# ## Download Data ## #
################################

print('Processing data...')
# make tensor of training data
#default values are range [-1, 1]
#save_image puts this into scale of [0,1]
#code largely borrowed from ebm-anatomy
if config['data'] == 'flowers':
  download_flowers_data()
data = {'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
        'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
        'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func),
        'FashionMNIST': lambda path, func: datasets.FashionMNIST(root=path, transform=func, download=True)}

transform = tr.Compose([tr.Resize(config['img_size']),
                        tr.CenterCrop(config['img_size']),
                        tr.ToTensor(),
                        tr.Normalize(tuple(0.5*t.ones(config['num_channels'])), tuple(0.5*t.ones(config['num_channels'])))])
q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)])
labels = [x[1] for x in data[config['data']]('./data/' + config['data'], transform)]

train_q = q[0:40000]
test_q = q[40000:]
train_labels = labels[0:40000]
test_labels = labels[40000:]
print(test_q.shape)
#split to train/test - test not currently needed

#restrict to digit if option chosen
#if so, we do single class generation
#otherwise, multiclass
if digit is not None:
  #unconditional filtering
  index_list = [i for i,val in enumerate(train_labels) if val==digit]
  index_test = [i for i,val in enumerate(test_labels) if val==digit]
  train_q = train_q[index_list,:,:,:].to(device)
  test_q = test_q[index_test,:,:,:].to(device)

if config['init'] == 'persistent':
  s_t_0 = (2*(t.rand([500000, config['num_channels'], config['img_size'], config['img_size']])) - 1) #initialize uniformly random images for persistent CD
#size of bank of images will change training from data -> uniform

#test data is not currently used but is separated in case of future use

################################
# ## Define Networks ## #
################################

# Define Descriptor
#1 input channel for mnist
class Descriptor(nn.Module):
  def __init__(self):
    super(Descriptor, self).__init__()

    self.conv1 = nn.Conv2d(config['num_channels'], config['n_f'], kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(config['n_f'], config['n_f'] * 2, kernel_size=4, stride=2, padding=1)
    self.conv3 = nn.Conv2d(config['n_f'] * 2, config['n_f'] * 4, kernel_size=4, stride=2, padding=1)
    self.conv4 = nn.Conv2d(config['n_f'] * 4, config['n_f'] * 8, kernel_size=4, stride=2, padding=1)
    self.conv5 = nn.Conv2d(config['n_f'] * 8, 1, kernel_size=4, stride=1, padding=0)

    #self.fc = nn.Linear(16384,1)
    self.leakyrelu = nn.LeakyReLU(config['leak'])

    #spectral norm option. For this, usually keep step size/num steps large
    if config['spec_norm'] == 'spec_norm':
      self.conv1 = nn.utils.spectral_norm(self.conv1)
      self.conv2 = nn.utils.spectral_norm(self.conv2)
      self.conv3 = nn.utils.spectral_norm(self.conv3)
      self.conv4 = nn.utils.spectral_norm(self.conv4)
      self.conv5 = nn.utils.spectral_norm(self.conv5)

  def forward(self, x):
    self.x = x
    out = self.conv1(x)
    out = self.leakyrelu(out)
    out = self.conv2(out)
    out = self.leakyrelu(out)
    out = self.conv3(out)
    out = self.leakyrelu(out)
    out = self.conv4(out)
    out = self.leakyrelu(out)
    out = self.conv5(out)
    return out.squeeze()

# define generator
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.convt1 = nn.ConvTranspose2d(config['z_size'], 256, kernel_size=4, stride=1, padding=0)
    self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
    self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
    self.convt4 = nn.ConvTranspose2d(64, config['num_channels'], kernel_size=5, stride=2, padding=2, output_padding=1)
    self.bn1 = nn.BatchNorm2d(256)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(64)
    self.leakyrelu = nn.LeakyReLU()
    self.tanh = nn.Tanh()

  def forward(self, z):
    self.z = z
    out = self.convt1(z)
    out = self.bn1(out)
    out = self.leakyrelu(out)
    out = self.convt2(out)
    out = self.bn2(out)
    out = self.leakyrelu(out)
    out = self.convt3(out)
    out = self.bn3(out)
    out = self.leakyrelu(out)
    out = self.convt4(out)
    out = self.tanh(out)
    return out

# take sample data using ebm-anatomy method

################################
# ## FUNCTIONS FOR SAMPLING/PLOTTING ## #
################################

# sample batch from given array of images
def sample_image_set(image_set, num_samps):
  rand_inds = t.randperm(image_set.shape[0])[0:num_samps]
  return image_set[rand_inds], rand_inds

# sample positive images from dataset distribution q (add noise to ensure min sd is at least langevin noise sd)
def sample_q(num_samps):
  x_q = sample_image_set(train_q, num_samps)[0]
  return x_q

def save_data_plot(data_in, dir_in, name_in):
  #save after 200 iterations in case initial values are extremely large
  fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
  ax.plot(data_in[200:])
  fig.savefig(dir_in+'post_200_'+name_in+'.png')   # save the figure to file
  plt.close(fig)    # close the figure window

  fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
  ax.plot(data_in)
  fig.savefig(dir_in+'all_'+name_in+'.png')   # save the figure to file
  plt.close(fig)    # close the figure window

  np.save(dir_in+name_in+'.npy', data_in) #save array for future use

##########################################################
# ## Define Main Class with Train/Test Functions ## #
##########################################################

#G0/1/2, D0/1/2 are notation from coopnets paper
class CoopNets(nn.Module):
  def __init__(self):
    super(CoopNets, self).__init__()
    self.descriptor = Descriptor().cuda()
    self.generator = Generator().cuda()

  #Define Langevin Dynamics for Generator - largely untouched from coopnets code
  def langevin_dynamics_generator(self, z, obs):
    obs = obs.detach()
    criterian = nn.MSELoss(size_average=False, reduce=True)
    for i in range(config['langevin_step_num_gen']):
      noise = Variable(t.randn(config['batch_size'], config['z_size'], 1, 1).cuda())
      z = Variable(z, requires_grad=True)
      gen_res = self.generator(z)
      gen_loss = 1.0 / (2.0 * config['sigma_gen'] * config['sigma_gen']) * criterian(gen_res, obs)
      gen_loss.backward()
      grad = z.grad
      z = z - 0.5 * config['langevin_step_size_gen'] * config['langevin_step_size_gen'] * (z + grad)
      if config['with_noise'] == 'yes':
        z += config['langevin_step_size_gen'] * noise

      return z

  #udpated langevin dynamics function for descriptor (main energy model)
  #this is no longer actual langevin dynamics. Instead, it is ADAM or SGD
  #with/without added noise
  def langevin_dynamics_descriptor(self, x):
      
    x = Variable(x.data, requires_grad = True)
    g_start = t.zeros(1).to(device) #save initial gradient
    g_end = t.zeros(1).to(device) #save end gradient

    #ADAM parameters
    alpha = config['langevin_step_size_des']
    beta_1 = 0.5
    beta_2 = 0.999						#initialize the values of the parameters
    epsilon = 1e-8
    m_t = 0 
    v_t = 0 

    for i in range(config['langevin_step_num_des']):
      x_feature = self.descriptor(x) #return energy of x
      grad = t.autograd.grad(x_feature.sum(), [x])[0] #capture gradient

      #using code https://github.com/sagarvegad/Adam-optimizer/blob/master/Adam.py
      if config['minimizer'] == 'adam': 
        g_t = grad		#computes the gradient of the stochastic function
        m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
        m_cap = m_t/(1-(beta_1**(i+1)))		#calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**(i+1)))		#calculates the bias-corrected estimates
        x_prev = x								
        x = x_prev + (alpha*m_cap)/(t.sqrt(v_cap)+epsilon)	#updates the parameters

      #alternatively run SGD with/without momementum/noise
      if config['minimizer'] == 'sgd':
        if (i==0):
          sgd_m_t = grad
        else:
          sgd_m_t = config['sgd_beta']*sgd_m_t + (1-config['sgd_beta'])*grad
        x = x + config['langevin_step_size_des']*sgd_m_t #add gradient
        if config['with_noise']=='yes':
          x = x + 1e-2 * t.randn_like(x) #add noise to maximization

      if (i==0):
        g_start = grad.view(grad.shape[0], -1).norm(dim=1).mean() #store start gradient

    g_end = grad.view(grad.shape[0], -1).norm(dim=1).mean() #store end gradient
    return x, g_start, g_end

  #Main training function
  def train(self):

    #initialize optimizers
    des_optimizer = t.optim.Adam(self.descriptor.parameters(), lr=config['lr_des'], weight_decay = config['weight_decay'],
                                         betas=[config['beta1_des'], 0.999])
    gen_optimizer = t.optim.Adam(self.generator.parameters(), lr=config['lr_gen'],
                                         betas=[config['beta1_gen'], 0.999])
    mse_loss = t.nn.MSELoss(size_average=False, reduce=True)

    start_time = time.time()
    #store diagnostics as empty
    gen_loss_epoch, des_loss_epoch, recon_loss_epoch = [], [], [] #losses
    energy_data = [] #energy of data
    energy_start, energy_end = [], [] #energy at start and end of GD 
    gradient_start, gradient_end = [], [] #gradient at start and end of GD

    #train for 'num_iter'
    for i in range(config['num_iter']):
      
      #sample observations and add noise
      obs_data = sample_q(config['batch_size'])
      random_noise = t.randn(config['batch_size'], config['num_channels'], config['img_size'], config['img_size'])
      obs_data = obs_data + random_noise*config['data_epsilon']
      obs_data = Variable(obs_data.cuda())

      # G0
      #sample from generator
      if config['init'] == 'generator':
        z = t.randn(config['batch_size'], config['z_size'], 1, 1)
        z = Variable(z.cuda(), requires_grad=True)
        gen_res = self.generator(z)

      # other sampling methods
      if config['init'] == 'gaussian':
        gen_res = t.randn([config['batch_size'], config['num_channels'], config['img_size'], config['img_size']]).cuda()
      if config['init'] == 'uniform':
        gen_res = (2*(t.rand([config['batch_size'], config['num_channels'], config['img_size'], config['img_size']])) - 1).cuda()
      if config['init'] == 'data':
        gen_res = sample_q(config['batch_size'])
      if config['init'] == 'persistent':
        gen_res, rand_inds = sample_image_set(s_t_0, config['batch_size'])
        random_noise = t.randn(config['batch_size'], config['num_channels'], config['img_size'], config['img_size'])
        gen_res = (gen_res + random_noise*config['data_epsilon']).cuda()
      gen_res_copy = gen_res.detach().clone()
          
      # D1
      if config['langevin_step_num_des'] > 0:
        revised, g_start, g_end = self.langevin_dynamics_descriptor(gen_res)
      # G1
      if config['init'] == 'generator':
        if config['langevin_step_num_gen'] > 0:
          z = self.langevin_dynamics_generator(z, revised)
      # D2 - collect energies
      obs_feature = self.descriptor(obs_data)
      revised_feature = self.descriptor(revised)
      
      #compute description loss as difference in energies - ensures matching gradient from EBM theory
      des_loss = (revised_feature.mean() - obs_feature.mean()).sum() 
      
      #if values get too large, stop training
      if abs(des_loss.detach().cpu().numpy()) > 1e+11:
        print('error diverged ')
        plot_ims(EXP_DIR + 'shortrun/' + 'mcmc_{:>06d}_failiure.png'.format(i+1), revised)
        plot_ims(EXP_DIR + 'shortrun/' + 'init_{:>06d}_failure.png'.format(i+1), gen_res[0:config['batch_size']])
        plot_ims(EXP_DIR + 'shortrun/' + 'data_{:>06d}_failure.png'.format(i+1), obs_data)
        break

      #iterate training of descriptor model
      des_optimizer.zero_grad()
      des_loss.backward()
      des_optimizer.step()

      # G2
      #iterate training of generator
      ini_gen_res = gen_res.detach() # 
      if config['init'] == 'generator':
        if config['langevin_step_num_gen'] > 0:
          gen_res = self.generator(z)
        gen_loss = 1.0 / (2.0 * config['sigma_gen'] * config['sigma_gen']) * mse_loss(gen_res,
                                                                                      revised.detach())

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
        gen_loss_epoch.append(gen_loss.cpu().data)
        recon_loss = mse_loss(revised, ini_gen_res)
        recon_loss_epoch.append(recon_loss.cpu().data)

      #store diagnostics
      des_loss_epoch.append(des_loss.cpu().data)
      energy_data.append(obs_feature.mean().cpu().data)
      energy_end.append(revised_feature.mean().cpu().data)
      energy_start.append(self.descriptor(gen_res_copy).mean().cpu().data)
      gradient_start.append(g_start.cpu().data)
      gradient_end.append(g_end.cpu().data)

      #modify learning rate as model continues
      if config['lr_decrease'] == 'yes':
        for param_group in des_optimizer.param_groups:
          param_group['lr'] = param_group['lr']*.999

      #persistent update
      if config['init'] == 'persistent':
        # update persistent image bank
        s_t_0[rand_inds] = revised.detach().cpu().clone()
          
      #store synthesized images as training runs
      if (i + 1) == 1 or (i + 1) % 200 == 0:
        # visualize synthesized images
        if config['init'] == 'generator':
          print('gen_loss')
          print(gen_loss)
        print('des_loss')
        print(des_loss)
        plot_ims(EXP_DIR + 'shortrun/' + 'mcmc_{:>06d}.png'.format(i+1), revised)
        plot_ims(EXP_DIR + 'shortrun/' + 'init_{:>06d}.png'.format(i+1), gen_res[0:config['batch_size']])
        plot_ims(EXP_DIR + 'shortrun/' + 'data_{:>06d}.png'.format(i+1), obs_data)

      #what is outputted here often changes to test how the model makes results 
      #that are a different initialization than trained
      #test different number of steps or different noise initialization
      if (i + 1) == 1 or (i+1) % 1000 == 0 or i == 199 or i == 99 or i == 299 or i == 399:
        long_init = (2*(t.rand([config['batch_size'], config['num_channels'], config['img_size'], config['img_size']])) - 1).cuda() #uniform
        plot_ims(EXP_DIR + 'longrun/' + 'init_{:>06d}.png'.format(i+1), long_init)
        for lgn in range(1):
          long_init, trash1, trash2 = self.langevin_dynamics_descriptor(long_init)
        plot_ims(EXP_DIR + 'longrun/' + 'mcmc_{:>06d}.png'.format(i+1), long_init)

    #save network weights
    t.save(self.descriptor, EXP_DIR + 'checkpoints/' + 'final_descriptor.pth')
    t.save(self.generator, EXP_DIR + 'checkpoints/' + 'final_generator.pth')

    #save diagnostic graphs
    save_data_plot(des_loss_epoch, EXP_DIR, 'des_loss')
    save_data_plot(gen_loss_epoch, EXP_DIR, 'gen_loss')
    save_data_plot(recon_loss_epoch, EXP_DIR, 'recon_loss')
    save_data_plot(energy_data, EXP_DIR, 'energy_data')
    save_data_plot(energy_start, EXP_DIR, 'energy_start')
    save_data_plot(energy_end, EXP_DIR, 'energy_end')
    save_data_plot(gradient_start, EXP_DIR, 'gradient_start')
    save_data_plot(gradient_end, EXP_DIR, 'gradient_end')

  #output test images from completed model
  def test(self, d_path, g_path, test_size, init_type):
        
    if init_type == 'generator':
      self.generator = t.load(g_path).eval()

    self.descriptor = t.load(d_path).eval()

    #initialize data by init_type
    if init_type == 'generator':
      z = t.randn(test_size, config['z_size'], 1, 1)
      z = Variable(z.cuda())
      gen_res = self.generator(z)
    if init_type == 'gaussian':
      gen_res = t.randn([test_size, config['num_channels'], config['img_size'], config['img_size']]).cuda()
    if init_type == 'uniform':
      gen_res = (2*(t.rand([test_size, config['num_channels'], config['img_size'], config['img_size']])) - 1).cuda()
    if init_type == 'persistent':
      gen_res, rand_inds = sample_image_set(s_t_0, test_size)
      gen_res = gen_res.cuda()

    #run through energy minimization
    revised, trash1, trash2 = self.langevin_dynamics_descriptor(gen_res)

    #for i in range(test_size):
    #  plot_ims(EXP_DIR + 'test_sample/' + 'init_{:>06d}.png'.format(i+1), revised[i,:,:,:])
    #save numpy file for FID computation
    np.save(EXP_DIR+'test_model.npy', revised.detach().cpu().numpy())

    print ('===Image generation done.===')

##########################
# Run Models
##########################

model=CoopNets()
model.train()
model.test(EXP_DIR+'checkpoints/final_descriptor.pth',EXP_DIR+'checkpoints/final_generator.pth', config['test_size'], config['init'])
