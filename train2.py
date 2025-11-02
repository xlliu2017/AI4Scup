# this is a new model for channel expansion for coarse level and change the iteration for efficient computation but performance seems to be worse than the previous model

from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

import yaml
import argparse 
from bisect import bisect
import os
import torch
import shutil
import warnings
import wandb
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from dataset2 import datasetFactory
# os.environ['NCCL_P2P_DISABLE']='1'
wandb.init(mode = 'disabled')
#wandb.init(project="lbs")
# from model import FNO
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




class CosineAnnealingWarmRestartsDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, decay_factor=0.9):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_min = eta_min
        self.T_i = T_0
        self.decay_factor = decay_factor
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestartsDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == -1:
            return self.base_lrs

        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult

        return [self.base_eta_min + (base_lr * (self.decay_factor ** self.cycle) - self.base_eta_min) *
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        if self.last_epoch == -1:
            return self.base_lrs

        cos = (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return [self.base_eta_min + (base_lr * (self.decay_factor ** self.cycle) - self.base_eta_min) * cos
                for base_lr in self.base_lrs]

class LpLoss(nn.Module):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    

class RRMSE(object):
    def __init__(self, ):
        super(RRMSE, self).__init__()
        
    def __call__(self, x, y):
        num_examples = x.size()[0]
        norm = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), 2 , 1)**2
        normy = torch.norm( y.view(num_examples,-1), 2 , 1)**2
        mean_norm = torch.mean((norm/normy)**(1/2))
        return mean_norm
    
##########################################################################################
#fourier convolution 2d block
class fourier_conv_2d(nn.Module):
    def __init__(self, in_, out_, wavenumber1, wavenumber2):
        super(fourier_conv_2d, self).__init__()
        self.out_ = out_
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_ * out_))
        self.weights1 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, 2 , dtype=torch.float32))
        self.weights2 = nn.Parameter(scale * torch.rand(in_, out_, wavenumber1, wavenumber2, 2 , dtype=torch.float32))
        # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ,2), (in_channel, out_channel, x,y,2) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
    def forward(self, x):
        #input: batch,channel,x,y
        #out: batch,channel,x,y
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.view_as_real(torch.fft.rfft2(x))#input: batch,channel,x,y->batch,channel,x,y,2
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1,2, dtype=torch.float32, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2,:], self.weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)))
        return x

class Fourier_layer(nn.Module):
    def __init__(self,  features_, wavenumber, activation = 'relu', is_last = False):
        super(Fourier_layer, self).__init__()
        self.W =  nn.Conv2d(features_, features_, 1)
        self.fourier_conv = fourier_conv_2d(features_, features_ , *wavenumber)
        if is_last== False: 
            self.act = F.gelu
        else: 
            self.act = nn.Identity()
    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return self.act(x1 + x2) 
def get_grid2D(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    
    return torch.cat((gridx, gridy), dim=-1).to(device)

def set_activ(activation):
    if activation is not None: 
        activation = activation.lower()
    if activation == 'relu':
        nonlinear = F.relu
    elif activation == "leaky_relu": 
        nonlinear = F.leaky_relu
    elif activation == 'tanh':
        nonlinear = F.tanh
    elif activation == 'sine':
        nonlinear= torch.sin
    elif activation == 'gelu':
        nonlinear= F.gelu
    elif activation == 'elu':
        nonlinear = F.elu_
    elif activation == None:
        nonlinear = nn.Identity()
    else:
        raise Exception('The activation is not recognized from the list')
    return nonlinear

    
##########################################
# Fully connected Layer
##########################################
class FCLayer(nn.Module):
    """Fully connected layer """
    def __init__(self, in_feature, out_feature, 
                        activation = "gelu",
                        is_normalized = True): 
        super().__init__()
        self.LinearBlock = nn.Linear(in_feature,out_feature)
        self.act = set_activ(activation)
    def forward(self, x):
        return self.act(self.LinearBlock(x))
##########################################
# Fully connected Block
##########################################
class FC_nn(nn.Module):
    r"""Simple MLP to code lifting and projection"""
    def __init__(self, sizes = [2, 128, 128, 1], 
                        activation = 'relu',
                        outermost_linear = True, 
                        outermost_norm = True,  
                        drop = 0.):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.net = nn.ModuleList([FCLayer(in_feature= m, out_feature= n, 
                                            activation=activation
                                            )   
                                for m, n in zip(sizes[:-2], sizes[1:-1])
                                ])
        if outermost_linear == True: 
            self.net.append(FCLayer(sizes[-2],sizes[-1], activation = None
                                    ))
        else: 
            self.net.append(FCLayer(in_feature= sizes[-2], out_feature= sizes[-1], 
                                    activation=activation
                                    ))
    def forward(self,x):
        for module in self.net:
            x = module(x)
            x = self.dropout(x)
        return x

######################################################
#fourier neural operator implementation
class FNO(pl.LightningModule):
    def __init__(self,     
                    wavenumber, features_, 
                    padding = 9, 
                    activation= 'relu',
                    lifting = None, 
                    proj =  None, 
                    dim_input = 4,  
                    loss = "rel_l2",
                    learning_rate = 1e-2, 
                    step_size= 100,
                    gamma= 0.5,
                    weight_decay= 1e-5,
                    eta_min = 5e-4,
                    normalize_param = None
                    ):
        super(FNO, self).__init__()
        self.padding = padding   
        self.layers = len(wavenumber)
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.eta_min = eta_min

        self.mean_sos, self.std_sos, self.mean_field, self.std_field = normalize_param
        if loss == 'l1':
            self.criterion = nn.L1Loss()
            self.criterion_val = LpLoss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            self.criterion_val = LpLoss()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
            self.criterion_val = LpLoss()
        elif loss == "rel_l2":
            self.criterion =LpLoss()
            self.criterion_val = RRMSE()
        
        if lifting is None:
            self.lifting = FC_nn([dim_input, features_//2, features_], 
                                    activation = "relu",
                                    outermost_norm=False
                                    )
        else: 
            self.lifting = lifting
        if  proj is None: 
            self.proj =  FC_nn([features_, features_//2, 2], 
                                activation = "relu",
                                outermost_norm=False
                                    )
        else: 
            self.proj = proj
        self.fno = []
        for l in range(self.layers-1):
            self.fno.append(Fourier_layer(features_ = features_, 
                                        wavenumber=[wavenumber[l]]*2, 
                                        activation = activation))
        
        self.fno.append(Fourier_layer(features_=features_, 
                                        wavenumber=[wavenumber[-1]]*2, 
                                        activation = activation,
                                        is_last= True))
        self.fno =nn.Sequential(*self.fno)
        self.val_iter = 0

    def forward(self, sos,theta):
        # 100,480,480,2
        #x = torch.cat((sos, src), dim=-1)
        
        #sos = (sos - self.mean_sos.to(sos.device)) / self.std_sos.to(sos.device)  
        x = sos
        grid = get_grid2D(x.shape, x.device)
        x = torch.cat((x,grid,theta), dim=-1) # 100,960,960,4
        field = theta.clone()
        x = self.lifting(x)  # 100,960,960,feature_
        x = x.permute(0, 3, 1, 2)# batch,feature,x,y:   100,feature_,960,960
        x = nn.functional.pad(x, [0,self.padding, 0,self.padding]) 
        x = self.fno(x)# batch,feature,x,y:   100,feature_,960+pad,960+pad
        x = x[..., :-self.padding, :-self.padding] # batch,feature,x,y:   100,feature_,960,960
        x = x.permute(0, 2, 3, 1 )
        x =self.proj(x)#*self.std_field.to(sos.device) +  self.mean_field.to(sos.device)   # batch,x,y,2:   100,960,960 ,2
        x =torch.view_as_real(torch.view_as_complex(field.to(x.device))*(1+torch.view_as_complex(x)))
        return x

    def training_step(self, batch: torch.Tensor, batch_idx):    
        sos,theta,y = batch
        batch_size = sos.shape[0]
        #y = y-self.homo_field.unsqueeze(0) # NEW
        
        #y = (y - self.mean_field.to(sos.device)) / self.std_field.to(sos.device)
        #out = (self(sos,theta) - self.mean_field.to(sos.device))/ self.std_field.to(sos.device)
        out = self(sos,theta)
        loss = self.criterion(out.view(batch_size,-1),y.view(batch_size,-1))#torch.mean(torch.abs(out.view(batch_size,-1)-10*y.view(batch_size,-1)) ** 2)
        #loss = torch.mean(torch.abs(out.view(batch_size,-1)- y.view(batch_size,-1)) ** 2)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"loss": loss.item()})
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        self.val_iter += 1
        sos,theta,y= val_batch
        batch_size = sos.shape[0]
        #out = self(sos,src)+10*self.homo_field.unsqueeze(0) #new
        out = self(sos,theta)
        val_loss = self.criterion_val(out.view(batch_size,-1),y.view(batch_size,-1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_loss": val_loss.item()})
        if self.val_iter %19 ==0:
            #self.log_wandb_image(wandb,sos[0].detach().cpu(),(y-self.homo_field.unsqueeze(0))[0].detach().cpu(),(out-10*self.homo_field.unsqueeze(0))[0].detach().cpu())
            self.log_wandb_image(wandb,sos[0].detach().cpu(),y[0].detach().cpu(),out[0].detach().cpu())
        return val_loss

    def log_wandb_image(self,wandb,  sos, field, pred_field):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax = ax.flatten()
        ax0 = ax[0].imshow(sos[...,0], cmap="inferno")
        ax[0].set_title("Sound speed")
        ax[1].imshow(field[...,0], cmap="seismic")
        ax[1].set_title("Field")
        ax[2].imshow(pred_field[...,0], cmap="seismic")
        ax[2].set_title("Predicted field")
        img = wandb.Image(plt)
        wandb.log({'Image': img})
        plt.close()

    def configure_optimizers(self, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if  scheduler is None:
            #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.step_size, eta_min= self.eta_min)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }

class Conv_Dyn(nn.Module):
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=1, bias=False, padding_mode='replicate', resolution=480):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.conv_3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.conv_4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.ln = nn.LayerNorm([resolution, resolution], elementwise_affine=True)
   

    def forward(self, out):
        u, f, a, diva = out
        if diva is None:
            diva = self.conv_0(F.tanh((self.conv_1(a))))
        # f = self.conv_3(f) - diva * self.conv_2(u) 
        f = self.conv_2(f - diva * u)
        u = u + self.conv_4(f)                              
    
        out = (u, f, a, diva)
        return out
    
class MgRestriction(nn.Module): 
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=2, padding=0, bias=False, padding_mode='zeros'):
        super().__init__()

        self.R_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.R_2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.R_3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)

    def forward(self, out):
        u_old, f_old, a_old, diva_old = out
        if diva_old is None:
            a_old = self.R_1(a_old)                            
        u = self.R_2(u_old)                              
        f = self.R_3(f_old)   
        out = (u, f, a_old, diva_old)
        return out
    

class MG_fem(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_iteration=[1,1,1,1,1,1]):
        super().__init__()
        self.num_iteration = num_iteration
        self.resolutions = [480, 239, 119, 59, 29, 14, 6]
        self.RTlayers = nn.ModuleList()
        for j in range(len(num_iteration)-1):
            if  j == 0 or j == 5 or j == 6:
                self.RTlayers.append(nn.ConvTranspose2d(in_channels=in_channels*(j+2), out_channels=out_channels*(j+1), kernel_size=4, stride=2, padding=0, bias=False))
            else:
                self.RTlayers.append(nn.ConvTranspose2d(in_channels=in_channels*(j+2), out_channels=out_channels*(j+1), kernel_size=3, stride=2, padding=0, bias=False))

        layers = []
        for l, num_iteration_l in enumerate(num_iteration): #l: l-th layer.   num_iteration_l: the number of iterations of l-th layer
            for i in range(num_iteration_l):
                layers.append(Conv_Dyn(in_channels=in_channels*(l+1), out_channels=out_channels*(l+1), resolution=self.resolutions[l]))
               

            setattr(self, 'layer'+str(l), nn.Sequential(*layers))
            # set attribute. This is equivalent to define
            # self.layer1 = nn.Sequential(*layers)
            # self.layer2 = nn.Sequential(*layers)
            # ...
            # self.layerJ = nn.Sequential(*layers)

            if l < len(num_iteration)-1:
                layers= [MgRestriction(in_channels=in_channels*(l+1), out_channels=out_channels*(l+2))] 
   

    def forward(self, u, f, a, diva_list=[None for _ in range(7)]):

        # u_list = []
        out_list = [0] * len(self.num_iteration)
        # out = (u, f, a) 
        if diva_list[0] is None:
            for l in range(len(self.num_iteration)):
                out = (u, f, a, diva_list[l])
                u, f, a, diva = getattr(self, 'layer'+str(l))(out) 
                out_list[l] = (u, f, a)
                diva_list[l] = diva
        else:
            for l in range(len(self.num_iteration)):
                out = (u, f, a, diva_list[l])
                u, f, a, diva = getattr(self, 'layer'+str(l))(out) 
                out_list[l] = (u, f, a)

        for j in range(len(self.num_iteration)-2,-1,-1):
            u, f, a = out_list[j][0], out_list[j][1], out_list[j][2]
            u_post = u + self.RTlayers[j](out_list[j+1][0])
            out_list[j] = (u_post, f, a)
            
        return out_list[0][0], out_list[0][1], out_list[0][2], diva_list

class MgNO(pl.LightningModule):
    def __init__(self,
                    lifting = None, 
                    proj =  None, 
                    dim_input = 4,  
                    features = 12,
                    loss = "l2",
                    learning_rate = 1e-2, 
                    step_size= 100,
                    gamma= 0.5,
                    weight_decay= 1e-5,
                    eta_min = 5e-4,
                    normalize_param = None,
                    iteration = 1,
                    epochs = 10
                    ):
        super(MgNO, self).__init__()
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.eta_min = eta_min
        self.iteration = iteration

        mean_sos, std_sos =  torch.tensor(1488.3911, dtype = torch.float32), torch.tensor(27.5279, dtype = torch.float32)

        self.register_buffer("mean_sos", mean_sos)
        self.register_buffer("std_sos", std_sos)
      

        if loss == 'l1':
            self.criterion = nn.L1Loss()
            self.criterion_val = LpLoss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            self.criterion_val = RRMSE()
        elif loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
            self.criterion_val = LpLoss()
        elif loss == "rel_l2":
            self.criterion =LpLoss()
            self.criterion_val = RRMSE()
        
        if lifting is None:
            self.lifting_1 = nn.Conv2d(2, features, kernel_size=1, bias=False)
            self.lifting_2 = nn.Conv2d(2, features, kernel_size=1, bias=False)
            self.lifting_3 = nn.Conv2d(1, features, kernel_size=1)
        else: 
            self.lifting = lifting

        if  proj is None: 
            self.proj = nn.Conv2d(features, 2, kernel_size=1)
        else: 
            self.proj = proj
        self.mgno = nn.ModuleList()
        for l in range(6):
            self.mgno.append(MG_fem(in_channels=features, out_channels=features, num_iteration=[1,1,1,1,1,1,]))
        
        
        self.val_iter = 0

    def forward(self, sos, theta):
        # theta is 100,480,480,2
        # normalize
        sos = (sos - self.mean_sos) / (self.std_sos * .6)
      
        u = self.lifting_1(theta)
        f = self.lifting_2(theta) 
        a = self.lifting_3(sos)
 
        u,f,a,diva_list = self.mgno[0](u,f,a,diva_list=[None for _ in range(6)])# batch,feature,x,y:   100,feature_,960+pad,960+pad
        u,f,a,diva_list = self.mgno[1](u,f,a,diva_list=[None for _ in range(6)])
        for _ in range(self.iteration):
            u,f,a,diva_list = self.mgno[1](u,f,a,diva_list)  

        u =self.proj(u)
        u = theta + u
        return u

    def training_step(self, batch: torch.Tensor, batch_idx):    
        sos,theta,y = batch
        batch_size = sos.shape[0]
        #y = y-self.homo_field.unsqueeze(0) # NEW
        
        #y = (y - self.mean_field.to(sos.device)) / self.std_field.to(sos.device)
        #out = (self(sos,theta) - self.mean_field.to(sos.device))/ self.std_field.to(sos.device)
        out = self(sos,theta)
        loss = self.criterion(out.view(batch_size,-1),y.view(batch_size,-1))#torch.mean(torch.abs(out.view(batch_size,-1)-10*y.view(batch_size,-1)) ** 2)
        #loss = torch.mean(torch.abs(out.view(batch_size,-1)- y.view(batch_size,-1)) ** 2)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"loss": loss.item()})
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        self.val_iter += 1
        sos,theta,y= val_batch
        batch_size = sos.shape[0]
        #out = self(sos,src)+10*self.homo_field.unsqueeze(0) #new
        out = self(sos,theta)
        val_loss = self.criterion_val(out.view(batch_size,-1),y.view(batch_size,-1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_loss": val_loss.item()})
        if self.val_iter %19 ==0:
            #self.log_wandb_image(wandb,sos[0].detach().cpu(),(y-self.homo_field.unsqueeze(0))[0].detach().cpu(),(out-10*self.homo_field.unsqueeze(0))[0].detach().cpu())
            self.log_wandb_image(wandb,sos[0].detach().cpu(),y[0].detach().cpu(),out[0].detach().cpu())
        return val_loss

    def log_wandb_image(self,wandb,  sos, field, pred_field):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax = ax.flatten()
        ax0 = ax[0].imshow(sos[...,0], cmap="inferno")
        ax[0].set_title("Sound speed")
        ax[1].imshow(field[...,0], cmap="seismic")
        ax[1].set_title("Field")
        ax[2].imshow(pred_field[...,0], cmap="seismic")
        ax[2].set_title("Predicted field")
        img = wandb.Image(plt)
        wandb.log({'Image': img})
        plt.close()


    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay) # 
        # Create the OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=22080,
            epochs=self.epochs,
            pct_start=0.01,
            anneal_strategy='cos',
            div_factor=0.2,
            final_div_factor=50
        )
        
        # Return the optimizer and scheduler configuration
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # 'step' for step-wise updating
                'frequency': 1,
                'name': 'learning_rate',
            },
        }    
##########################################################################################

def main(config_file):
    torch.set_float32_matmul_precision("medium")
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    c_model = config['model']
    c_train = config['train']
    c_proj = config['Project']
    ################################################################
    # selecting model
    
    if c_proj['checkpoint'] == False:
        save_file = os.path.join(config["ckpt"]["PATH"], 
                                config["ckpt"]["save_dir"])
        checkpoint_callback = ModelCheckpoint(                                
                                dirpath=save_file,
                                every_n_epochs = 1,
                                save_last = True,
                                monitor = 'val_loss',
                                mode = 'min',
                                save_top_k = c_proj['save_top_k'],
                                filename="model-{epoch:03d}-{val_loss:.4f}-{loss_epoch:.4f}",
                            )
    # if os.path.exists(save_file):
    #     print(f"The model directory exists. Overwrite? {c_proj['erase']}")
    #     if c_proj['erase'] == True:
    #         shutil.rmtree(save_file)
    if c_proj['do'] == 'train':
        train_dataloader, val_dataloader = datasetFactory(config=config, do=c_proj['do'])
    elif c_proj['do'] == 'test':
        val_dataloader = datasetFactory(config=config, do=c_proj['do'])

    if c_proj['do'] == 'test':
        model = torch.load('/ibex/ai/home/liux0t/AI4S-cupv2/submission14new.pt')
        # model.load_state_dict(torch.load('/ibex/ai/home/liux0t/AI4S-cupv2/save_files/MgNOv3.1/model-epoch=002-val_loss=0.0075.ckpt')['state_dict'])
        model.criterion_val = LpLoss()
        model.iteration = c_model['iteration']
    elif c_proj['fine_tune'] == True:
        model = torch.load('/ibex/ai/home/liux0t/AI4S-cupv2/submission13.3.pt')
        model.load_state_dict(torch.load('/ibex/ai/home/liux0t/AI4S-cupv2/save_files/MgNO14/last-v2.ckpt')['state_dict'])
        model.learning_rate = c_train['lr']
        model.weight_decay = c_train['weight_decay']
        model.iteration = c_model['iteration']
        model.epochs = c_train['epochs']
        if c_model['loss'] == 'rel_l2':
            model.criterion = LpLoss()
        elif c_model['loss'] == 'l2':
            model.criterion = nn.MSELoss()
        
    else:
        model = MgNO(lifting = None,
                    proj = None,
                    dim_input = 4,
                    features=24,
                    loss = c_model['loss'],
                    learning_rate = c_train['lr'], 
                    step_size= c_train['step_size'],
                    gamma= c_train['gamma'],
                    weight_decay= c_train['weight_decay'],
                    epochs=c_train['epochs'],
                    eta_min= c_train['eta_min'],
                    iteration = c_model['iteration']
                    )
    # model.load_state_dict(torch.load('/ibex/ai/home/liux0t/AI4S-cupv2/submission_11.pt')['state_dict']) #'/ibex/user/liux0t/AI4S-cupv2/save_files/FNO/model-epoch=010-val_loss=0.0311.ckpt'

    
    # print(model)
    
    max_epochs = config["train"]["epochs"]
    lr_monitor = LearningRateMonitor(logging_interval='step')
    csv_logger = CSVLogger("logs")
    if c_proj['devices'] == 1 :
        trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=c_proj['accelerator'], 
                            devices = c_proj['devices'],
                            callbacks = [checkpoint_callback,lr_monitor],
                            logger=csv_logger,
                            )#,
                            #strategy = 'deepspeed',gradient_clip_val=0.8)  # dp ddp deepspeed
    else: 
        device_num = [i for i in range(c_proj['devices'])]
        trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=c_proj['accelerator'], 
                            devices = device_num,
                            callbacks = [checkpoint_callback,lr_monitor],
                            strategy='ddp_find_unused_parameters_true',) #gradient_clip_val=0.8

    if c_proj['do'] == 'train':
        trainer.fit(model, train_dataloader, val_dataloader,)# ckpt_path='/ibex/ai/home/liux0t/AI4S-cupv2/save_files/MgNO5/last.ckpt') #ckpt_path='/ibex/user/liux0t/AI4S-cupv2/save_files/FNO/model-epoch=010-val_loss=0.0311.ckpt' '/ibex/user/liux0t/AI4S-cupv2/save_files/MgNOv3/model-epoch=001-val_loss=0.0078.ckpt'
        if c_proj['save'] == True:
            save_path = os.path.join(c_proj['save_path'], c_proj['save_name'])
            torch.save(model, save_path)
            print('save model done')
    elif c_proj['do'] == 'test':
        trainer.validate(model, val_dataloader)
    else:
        print('Please select the correct operation mode')
        raise Exception('Please select the correct operation mode')



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Training of the Architectures', add_help=True)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='/ibex/user/liux0t/AI4S-cupv2/config/MgNO.yaml')
    args=parser.parse_args()
    config_file = args.config_file
    main(config_file)