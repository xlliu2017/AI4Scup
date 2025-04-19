from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl
import yaml
import argparse 
from bisect import bisect
import os
import torch
import shutil
import warnings
import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
#wandb.init(project="lbs")
#wandb.init(mode = 'disabled')
# os.environ['NCCL_P2P_DISABLE']='1'


class LpLoss(object):
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
        field = theta[...,1:].clone()
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


class Model(nn.Module):
    def __init__(self, model_path,
                 normalize = False,
                 normalizer = None):
        super(Model, self).__init__()
        model_path= os.path.join(model_path, 'submission.pt')
        self.model = torch.load(model_path).cuda()
        self.model.eval()
        self.normalize = normalize

    def forward(self, sos,src):
        '''
        sos: speed of sound map, (batch, h, w)
        src: source angle, (batch,h,w)
        output: predicted wavefield , (batch,h,w,2)
        '''
       
        pred = self.model(sos,src)
     
        return pred