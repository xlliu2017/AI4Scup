# this is improved by chatgpt
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
    


# models.py (continued)

class ConvDyn(nn.Module):
    """
    Convolutional Dynamics Block.
    """
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=1, bias=False, padding_mode='replicate', resolution=480):
        super(ConvDyn, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, padding_mode=padding_mode)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode=padding_mode)
        self.layer_norm = nn.LayerNorm([resolution, resolution])
    
    def forward(self, out):
        u, f, a, diva = out
        if diva is None:
            diva = self.conv0(F.tanh(self.conv1(a)))
        f = self.conv2(f - diva * u)
        u = u + self.conv4(f)
        return u, f, a, diva

class MgRestriction(nn.Module):
    """
    Multigrid Restriction Operator.
    """
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=2, padding=0, bias=False, padding_mode='zeros'):
        super(MgRestriction, self).__init__()
        self.R1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode=padding_mode)
        self.R2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode=padding_mode)
        self.R3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, padding_mode=padding_mode)
    
    def forward(self, out):
        u_old, f_old, a_old, diva_old = out
        if diva_old is None:
            a_old = self.R1(a_old)
        u = self.R2(u_old)
        f = self.R3(f_old)
        return u, f, a_old, diva_old

class MG_FEM(nn.Module):
    """
    Multigrid Finite Element Method Module.
    """
    def __init__(self, in_channels=1, out_channels=1, num_iterations=[1,1,1,1,1,1]):
        super(MG_FEM, self).__init__()
        self.num_iterations = num_iterations
        self.resolutions = [480, 239, 119, 59, 29, 14, 6]
        self.rt_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels * (j + 2), out_channels * (j + 1), kernel_size=4 if j in [0, 5, 6] else 3, stride=2, padding=0, bias=False)
            for j in range(len(num_iterations) - 1)
        ])
        self.dyn_layers = nn.ModuleList([
            nn.Sequential(
                *[ConvDyn(in_channels, out_channels, resolution=res) for _ in range(num_iter)]
            )
            for res, num_iter in zip(self.resolutions, num_iterations)
        ])
        self.restrictions = nn.ModuleList([
            MgRestriction(in_channels, out_channels) for _ in range(len(num_iterations) - 1)
        ])
    
    def forward(self, u, f, a, diva_list=None):
        if diva_list is None:
            diva_list = [None] * len(self.num_iterations)
        
        out_list = []
        # Forward pass
        for l, dyn in enumerate(self.dyn_layers):
            u, f, a, diva = dyn((u, f, a, diva_list[l]))
            out_list.append((u, f, a))
            if l < len(self.restrictions):
                u, f, a, diva_list[l + 1] = self.restrictions[l]((u, f, a, diva_list[l + 1]))
        
        # Backward pass
        for j in reversed(range(len(self.rt_layers))):
            u, f, a = out_list[j]
            u = u + self.rt_layers[j](out_list[j + 1][0])
            out_list[j] = (u, f, a)
        
        return out_list[0][0], out_list[0][1], out_list[0][2], diva_list

class MgNOModel(pl.LightningModule):
    """
    Multigrid Neural Operator (MgNO) Model implemented with PyTorch Lightning.
    """
    def __init__(self, config):
        super(MgNOModel, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        # Extract hyperparameters for easier access
        model_config = config['model']
        train_config = config['train']
        
        # Normalization parameters (register as buffers to ensure they are saved and moved with the model)
        mean_sos = torch.tensor(model_config['mean_sos'], dtype=torch.float32)
        std_sos = torch.tensor(model_config['std_sos'], dtype=torch.float32)
        self.register_buffer("mean_sos", mean_sos)
        self.register_buffer("std_sos", std_sos)
        
        # Define Lifting Layers
        self.lifting_u = nn.Conv2d(2, model_config['features'], kernel_size=1, bias=False)
        self.lifting_f = nn.Conv2d(2, model_config['features'], kernel_size=1, bias=False)
        self.lifting_a = nn.Conv2d(1, model_config['features'], kernel_size=1)
        
        # Define Projection Layer
        self.proj = nn.Conv2d(model_config['features'], 2, kernel_size=1)
        
        # Define Multigrid Layers
        self.mgno = nn.ModuleList([
            MG_FEM(
                in_channels=model_config['features'], 
                out_channels=model_config['features'], 
                num_iterations=model_config['mg_iterations']
            )
            for _ in range(model_config['mg_layers'])
        ])
        
        # Define Loss Functions
        loss_type = model_config['loss']
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
            self.criterion_val = LpLoss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
            self.criterion_val = RRMSE()
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
            self.criterion_val = LpLoss()
        elif loss_type == "rel_l2":
            self.criterion = LpLoss()
            self.criterion_val = RRMSE()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
    def forward(self, sos, theta):
        """
        Forward pass through the MgNO model.
        """
        # Normalize input
        sos_norm = (sos - self.mean_sos) / (self.std_sos * 0.6)
        
        # Apply Lifting Layers
        u = self.lifting_u(theta)
        f = self.lifting_f(theta)
        a = self.lifting_a(sos_norm)
        
        # Initialize diva_list if not provided
        diva_list = [None] * len(self.mgno[0].num_iterations)
        
        # Pass through Multigrid Layers
        for mg in self.mgno:
            u, f, a, diva_list = mg(u, f, a, diva_list)
        
        # Apply Projection Layer
        u = self.proj(u)
        u = theta + u  # Residual connection
        
        return u
    
    def training_step(self, batch, batch_idx):
        """
        Training step executed for each batch.
        """
        sos, theta, y = batch
        out = self(sos, theta)
        loss = self.criterion(out.view(out.size(0), -1), y.view(y.size(0), -1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step executed for each batch.
        """
        sos, theta, y = batch
        out = self(sos, theta)
        val_loss = self.criterion_val(out.view(out.size(0), -1), y.view(y.size(0), -1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        
        
        return val_loss
    
    
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.train.learning_rate, 
            weight_decay=self.hparams.train.weight_decay
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.train.learning_rate,
            steps_per_epoch=self.hparams.train.steps_per_epoch,
            epochs=self.hparams.train.epochs,
            pct_start=0.002,
            anneal_strategy='cos',
            div_factor=4.0,
            final_div_factor=5.0
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
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
        model = torch.load('/ibex/ai/home/liux0t/AI4S-cupv2/submission13.3.pt')
        model.load_state_dict(torch.load('/ibex/ai/home/liux0t/AI4S-cupv2/save_files/MgNOv3.1/model-epoch=002-val_loss=0.0075.ckpt')['state_dict'])
        model.criterion_val = LpLoss()
        model.iteration = c_model['iteration']
    elif c_proj['fine_tune'] == True:
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
                    )
        # print the model's parameters's name required_grad
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
                
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
        trainer.fit(model, train_dataloader, val_dataloader, ) #ckpt_path='/ibex/user/liux0t/AI4S-cupv2/save_files/FNO/model-epoch=010-val_loss=0.0311.ckpt' '/ibex/user/liux0t/AI4S-cupv2/save_files/MgNOv3/model-epoch=001-val_loss=0.0078.ckpt'
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