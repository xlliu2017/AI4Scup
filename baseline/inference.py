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
from Baselines import *
from debug_tools import *
from DeepONetModules import *
from FNOModules import *
from my import *
from NIOModules import *
#wandb.init(project="lbs")
#wandb.init(mode = 'disabled')
# os.environ['NCCL_P2P_DISABLE']='1'



class Model(nn.Module):
    def __init__(self, model_path,
                 normalize = False,
                 normalizer = None):
        super(Model, self).__init__()
        self.model_path= os.path.join(model_path, 'model.pt')
        
        model = torch.load(self.model_path, map_location={'OpSquareModules': 'torch'})
        self.model = model.eval()
        
        self.normalize = normalize
        self.use_mask = True

        self.max_data = 1024.9685
        self.min_data = -974.7793
        self.file = dict()
        self.file["max_inp"] = 1024.9685
        self.file["min_inp"] = -974.7793
        self.file["max_out"] = 1593.158
        self.file["min_out"] = 1409.1952

        x = np.linspace(0,1,120)
        y = np.linspace(0,1,120)
        self.file["grid"] = np.zeros((120,120,2))
        self.file["grid"][:,:,0] = np.meshgrid(x,y)[1]
        self.file["grid"][:,:,1] = np.meshgrid(x,y)[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self,dobs_300k,dobs_400k,dobs_500k):
        # mask = np.load("/bohr/AI4S-CUP2-data-45zd/v3/mask.npy")
        # mask = np.load("/root/baseline/mask.npy")
        mask = np.load("./mask.npy")
        mask = torch.tensor(mask)
        dobs_300k = dobs_300k.squeeze()
        dobs_400k = dobs_400k.squeeze()
        dobs_500k = dobs_500k.squeeze()
        
        real_300k = dobs_300k.real
        imag_300k = dobs_300k.imag
        input_300k = torch.stack((real_300k,imag_300k))
        real_400k = dobs_400k.real
        imag_400k = dobs_400k.imag
        input_400k = torch.stack((real_400k,imag_400k))
        real_500k = dobs_500k.real
        imag_500k = dobs_500k.imag
        input_500k = torch.stack((real_500k,imag_500k))
        if self.use_mask:
            input_300k = input_300k * mask
            input_400k = input_400k * mask
            input_500k = input_500k * mask
        inputs = np.stack([input_300k, input_400k, input_500k], axis=0)

        test = dict()
        test["input"] = inputs.astype(np.float32)
        self.file['testing'] = test
        # print("ok")
        test_dataset = myDataset(device=self.device, which="testing", data_dict=self.file)
        grid = test_dataset.get_grid().squeeze(0)
        
        inputs = 2 * (inputs - self.min_data) / (self.max_data - self.min_data) - 1.
        inputs = torch.tensor(inputs)
        inputs = inputs.view(1, 2, 256, 768).permute(3, 0, 1, 2)
        inputs = inputs.unsqueeze(0)
        # print(inputs.shape)
        inputs = inputs.to(self.device)
        grid = grid.to(self.device)
        pred_test = self.model(inputs, grid)
        pred_test = test_dataset.denormalize(pred_test)
        # print(pred_test[0].cpu().shape)
        # print("ok")
        # print(pred_test)
        pred_test= F.interpolate(pred_test.unsqueeze(0), size=(300, 300), mode='bilinear', align_corners=False)
        pred_test = pred_test.squeeze(0)
        # print(pred_test.shape)
        return pred_test