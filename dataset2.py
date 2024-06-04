import argparse 
from bisect import bisect
import os
import torch
import shutil
import warnings
import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader    

class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()
        if x.dtype == torch.cfloat:
            self.mean = torch.mean(torch.view_as_real(x), dim=[0,1,2])
        else:
            self.mean = torch.mean(x, dim=[0,1,2])
        if x.dtype == torch.cfloat:
            self.std = torch.std(torch.view_as_real(x), dim=[0,1,2])
        else:
            self.std = torch.std(x, dim=[0,1,2])
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, sample_idx=None):
        return (x * (self.std + self.eps)) + self.mean

    def cuda(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class File_Loader(Dataset):
    def __init__(self, data_paths, target_paths, size=480):
        self.data_paths = data_paths
        self.target_paths = target_paths
        self.size = size
        self.src = np.load('/ibex/user/liux0t/AI4S-cupv2/u_homo_new.npy')

        self.start_indices = [0] * len(data_paths) #4600
        self.data_count = 0 
        for index in range(len(data_paths)):
            self.start_indices[index] = self.data_count
            self.data_count += 1

        self.start_indices_target = [0] * len(target_paths) 
        self.data_count_target = 0 
        for index in range(len(target_paths)):
            self.start_indices_target[index] = self.data_count_target 
            self.data_count_target += 8

        print('data_count',self.data_count,'data_count_target',self.data_count_target)
    def __len__(self):
        return self.data_count_target

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices_target, index) - 1
        index_in_memmap = index - self.start_indices_target[memmap_index]
        index_1 = index // 32
        index_2 = index - index_1 * 32

        # Load data and target for the given index
        data = np.load(self.data_paths[index_1], mmap_mode='r')[np.newaxis, :, :]
        target = np.load(self.target_paths[memmap_index], mmap_mode='r')[index_in_memmap]
        target = np.concatenate((np.real(target)[np.newaxis, :, :], np.imag(target)[np.newaxis, :, :]), axis=0) / 3000
        theta = np.transpose(np.copy(self.src[index_2]) / 3000, [2, 0, 1])

        # Convert numpy arrays to tensors
        data_tensor = torch.tensor(data, dtype=torch.float)
        target_tensor = torch.tensor(target, dtype=torch.float)
        theta_tensor = torch.tensor(theta, dtype=torch.float)

        return data_tensor, theta_tensor, target_tensor

    

class GettingLists(object):
    def __init__(self,train_num = 6300,
                    valid_num = 900 ,
                    PATH_data = 'lbs', 
                    PATH_target = 'lbs',
                    batchsize= int(2000),
                    ):
        super(GettingLists, self).__init__()
        self.PATH_data = PATH_data
        self.PATH_target = PATH_target
        self.batchsize = batchsize
        self.train_num = train_num
        self.valid_num = valid_num
        self.total_num = train_num+valid_num
        self.velo_list = np.array([os.path.join(self.PATH_data,f'dataset_train_{i+1}/speed',f'train_{k}.npy') for i in range(0,8) for k in range(1+i*900,1+(i+1)*900)])
        self.target_list = np.array([os.path.join(self.PATH_data,f'dataset_train_{i+1}/field',f'train_{k}_{j}.npy') for i in range(0,8) for k in range(1+i*900,1+(i+1)*900) for j in range(1,5)])
        print(len(self.velo_list))
        print(len(self.target_list))
    def get_list(self, do):
        if do == 'train':
            # in_limit_train= np.array([os.path.join(self.PATH_data,f'train_{k}.npy') for k in  self.velo_list_train ])
            # out_limit_train = np.array([os.path.join(self.PATH_target,f'train_{k}_{i}.npy') for k in  self.pressure_list_train for i in self.num_list])
            in_limit_train = np.array(self.velo_list[:self.train_num])
            out_limit_train= np.array(self.target_list[:self.train_num*4])
            
            return in_limit_train, out_limit_train
        elif do == 'validation':
            # in_limit_valid= np.array([os.path.join(self.PATH_data,f'train_{k}.npy') for k in  self.velo_list_test ])
            # out_limit_valid = np.array([os.path.join(self.PATH_target,f'train_{k}_{i}.npy') for k in  self.pressure_list_test for i in self.num_list]) 
            in_limit_valid = np.array(self.velo_list[self.train_num:self.train_num+self.valid_num])
            out_limit_valid= np.array(self.target_list[self.train_num*4:self.train_num*4+self.valid_num*4])
            return  in_limit_valid, out_limit_valid
        elif do =='test':
            # in_limit_test= np.array([os.path.join(self.PATH_data,f'train_{k}.npy') for k in  self.velo_list_test ])
            # out_limit_test = np.array([os.path.join(self.PATH_target,f'train_{k}_{i}.npy') for k in  self.pressure_list_test for i in self.num_list]) 
            in_limit_test = np.array(self.velo_list[self.train_num:self.train_num+self.valid_num])
            out_limit_test= np.array(self.target_list[self.train_num*4:self.train_num*4+self.valid_num*4])
            return in_limit_test, out_limit_test  
        
    def __call__(self, do = 'train'):
        return self.get_list(do)
    def get_dataloader(self,do,config):
        workers = config['data']['load_workers']
        size = config['data']['size']
        batchsize = self.batchsize
        if do == 'train':
            list_x_train, list_y_train = self.__call__('train')
            list_x_valid, list_y_valid = self.__call__('validation')
            Train_Data_set = File_Loader(list_x_train,list_y_train, size = size)
            Valid_Data_set = File_Loader(list_x_valid,list_y_valid, size = size)
            train_loader = DataLoader(dataset = Train_Data_set, 
                                    shuffle = True, 
                                    batch_size = batchsize,
                                    num_workers= workers)
            valid_loader = DataLoader(dataset = Valid_Data_set, 
                                    shuffle = False, 
                                    batch_size =batchsize,
                                    num_workers= workers)
          
            return train_loader, valid_loader
        elif do == 'test':
            list_x_test, list_y_test = self.__call__('test')
            Test_Data_set = File_Loader(list_x_test, list_y_test, size = size)
            test_loader = DataLoader(dataset = Test_Data_set, 
                                    shuffle = False, 
                                    batch_size = batchsize,
                                    num_workers= workers)
            return test_loader

def datasetFactory(config,do='train'):
    c_data = config['data']
    gl = GettingLists(train_num = c_data["train_num"],
                     valid_num = c_data['valid_num'],
                     PATH_data = c_data['PATH_data'],
                     PATH_target= c_data['PATH_target'],
                     batchsize = c_data['batch'])
    return gl.get_dataloader(do = do,config = config)    
