import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random
class MyDataset(Dataset):
    def __init__(self, norm, device, which, noise=0, samples=7200, data_dict=None, deactivate_random = False):
        print("Training with ", samples, " samples")
        if data_dict == None:
            print("should have a data_dict")
            exit()
        else:
            self.data_dict = data_dict[which]
        # self.mod = mod
        self.noise = noise
        self.length = samples
        self.start = 0
        self.which = which
        self.grid = data_dict["grid"]

        self.mean_inp = torch.from_numpy(data_dict['mean_inp_fun'][:, :]).type(torch.float32)
        self.mean_out = torch.from_numpy(data_dict['mean_out_fun'][:, :]).type(torch.float32)
        self.std_inp = torch.from_numpy(data_dict['std_inp_fun'][:, :]).type(torch.float32)
        self.std_out = torch.from_numpy(data_dict['std_out_fun'][:, :]).type(torch.float32)
        self.min_data = torch.tensor(data_dict["min_inp"])
        self.max_data = torch.tensor(data_dict["max_inp"])
        self.min_model = torch.tensor(data_dict["min_out"])
        self.max_model = torch.tensor(data_dict["max_out"])

        self.inp_dim_branch = 4
        self.n_fun_samples = 100

        self.norm = norm
        # self.inputs_bool = inputs_bool

        self.device = device

        self.min_data_logt = torch.log(self.min_data)
        self.max_data_logt = torch.log(self.max_data)
        self.deactivate_random =  deactivate_random 

    def __len__(self):
        return self.length
    def add_awgn(self, signal, snr_dB):
        # 计算信号的功率
        signal_power = torch.mean(signal**2)
        # 计算噪声的功率
        snr_linear = 10**(snr_dB / 10.0)
        noise_power = signal_power / snr_linear
        # 生成高斯白噪声
        noise = torch.sqrt(noise_power) * torch.randn_like(signal)
        # 将噪声添加到信号中
        noisy_signal = signal + noise
        return noisy_signal

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.data_dict['sample_' + str(index)]["input"][:]).type(torch.float32)
        # inputs = self.inputs[index]
        labels = torch.from_numpy(self.data_dict['sample_' + str(index)]["output"][:]).type(torch.float32)
        # labels = self.outputs[index]
        #inputs = inputs * (1 + self.noise * torch.randn_like(inputs))
        if not self.deactivate_random: 
            random_number = random.random()
            if random_number < 0.33:
                inputs = self.add_awgn(inputs,10)
            elif random_number < 0.66:
                inputs = self.add_awgn(inputs,5)
            else:
                inputs = inputs
        else:
            print('noise free')
            inputs = inputs
        


        if self.norm == "norm":
            inputs = self.normalize(inputs, self.mean_inp, self.std_inp)
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "norm-inp":
            inputs = self.normalize(inputs, self.mean_inp, self.std_inp)
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "log-minmax":
            inputs = (np.log1p(np.abs(inputs))) * np.sign(inputs)
            inputs = 2 * (inputs - self.min_data_logt) / (self.max_data_logt - self.min_data_logt) - 1.
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "norm-out":
            inputs = 2 * (inputs - self.min_data) / (self.max_data - self.min_data) - 1.
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "minmax":
            inputs = 2 * (inputs - self.min_data) / (self.max_data - self.min_data) - 1.
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "none":
            inputs = inputs
            labels = labels
        else:
            raise ValueError()
        # inputs [3, 2, 256, 256]
        # print(inputs.shape)
        # inputs = inputs.view(1, 2, 256, 768).permute(3, 0, 1, 2)
        # inputs = inputs.view(1, 2, 256, 256).permute(3, 0, 1, 2)
        # inputs = inputs.view(1, 2, 256, 64).permute(3, 0, 1, 2) # 128,1,2,64
        inputs = inputs.view(6, 256, 256)
        return inputs, labels

    def normalize(self, tensor, mean, std):
        return (tensor - mean) / (std + 1e-16)

    def denormalize(self, tensor):
        if self.norm == "norm" or self.norm == "norm-out":
            return tensor * (self.std_out + 1e-16).to(self.device) + self.mean_out.to(self.device)
        elif self.norm == "none":
            return tensor
        else:
            return (self.max_model - self.min_model) * (tensor + torch.tensor(1., device=self.device)) / 2 + self.min_model.to(self.device)

    def get_grid(self):
        grid = torch.from_numpy(self.grid).type(torch.float32)

        return grid.unsqueeze(0)



class myDataset(Dataset):
    def __init__(self, device, which, data_dict=None):
        if data_dict == None:
            print("should have a data_dict")
            exit()
        else:
            self.data_dict = data_dict[which]
        # self.mod = mod
        self.length = 1
        self.start = 0
        self.which = which
        self.grid = data_dict["grid"]

        self.min_data = torch.tensor(data_dict["min_inp"])
        self.max_data = torch.tensor(data_dict["max_inp"])
        self.min_model = torch.tensor(data_dict["min_out"])
        self.max_model = torch.tensor(data_dict["max_out"])

        self.inp_dim_branch = 4
        self.n_fun_samples = 100

        self.device = device


    def __len__(self):
        return self.length
 
    def __getitem__(self, index):
        inputs = torch.from_numpy(self.data_dict["input"][:]).type(torch.float32)
        # inputs = self.inputs[index]
        # labels = torch.from_numpy(self.data_dict['sample_' + str(index)]["output"][:]).type(torch.float32)
        # labels = self.outputs[index]
        inputs = inputs
        

        inputs = 2 * (inputs - self.min_data) / (self.max_data - self.min_data) - 1.
        labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.

        inputs = inputs.view(1, 2, 256, 768).permute(3, 0, 1, 2)
        print(inputs.shape)
        print("ready")
        return inputs

    def normalize(self, tensor, mean, std):
        return (tensor - mean) / (std + 1e-16)

    def denormalize(self, tensor):
        return (self.max_model - self.min_model) * (tensor + torch.tensor(1., device=self.device)) / 2 + self.min_model.to(self.device)

    def get_grid(self):
        grid = torch.from_numpy(self.grid).type(torch.float32)

        return grid.unsqueeze(0)