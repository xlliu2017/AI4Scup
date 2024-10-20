import numpy as np
import torch
import torch.nn as nn

from Baselines2 import EncoderHelm2
from DeepONetModules import FeedForwardNN, DeepOnetNoBiasOrg
from testMG2 import MgNO
from testunet import Unet

################################################################

class NIMgNOHelmPermInv(nn.Module):
    def __init__(self,
                 input_dimensions_branch,
                 input_dimensions_trunk,
                 network_properties_branch,
                 network_properties_trunk,
                 fno_architecture,
                 device,
                 retrain_seed,
                 fno_input_dimension=1000,
                 padding_frac=1 / 4):
        super(NIMgNOHelmPermInv, self).__init__()
        output_dimensions = network_properties_trunk["n_basis"]
        fno_architecture["retrain_fno"] = retrain_seed
        network_properties_branch["retrain"] = retrain_seed
        network_properties_trunk["retrain"] = retrain_seed
        # self.fno_inputs = fno_input_dimension
        self.trunk = FeedForwardNN(input_dimensions_trunk, output_dimensions, network_properties_trunk)
        self.fno_layers = fno_architecture["n_layers"]
        print("Using InversionNet Encoder")
        self.branch = EncoderHelm2(output_dimensions)
        self.deeponet = DeepOnetNoBiasOrg(self.branch, self.trunk)
        # self.fc0 = nn.Linear(2 + 2, fno_architecture["width"])
        # self.fc0 = nn.Linear(2 + 1, fno_architecture["width"])
        self.fc0 = nn.Linear(3, fno_architecture["width"])
        # self.correlation_network = nn.Sequential(nn.Linear(2, 50), nn.LeakyReLU(),
        #                                         nn.Linear(50, 50), nn.LeakyReLU(),
        #                                         nn.Linear(50, 1)).to(device)
        num_iteration = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        resolution = 120
        in_channels = 768
        out_channels = 32
        self.mgno = MgNO(4, out_channels, in_channels, num_iteration, resolution=resolution).to('cuda')

        # if self.fno_layers != 0:
        #     self.fno = FNO_WOR(fno_architecture, device=device, padding_frac=padding_frac)

        # self.attention = Attention(70 * 70, res=70 * 70)
        self.device = device

    def forward(self, x, grid):

        # x has shape N x L x nb
        # if self.training:
        #     L = np.random.randint(2, x.shape[1])
        #     idx = np.random.choice(x.shape[1], L)
        #     x = x[:, idx]
        # else:
        L = x.shape[1]

        nx = (grid.shape[0])
        ny = (grid.shape[1])

        grid_r = grid.view(-1, 2)
        x = self.branch(x)
        # x = self.attention(x)

        # x = x.view(x.shape[0], x.shape[1], nx, ny)

        # x = x.reshape(x.shape[0], x.shape[1], nx * ny)

        x = x.view(x.shape[0], x.shape[1], nx, ny)
        x = self.mgno(x)

        # grid = grid.unsqueeze(0)
        # grid = grid.expand(x.shape[0], grid.shape[1], grid.shape[2], grid.shape[3]).permute(0, 3, 1, 2)

        # x = torch.cat((grid, x), 1)

        # weight_trans_mat = self.fc0.weight.data
        # bias_trans_mat = self.fc0.bias.data
        # # weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L), weight_trans_mat[:, 3].view(-1, 1)], dim=1)
        # weight_trans_mat = torch.cat([weight_trans_mat[:, :2], weight_trans_mat[:, 2].view(-1, 1).repeat(1, L) / L], dim=1)
        # # weight_trans_mat = torch.cat([weight_trans_mat.repeat(1, L)], dim=1)
        # x = x.permute(0, 2, 3, 1)
        # input_corr = x[..., np.random.randint(0, L, 2)]
        # out_corr = self.correlation_network(input_corr)
        # x = torch.concat((x, out_corr), -1)
        # x = torch.matmul(x, weight_trans_mat.T) + bias_trans_mat
        # if self.fno_layers != 0:
        #     x = self.fno(x)

        return x

    def print_size(self):
        print("Branch prams:")
        b_size = self.branch.print_size()
        print("Trunk prams:")
        t_size = self.trunk.print_size()
        if self.fno_layers != 0:
            print("FNO prams:")
            f_size = self.mgno.print_size()
        else:
            print("NO FNO")
        # print("Attention prams:")
        # a_size = self.attention.print_size()

        if self.fno_layers != 0:
            size = b_size + t_size + f_size
        else:
            size = b_size + t_size
        print(size)
        return size

    def regularization(self, q):
        reg_loss = 0
        for name, param in self.named_parameters():
            reg_loss = reg_loss + torch.norm(param, q)
        return reg_loss

