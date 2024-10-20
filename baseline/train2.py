import copy
import json
import os
import random
import sys
from timeit import default_timer as timer
from PIL import Image
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
# from GPUtil.GPUtil import getGPUs
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from NIOModules import NIOHelmPermInv
from NIMgNO import NIMgNOHelmPermInv
from Baselines import InversionNetHelm
from mgno import MgNO
from testunet import Unet
from debug_tools import CudaMemoryDebugger
from my2 import MyDataset

wandb.init(project="NIO")
# wandb.init(mode = 'disabled')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seed = 3407
random.seed(seed)  # python random generator
np.random.seed(seed)  # numpy random generator
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#  CUDA_VISIBLE_DEVICES=0 python RunNio.py Example mine nio_new 0

# sys.argv  ['RunNio.py', 'Example', 'mine', 'nio_new', '0']
folder = '/ibex/user/liux0t/AI4S_recons/baseline/weights/nio_new'
freq_print = 4 # 4
add_ssim = True

training_properties_ = {
    "step_size": 15,
    "gamma": 1,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.0008,
    "norm": "minmax",
    "weight_decay": 1e-06,
    "reg_param": 0.0,
    "reg_exponent": 2,
    "inputs": 2,
    "b_scale": 0.,
    "retrain": 775,
    "mapping_size_ff": 32,
    "scheduler": "step",
    "ssim_ratio": 0.0
}

branch_architecture_ = {
    "n_hidden_layers": 2,
    "neurons": 50,
    "act_string": "tanh",
    "dropout_rate": 0.0,
    "kernel_size": 3
}

trunk_architecture_ = {
    "n_hidden_layers": 8,
    "neurons": 100,
    "act_string": "leaky_relu",
    "dropout_rate": 0.0,
    "n_basis": 25
}

fno_architecture_ = {
    "width": 32,
    "modes": 25,
    "n_layers": 4,
}

denseblock_architecture_ = {
    "n_hidden_layers": 2,
    "neurons": 50,
    "act_string": "tanh",
    "retrain": 127,
    "dropout_rate": 0.0
}

    # problem = sys.argv[2]  # mine
    # mod = sys.argv[3]  # nio_new
    # max_workers = int(sys.argv[4])  # 0
    # mod = "nio_new"
max_workers = 10

if torch.cuda.is_available():
    memory_avail = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("Running on ", torch.cuda.get_device_name(0), "Total memory: ", memory_avail, " GB")

print_mem = False
disable = False
full_training = False


def log_wandb_image(wandb, sos, field, pred_field):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(sos)
    axes[0].set_title("Sound speed")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(field, cmap="inferno")
    axes[1].set_title("Field")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(pred_field, cmap="inferno")
    axes[2].set_title("Predicted field")
    fig.colorbar(im3, ax=axes[2])
    img = wandb.Image(plt)
    wandb.log({'Image': img})
    plt.close()


step_size = training_properties_["step_size"]
gamma = training_properties_["gamma"]
norm = training_properties_["norm"]
epochs = training_properties_["epochs"]
batch_size = training_properties_["batch_size"]
learning_rate = training_properties_["learning_rate"]
weight_decay = training_properties_["weight_decay"]
reg_param = training_properties_["reg_param"]
reg_exponent = training_properties_["reg_exponent"]
inputs_bool = training_properties_["inputs"]
retrain_seed = training_properties_["retrain"]
b_scale = training_properties_["b_scale"]
mapping_size = training_properties_["mapping_size_ff"]
scheduler_string = training_properties_["scheduler"]
ssim_ratio = training_properties_["ssim_ratio"]

dict_hp = training_properties_.copy()
branch_architecture_copy = branch_architecture_.copy()

branch_architecture_copy["n_hidden_layers_b"] = branch_architecture_copy.pop("n_hidden_layers")
branch_architecture_copy["dropout_rate_b"] = branch_architecture_copy.pop("dropout_rate")
branch_architecture_copy["neurons_b"] = branch_architecture_copy.pop("neurons")
branch_architecture_copy["act_string_b"] = branch_architecture_copy.pop("act_string")

dict_hp.update(branch_architecture_copy)
dict_hp.update(trunk_architecture_)
dict_hp.update(fno_architecture_)
######################################################################################
use_mask = True
######################################################################################
resize_size = (120,120) #initially 120
# resize_size = (300, 300)
ratio = 1  # 256/ratio is input.shape[2] 32
output_bg = 0

data = dict()
val = dict()
test = dict()
# base_dir_dobs = "/data/zhengyihang/direct_inverse_data/dobs_masked/train"
# base_dir_speed = "/data/zhengyihang/direct_inverse_data/speed/train"

# base_dir_dobs = "/home/pub/zhengyihang/direct_inverse_data/dobs_masked/train"
# base_dir_speed = "/home/pub/zhengyihang/direct_inverse_data/speed/train"

# base_dir_dobs = "/bohr/direct-inverse-data-6uvk/v1/300k"
# base_dir_speed = "/bohr/direct-inverse-data-6uvk/v1/speed/train"

base_dir_dobs_300k = "/ibex/user/liux0t/AI4S_recons/dobs_300k_train"
base_dir_dobs_400k = "/ibex/user/liux0t/AI4S_recons/dobs_400k_train"
base_dir_dobs_500k = "/ibex/user/liux0t/AI4S_recons/dobs_500k_train"
mask = np.load("/ibex/user/liux0t/AI4S_recons/mask.npy")

base_dir_speed = "/ibex/user/liux0t/AI4S_recons/speed_train"
max_input = -np.inf
max_output = -np.inf
min_input = np.inf
min_output = np.inf
ind = 0
all_inp = None
all_out = None
train_ind = 0
val_ind = 0
for i in os.listdir(base_dir_dobs_300k):
    index = i.split("_")[-1].split(".")[0]
    if ind % 200 == 1:
        if not use_mask:
            print("load", ind, "unmasked samples")
        if use_mask:
            print("load", ind, "masked samples")
    if ind % 5:

        data["sample_" + str(train_ind)] = dict()

        inp_300k = np.load(os.path.join(base_dir_dobs_300k, i))
        real_300k = inp_300k.real
        imag_300k = inp_300k.imag
        inputs_300k = np.stack((real_300k, imag_300k))  # 2,256,256
        inputs_300k = np.stack([inputs_300k[:, :, ratio * kk] for kk in range(inputs_300k.shape[1] // ratio)], axis=-1)  # 2,256,32
        
        inp_400k = np.load(os.path.join(base_dir_dobs_400k, i))
        real_400k = inp_400k.real
        imag_400k = inp_400k.imag
        inputs_400k = np.stack((real_400k, imag_400k))  # 2,256,256
        inputs_400k = np.stack([inputs_400k[:, :, ratio * kk] for kk in range(inputs_400k.shape[1] // ratio)], axis=-1)  # 2,256,32

        inp_500k = np.load(os.path.join(base_dir_dobs_500k, i))
        real_500k = inp_500k.real
        imag_500k = inp_500k.imag
        inputs_500k = np.stack((real_500k, imag_500k))  # 2,256,256
        inputs_500k = np.stack([inputs_500k[:, :, ratio * kk] for kk in range(inputs_500k.shape[1] // ratio)], axis=-1)  # 2,256,32

        inputs = np.stack([inputs_300k,inputs_400k,inputs_500k], axis= 0)
        if use_mask:
            inputs = inputs * mask
        # print(inputs.shape) # (3,2,256,256)
        data["sample_" + str(train_ind)]["input"] = inputs  # 2,256,32
        img = Image.fromarray(np.load(os.path.join(base_dir_speed, "train_" + index + ".npy"))[240 - 150:240 + 150,
                                240 - 150:240 + 150])
        data["sample_"+str(train_ind)]["output"] = np.array(img.resize(resize_size))-output_bg
        # data["sample_" + str(train_ind)]["output"] = np.array(img) - output_bg
    else:
        val["sample_" + str(val_ind)] = dict()

        inp_300k = np.load(os.path.join(base_dir_dobs_300k, i))
        real_300k = inp_300k.real
        imag_300k = inp_300k.imag
        inputs_300k = np.stack((real_300k, imag_300k))  # 2,256,256
        inputs_300k = np.stack([inputs_300k[:, :, ratio * kk] for kk in range(inputs_300k.shape[1] // ratio)], axis=-1)  # 2,256,32
        
        inp_400k = np.load(os.path.join(base_dir_dobs_400k, i))
        real_400k = inp_400k.real
        imag_400k = inp_400k.imag
        inputs_400k = np.stack((real_400k, imag_400k))  # 2,256,256
        inputs_400k = np.stack([inputs_400k[:, :, ratio * kk] for kk in range(inputs_400k.shape[1] // ratio)], axis=-1)  # 2,256,32

        inp_500k = np.load(os.path.join(base_dir_dobs_500k, i))
        real_500k = inp_500k.real
        imag_500k = inp_500k.imag
        inputs_500k = np.stack((real_500k, imag_500k))  # 2,256,256
        inputs_500k = np.stack([inputs_500k[:, :, ratio * kk] for kk in range(inputs_500k.shape[1] // ratio)], axis=-1) # 2,256,32
        
        inputs = np.stack([inputs_300k,inputs_400k,inputs_500k], axis= 0)
        if use_mask:
            inputs = inputs * mask
        # print(inputs.shape)
        
        val["sample_" + str(val_ind)]["input"] = inputs
        img = Image.fromarray(np.load(os.path.join(base_dir_speed, "train_" + index + ".npy"))[240 - 150:240 + 150,
                                240 - 150:240 + 150])
        val["sample_"+str(val_ind)]["output"] = np.array(img.resize(resize_size))-output_bg # 120,120
        # val["sample_" + str(val_ind)]["output"] = np.array(img) - output_bg  # 120,120
        val_ind += 1
        ind += 1
        continue
    if ind == 1:
        all_inp = data["sample_" + str(train_ind)]["input"][np.newaxis, :]
        all_out = data["sample_" + str(train_ind)]["output"][np.newaxis, :]
        ind += 1
    else:
        all_inp = np.stack(data["sample_" + str(train_ind)]["input"][np.newaxis, :], 0)
        all_out = np.stack(data["sample_" + str(train_ind)]["output"][np.newaxis, :], 0)
        ind += 1

    if np.max(data["sample_" + str(train_ind)]["input"]) > max_input:
        max_input = np.max(data["sample_" + str(train_ind)]["input"])
    if np.min(data["sample_" + str(train_ind)]["input"]) < min_input:
        min_input = np.min(data["sample_" + str(train_ind)]["input"])
    if np.max(data["sample_" + str(train_ind)]["output"]) > max_output:
        max_output = np.max(data["sample_" + str(train_ind)]["output"])
    if np.min(data["sample_" + str(train_ind)]["output"]) < min_output:
        min_output = np.min(data["sample_" + str(train_ind)]["output"])
    train_ind += 1
mean_inp_fun = np.mean(all_inp, 0)  # 2,256,256
mean_out_fun = np.mean(all_out, 0)  # 120,120
std_inp_fun = np.std(all_inp, 0)
std_out_fun = np.std(all_out, 0)
file = dict()
# input (2, 256, 256)  output (120, 120)

print(len(data), len(val), len(test), min_output, max_output)  # 5760 1440 0  1408.692 1595.1279
file["training"] = data
file["validation"] = val
file["max_inp"] = max_input
file["max_out"] = max_output
file["min_inp"] = min_input
file["min_out"] = min_output
file["mean_inp_fun"] = mean_inp_fun
file["mean_out_fun"] = mean_out_fun
file["std_inp_fun"] = std_inp_fun
file["std_out_fun"] = std_out_fun
x = np.linspace(0, 1, resize_size[0])
y = np.linspace(0, 1, resize_size[1])
file["grid"] = np.zeros((resize_size[0], resize_size[1], 2))
file["grid"][:, :, 0] = np.meshgrid(x, y)[1]
file["grid"][:, :, 1] = np.meshgrid(x, y)[0]

fno_input_dimension = denseblock_architecture_["neurons"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = MyDataset(norm=norm, device=device, which="training", data_dict=file,
                          samples=5760)
test_dataset = MyDataset(norm=norm, device=device, which="validation", noise=0.1,
                         data_dict=file, samples=1440)
inp_dim_branch = train_dataset.inp_dim_branch
n_fun_samples = train_dataset.n_fun_samples

grid = train_dataset.get_grid().squeeze(0)

# if not os.path.isdir(folder):
    # print("Generated new folder")
    # # os.mkdir(folder)
    # df = pd.DataFrame.from_dict([training_properties_]).T
    # df.to_csv(folder + '/training_properties.txt', header=False, index=True, mode='a')
    # df = pd.DataFrame.from_dict([branch_architecture_]).T
    # df.to_csv(folder + '/branch_architecture.txt', header=False, index=True, mode='a')
    # df = pd.DataFrame.from_dict([trunk_architecture_]).T
    # df.to_csv(folder + '/trunk_architecture.txt', header=False, index=True, mode='a')
    # df = pd.DataFrame.from_dict([fno_architecture_]).T
    # df.to_csv(folder + '/fno_architecture.txt', header=False, index=True, mode='a')
    # df = pd.DataFrame.from_dict([denseblock_architecture_]).T
    # df.to_csv(folder + '/denseblock_architecture.txt', header=False, index=True, mode='a')

    # print("Using FCNIO")
# model = NIOHelmPermInv(input_dimensions_branch=inp_dim_branch,
#                         input_dimensions_trunk=grid.shape[2],
#                         network_properties_branch=branch_architecture_,
#                         network_properties_trunk=trunk_architecture_,
#                         fno_architecture=fno_architecture_,
#                         device=device,
#                         retrain_seed=retrain_seed,
#                         fno_input_dimension=fno_input_dimension)  # 用这个model
# model = InversionNetHelm(start=32).to(device)

# num_iteration = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
# resolution = 256
# in_channels = 6
# out_channels = 48
# model = MgNO(4, out_channels, in_channels, num_iteration, resolution=resolution).to('cuda')

input_channels = 6      # e.g., RGB image
output_channels = 32     # e.g., Grayscale output
hidden_channels = 32
activation = "gelu"
norm = True
ch_mults = (1, 2, 2, 2, 4)
is_attn = (False, False, False, False, False)
mid_attn = False
n_blocks = 3
use1x1 = False

# Initialize the model
model = Unet(
    input_channel=input_channels,
    output_channel=output_channels,
    hidden_channels=hidden_channels,
    activation=activation,
    norm=norm,
    ch_mults=ch_mults,
    is_attn=is_attn,
    mid_attn=mid_attn,
    n_blocks=n_blocks,
    use1x1=use1x1,
)

start_epoch = 0
best_model_testing_error = 100
best_model = None

model.to(device)
size = model.print_size()
f = open(folder + "/size.txt", "w")
print(size, file=f)

batch_acc = 32
if torch.cuda.is_available():
    print("cuda available")
    batch_acc = batch_acc * torch.cuda.device_count()

print("Maximum number of workers: ", max_workers)
training_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=max_workers, pin_memory=True)
print("training set length:", len(training_set))
testing_set = DataLoader(test_dataset, batch_size=40, shuffle=True, num_workers=max_workers, pin_memory=True)
n_iter_per_epoch = int((train_dataset.length + 1) / batch_size)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def ssim_loss(pred, target, window_size=11, reduction='mean'):
    K1 = 0.01
    K2 = 0.03
    L = 1  # Dynamic range of pixel values (assumed to be 1)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    window = torch.tensor([[[[0.0625, 0.125, 0.0625],
                             [0.125, 0.25, 0.125],
                             [0.0625, 0.125, 0.0625]]]], dtype=torch.float32)

    window = window.expand(pred.shape[1], 1, -1, -1).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=pred.shape[1])
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=target.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.shape[1]) - mu1_mu2

    SSIM_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    SSIM_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    SSIM = SSIM_n / SSIM_d

    if reduction == 'mean':
        return torch.mean((1 - SSIM) / 2)
    elif reduction == 'sum':
        return torch.sum((1 - SSIM) / 2)
    else:
        return (1 - SSIM) / 2


if scheduler_string == "cyclic":
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False,
                                                  step_size_up=int(n_iter_per_epoch / 2) * epochs, mode="triangular2")
elif scheduler_string == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_iter_per_epoch, gamma=gamma)
elif scheduler_string == "cycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=n_iter_per_epoch,
                                                    epochs=epochs, pct_start=0.3, div_factor=5, final_div_factor=50)
else:
    raise ValueError
# if os.path.isfile(folder + "/optimizer_state.pkl"):
#     checkpoint = torch.load(folder + "/optimizer_state.pkl")
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
p = 1
if p == 2:
    my_loss = torch.nn.MSELoss()
elif p == 1:
    my_loss = torch.nn.L1Loss()
else:
    raise ValueError("Choose p = 1 or p=2")

loss_eval = torch.nn.L1Loss()
writer = SummaryWriter(log_dir=folder)

lr_all = list()
training_all = list()

counter = 0
patience = int(0.1 * epochs)
time_per_epoch = 0
for epoch in range(start_epoch, epochs + start_epoch):
    print("mem allocated:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024)
    bar = tqdm(unit="batch", disable=disable)
    with bar as tepoch:
        start = timer()
        tepoch.set_description(f"Epoch {epoch}")
        train_mse = 0.0
        running_relative_train_mse = 0.0
        model.train()
        grid = grid.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        for step, (input_batch, output_batch) in enumerate(training_set):
            # print("herere")
            # if torch.cuda.is_available():
            #     mem = str(round(getGPUs()[0].memoryUtil, 2) * 100) + "%"
            # else:
            #     mem = str(0.) + "%"

            tepoch.update(1)
            input_batch = input_batch.to(device, non_blocking=True)
            output_batch = output_batch.to(device, non_blocking=True)

            pred_train = model(input_batch) #, grid

            loss_f = my_loss(pred_train, output_batch) / torch.mean(abs(output_batch) ** p) ** (1 / p)
            # if add_ssim:
            #     ssimloss = ssim_loss(pred_train.unsqueeze(1),output_batch.unsqueeze(1))
            #     loss_f += ssim_ratio * ssimloss

            # if reg_param != 0:
            #     loss_f += reg_param * model.regularization(reg_exponent)

            loss_f.backward()

            ########################################################################################
            # Evaluation
            ########################################################################################
            train_mse = train_mse * step / (step + 1) + loss_f / (step + 1)
            wandb.log({"train_mse": train_mse.item()})
            wandb.log({"lr": scheduler.get_last_lr()[0]})
            tepoch.set_postfix({'Batch': step + 1,
                                'Train loss (in progress)': train_mse.item(),
                                'lr': scheduler.get_last_lr()[0],
                                # "GPU Mem": mem,
                                "Patience:": counter,
                                })
            if (step + 1) % int(batch_size / batch_acc) == 0 or (step + 1) == len(training_set):
                optimizer.step()  # Now we can do an optimizer step
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if p == 1:
            writer.add_scalar("train_loss/L1 Error", train_mse, epoch)
        if p == 2:
            writer.add_scalar("train_loss/L2 Error", train_mse, epoch)
        end = timer()
        elapsed = end - start
        ########################################################################################
        # Evaluation
        ########################################################################################
        if epoch == 0:
            running_relative_test_mse = 0.0
        if epoch % freq_print == 0:
            if not full_training:
                running_relative_test_mse = 0.0

                model.eval()
                with torch.no_grad():
                    for step, (input_batch, output_batch) in enumerate(testing_set):
                        input_batch = input_batch.to(device, non_blocking=True)
                        # print(input_batch.shape)
                        output_batch = output_batch.to(device, non_blocking=True)
                        pred_test = model(input_batch) #, grid
                        # print(pred_test.shape)
                        pred_test = train_dataset.denormalize(pred_test)
                        output_batch = train_dataset.denormalize(output_batch)
                        loss_test = loss_eval(pred_test, output_batch) / loss_eval(
                            torch.zeros_like(output_batch).to(device), output_batch)
                        running_relative_test_mse = running_relative_test_mse * step / (
                                    step + 1) + loss_test.item() ** (1 / p) * 100 / (step + 1)
                        # log_wandb_image(wandb, input_batch[0, :, 0, 0, :].detach().cpu().numpy(),
                        #                 output_batch[0, :, :].detach().cpu().numpy(),
                        #                 pred_test[0, :, :].detach().cpu().numpy())
                    writer.add_scalar("val_loss/Relative Testing Error", running_relative_test_mse, epoch)
                    wandb.log({"val_loss": running_relative_test_mse})
            else:
                running_relative_test_mse = train_mse.item()

            if running_relative_test_mse < best_model_testing_error:
                best_model_testing_error = running_relative_test_mse
                # torch.save(model, folder + "/model.pkl")
                torch.save(model, folder + "/model.pt")
                # writer.add_scalar("val_loss/Best Relative Testing Error", best_model_testing_error, epoch)
                # writer.add_scalar("time/Elapsed", elapsed, epoch)
                counter = 0
            else:
                counter += 1
        else:
            # torch.save(model, folder + "/model.pkl")
            torch.save(model, folder + "/model.pt")
            print("save pt success")

        time_per_epoch = time_per_epoch * epoch / (epoch + 1) + elapsed / (epoch + 1)

        with open(folder + '/errors.txt', 'w') as file:
            file.write("Training Error: " + str(train_mse.item()) + "\n")
            file.write("Testing Error: " + str(running_relative_test_mse) + "\n")
            file.write("Best Testing Error: " + str(best_model_testing_error) + "\n")
            file.write("Current Epoch: " + str(epoch) + "\n")
            file.write("Time per Epoch: " + str(time_per_epoch) + "\n")
            file.write("Workers: " + str(max_workers) + "\n")

        torch.save({'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()},
                   folder + "/optimizer_state.pkl")
        tepoch.set_postfix({"Val loss": running_relative_test_mse})
        tepoch.close()

    # if counter > patience:
    #    print("Early stopping:", epoch, counter)
    #    break

writer.flush()
writer.close()
