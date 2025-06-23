import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDyn(nn.Module):
    """
    Convolutional Dynamics Block.
    """
    def __init__(self, kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=1, bias=False, padding_mode='replicate', resolution=480):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
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
        super().__init__()
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
    def __init__(self, in_channels=1, out_channels=1, num_iterations=[1,1,1,1,1,1], resolution=220):
        super().__init__()
        self.num_iterations = num_iterations
        self.resolutions = self.calculate_downsampling_levels(resolution, kernel_sizes=[3, 3, 3, 3, 3, 3],)
        print(self.resolutions)
        self.upsample_kernels = self.calculate_adjusted_upsample_kernels_simple(self.resolutions[-1], self.resolutions)

        self.rt_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=in_channels , out_channels=out_channels , kernel_size=self.upsample_kernels[j], stride=2, padding=0, bias=False)
            for j in range(len(num_iterations) - 1)
        ])
        
     
        
        self.dyn_layers = nn.ModuleList()
        layers = []
        for res, num_iter in zip(self.resolutions, num_iterations):
            
            # Add dynamic layers (ConvDyn)
            layers.append(
                nn.Sequential(
                    *[ConvDyn(in_channels=in_channels, out_channels=out_channels, resolution=res) for _ in range(num_iter)]
                )
            )
            self.dyn_layers.append(nn.Sequential(*layers))
            # Add restriction layers if not the last layer
            layers = []
            if len(self.dyn_layers) < len(self.resolutions):
                layers.append(MgRestriction(in_channels=in_channels, out_channels=out_channels))

    def calculate_downsampling_levels(self, H, kernel_sizes, stride=2, padding=0):
        image_sizes = [H]  # Initialize with the original image size
        
        for kernel_size in kernel_sizes:
            H_out = (H + 2 * padding - kernel_size) // stride + 1
            image_sizes.append(H_out)
            H = H_out  # Update input size for the next level
        
        return image_sizes


    def calculate_adjusted_upsample_kernels_simple(self, H, downsampling_sizes, stride=2, padding=0):
        adjusted_kernel_sizes = []  # To store the adjusted kernel sizes
        
        # Start from the smallest downsampled size and work upwards
        for i in range(len(downsampling_sizes) - 1, 0, -1):
            H_in = downsampling_sizes[i]
            H_out = downsampling_sizes[i - 1]
            
            # Use the formula to directly match the downsampling sizes by adjusting the kernel size
            kernel_size = H_out - stride * (H_in - 1)
            adjusted_kernel_sizes.append(kernel_size)
        
        return adjusted_kernel_sizes[::-1]
            
    
    def forward(self, u, f, a, diva_list=[None] * 6):
        
        out_list = []
        # Forward pass
        for l, dyn in enumerate(self.dyn_layers):
            u, f, a, diva = dyn((u, f, a, diva_list[l]))
            out_list.append((u, f, a))
            # if l < len(self.restrictions):
            #     u, f, a, diva_list[l + 1] = self.restrictions[l]((u, f, a, diva_list[l + 1]))
        
        # Backward pass
        for j in reversed(range(len(self.rt_layers))):
            u, f, a = out_list[j]
            u = u + self.rt_layers[j](out_list[j + 1][0])
            out_list[j] = (u, f, a)
        
        return out_list[0][0], diva_list

class MgNO(nn.Module):
    def __init__(self,
                    lifting = None, 
                    proj =  None, 
                    dim_input = 4,  
                    features = 12,
                    ):
        super(MgNO, self).__init__()

        mean_sos, std_sos =  torch.tensor(1488.3911, dtype = torch.float32), torch.tensor(27.5279, dtype = torch.float32)

        self.register_buffer("mean_sos", mean_sos)
        self.register_buffer("std_sos", std_sos)
    
        self.lifting_1 = nn.Conv2d(2, features, kernel_size=1, bias=False)
        self.lifting_2 = nn.Conv2d(2, features, kernel_size=1, bias=False)
        self.lifting_3 = nn.Conv2d(1, features, kernel_size=1)
        self.proj = nn.Conv2d(features, 2, kernel_size=1)
     
        self.mgno = nn.ModuleList()
        for l in range(6):
            self.mgno.append(MG_fem(in_channels=features, out_channels=features, num_iteration=[1,1,1,1,1,1,]))
        

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

def benchmark_mg_fem(in_channels, out_channels, input_size, num_iterations, device):
    model = MG_FEM(in_channels, out_channels, num_iterations=num_iterations, resolution=220).to(device)
    model = torch.compile(model)
    input_tensor_u = torch.randn((10, in_channels, input_size, input_size), device=device)
    input_tensor_f = torch.randn((10, in_channels, input_size, input_size), device=device)
    input_tensor_a = None
    resolutions = [480, 239, 119, 59, 29, 14]
    input_tensor_diva = [torch.randn((10, in_channels, resolutions[i], resolutions[i]), device=device) for i in range(len(num_iterations))]

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    ) as prof:
        with torch.no_grad():
            output = model(input_tensor_u, input_tensor_f, input_tensor_a, input_tensor_diva)
        prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Example usage
# device = torch.device("cuda")
# benchmark_mg_fem(32, 32, 480, [1, 1, 1, 1, 1, 1], device)
in_channels = 20
input_size = 480
num_iterations = [1, 1, 1, 1, 1, ]
model = MG_FEM(in_channels=in_channels, out_channels=in_channels, resolution=input_size, num_iterations=num_iterations).to('cuda')
resolutions = model.resolutions
print(model)
# model = torch.compile(model)

device = 'cuda'


u = torch.randn((40, in_channels, input_size, input_size), device=device)
f = torch.randn((40, in_channels, input_size, input_size), device=device)
a = torch.randn((40, in_channels, input_size, input_size), device=device)

diva_list = [torch.randn((40, in_channels, resolutions[i], resolutions[i]), device=device) for i in range(len(num_iterations))]
diva_list = [None] * 6

def run_model():
    # model.eval()
    # record the time
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    tic.record()
    with torch.no_grad():
        for i in range(100):
            output = model(u, f, a, diva_list)
    toc.record()
    torch.cuda.synchronize()
    print(tic.elapsed_time(toc))

run_model()

# Set up the profiler
# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ],
#     record_shapes=True,  # Optional: Records the shape of tensors
#     profile_memory=True,  # Optional: Track memory usage
#     with_stack=True  # Optional: Include stack traces
# ) as prof:
#     with torch.profiler.record_function("model_inference"):
#         run_model()  # Run the model inference you want to profile

# # Print the profiler results
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))






