import torch
import torch.nn as nn


class MgIte(nn.Module):
    def __init__(self, A, S):
        super().__init__()
 
        self.A = A
        self.S = S

    def forward(self, out):

        u, f = out
        u = u + (self.S(f-self.A(u)))  
        out = (u, f)
        return out

class MgIte_init(nn.Module):
    def __init__(self, S):
        super().__init__()
        
        self.S = S

    def forward(self, f):
        u = self.S(f)
        return (u, f)

class Restrict(nn.Module):
    def __init__(self, Pi=None, R=None, A=None):
        super().__init__()
        self.Pi = Pi
        self.R = R
        self.A = A
    def forward(self, out):
        u, f = out
        f = self.R(f-self.A(u))
        u = self.Pi(u)                              
        out = (u,f)
        return out


class MgConv(nn.Module):
    def __init__(self, num_iteration, out_channel, in_channel, resolution=64, 
                padding_mode='zeros', normalization=True):
        super().__init__()
        self.num_iteration = num_iteration
    
        # Calculate resolutions for downsampling
        self.resolutions = self.calculate_downsampling_levels(resolution, kernel_sizes=[3] * (len(num_iteration) - 1))
        self.upsample_kernels = self.calculate_adjusted_upsample_kernels_simple(self.resolutions[-1], self.resolutions)
        print(self.resolutions)

        # Create normalization layers
        if normalization:
            self.norm_layer_list = nn.ModuleList([
                nn.GroupNorm(1, out_channel, eps=1e-5, affine=True)
                for j in range(len(num_iteration) - 1)
            ])
        else:
            self.norm_layer_list = nn.ModuleList([nn.Identity() for j in range(len(num_iteration) - 1)])

        # Create transposed convolution layers for upsampling
        self.rt_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=self.upsample_kernels[j], stride=2, padding=0, bias=False)
            for j in range(len(num_iteration) - 1)
        ])

        self.layers = nn.ModuleList()
        self.post_smooth_layers = nn.ModuleList()
        
        layer = []
        for l, num_iteration_l in enumerate(num_iteration):  # l: l-th layer. num_iteration_l: iterations of l-th layer
            
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                if l == 0 and i == 0:
                    layer.append(MgIte_init(S))
                else:
                    A = nn.Conv2d(out_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                    layer.append(MgIte(A, S))
            if num_iteration_l[1] != 0:
                for i in range(num_iteration_l[1]):
                    S = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                    A = nn.Conv2d(out_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                    post_smooth_layers.append(MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())

            self.layers.append(nn.Sequential(*layer))
            self.post_smooth_layers.append(nn.Sequential(*post_smooth_layers))

            if l < len(num_iteration) - 1:
                A = nn.Conv2d(out_channel, in_channel, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
                Pi = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
  
                layer = [Restrict(Pi, R, A)]
               
        
    def calculate_downsampling_levels(self, H, kernel_sizes, stride=2, padding=1):
        image_sizes = [H]  # Start with the initial input size
        for kernel_size in kernel_sizes:
            # Apply the correct formula considering padding and stride
            H_out = (H + 2 * padding - kernel_size) // stride + 1
            image_sizes.append(H_out)
            H = H_out  # Update H to the current output size for the next iteration
        return image_sizes


    def calculate_adjusted_upsample_kernels_simple(self, H, downsampling_sizes, stride=2, padding=0):
        adjusted_kernel_sizes = []
        for i in range(len(downsampling_sizes) - 1, 0, -1):
            H_in = downsampling_sizes[i]
            H_out = downsampling_sizes[i - 1]
            kernel_size = H_out - stride * (H_in - 1)
            adjusted_kernel_sizes.append(kernel_size)
        return adjusted_kernel_sizes[::-1]
    
    def forward(self, f):
        out_list = [0] * len(self.num_iteration)
        out = f 

        # Downsampling path
        for l in range(len(self.num_iteration)):
            out = self.layers[l](out)
            out_list[l] = out

        # Upsampling path
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = self.norm_layer_list[j](u + self.rt_layers[j](out_list[j + 1][0]))
            out = (u_post, f)
            out_list[j] = self.post_smooth_layers[j](out) 
            
        return out_list[0][0]

class MgNO(nn.Module):
    def __init__(self, num_layer, out_channel, in_channel, num_iteration, output_dim=1, resolution=64,
                normalizer=None,  activation='gelu', init=False, ):
        super().__init__()
        self.num_layer = num_layer
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.num_iteration = num_iteration

        
        self.conv_list = nn.ModuleList([])   
        for _ in range(num_layer):
            self.conv_list.append(MgConv(num_iteration, out_channel, out_channel, resolution=resolution))
        
        self.last_layer = nn.Conv2d(out_channel, output_dim, kernel_size=1)
        self.normalizer = normalizer 

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else: raise NameError('invalid activation') 
        
    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u))
        return self.normalizer.decode(self.last_layer(u)) if self.normalizer else self.last_layer(u) 

# Test the code
num_iteration = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
resolution = 220
in_channels = 1
out_channels = 1
model = MgNO(6, out_channels, in_channels, num_iteration, resolution=resolution).to('cuda')
# model = MgConv(num_iteration, in_channels, out_channels, resolution=resolution).to('cuda')

x = torch.randn(10, 1, resolution, resolution).to('cuda')

tic = torch.cuda.Event(enable_timing=True)
toc = torch.cuda.Event(enable_timing=True)
tic.record()
with torch.no_grad():
    for i in range(10):
        output = model(x)
toc.record()
torch.cuda.synchronize()
print(tic.elapsed_time(toc))


