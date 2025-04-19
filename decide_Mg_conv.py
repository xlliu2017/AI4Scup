# Let's modify the UNet code to print the image size at each level

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, image_size, levels=4, initial_features=64):
        super(UNet, self).__init__()
        self.levels = levels
        self.initial_features = initial_features

        # Encoding layers
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        # Decoding layers
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        features = initial_features
        current_image_size = image_size

        print(f"Initial image size: {current_image_size}")

        # Encoder part
        for i in range(levels):
            self.encoders.append(
                self.conv_block(input_channels if i == 0 else features, features)
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_image_size //= 2  # Image size is halved with each pooling layer
            print(f"After encoder level {i + 1}, image size: {current_image_size}")
            features *= 2  # Double the number of features for the next level

        # Decoder part
        for i in range(levels):
            features //= 2
            kernel_size, padding = self.get_upsample_params(current_image_size)
            self.upsamples.append(
                nn.ConvTranspose2d(features * 2, features, kernel_size=kernel_size, stride=2, padding=padding)
            )
            current_image_size = current_image_size * 2  # Image size is doubled after each upsampling
            print(f"After decoder level {i + 1}, image size: {current_image_size}")

        # Final convolution
        self.final_conv = nn.Conv2d(initial_features, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def get_upsample_params(self, image_size):
        """
        Given the current image size, this function returns a suitable kernel size
        and padding for the ConvTranspose2d layer such that the upsampled image has
        the same size as the encoder output for concatenation.
        """
        if image_size % 2 == 0:
            # For even image size, use kernel size 2 and padding 0 to exactly double the size
            return 2, 0
        else:
            # For odd image size, we need to add 1 padding to ensure sizes match
            return 3, 1

# Example usage
# input_channels = 1
# output_channels = 2
# image_size = 221  # Example image size
# model = UNet(input_channels, output_channels, image_size)
def calculate_downsampling_levels(H, kernel_sizes, stride=2, padding=0):
    image_sizes = [H]  # Initialize with the original image size
    
    for kernel_size in kernel_sizes:
        H_out = (H - kernel_size) // stride + 1
        image_sizes.append(H_out)
        H = H_out  # Update input size for the next level
    
    return image_sizes



def calculate_upsampling_levels(H, kernel_sizes, stride=2, padding=0, output_padding=0):
    image_sizes = [H]  # Initialize with the last downsampling size (input to upsampling)
    
    for kernel_size in kernel_sizes:
        H_out = stride * (H - 1) + kernel_size - 2 * padding + output_padding
        image_sizes.append(H_out)
        H = H_out  # Update input size for the next level
    
    return image_sizes[::-1]  # Reverse the sizes to match the downsampling order



def calculate_adjusted_upsample_kernels_simple(H, downsampling_sizes, stride=2, padding=0):
    adjusted_kernel_sizes = []  # To store the adjusted kernel sizes
    
    # Start from the smallest downsampled size and work upwards
    for i in range(len(downsampling_sizes) - 1, 0, -1):
        H_in = downsampling_sizes[i]
        H_out = downsampling_sizes[i - 1]
        
        # Use the formula to directly match the downsampling sizes by adjusting the kernel size
        kernel_size = H_out - stride * (H_in - 1)
        adjusted_kernel_sizes.append(kernel_size)
    
    return adjusted_kernel_sizes

# Example usage
H_initial = 256  # Initial input image size for downsampling layers
downsample_kernels = [3, 3, 3, 3, 3, 3]  # Kernel sizes for downsampling layers

# Calculate the sizes after downsampling
downsampling_sizes = calculate_downsampling_levels(H_initial, downsample_kernels)
print(f"Downsampling sizes: {downsampling_sizes}")

# Adjust the kernel sizes for the upsampling layers so that the sizes match the downsampling sizes
adjusted_upsample_kernels = calculate_adjusted_upsample_kernels_simple(downsampling_sizes[-1], downsampling_sizes)
print(f"Adjusted upsampling kernel sizes: {adjusted_upsample_kernels}")

# Now calculate the upsampling sizes using the adjusted kernel sizes
upsampling_sizes = calculate_upsampling_levels(downsampling_sizes[-1], adjusted_upsample_kernels)
print(f"Upsampling sizes: {upsampling_sizes}")

# Check if the downsampling sizes match the upsampling sizes (excluding the original size)
match = downsampling_sizes == upsampling_sizes
print(f"Do the sizes match? {match}")




