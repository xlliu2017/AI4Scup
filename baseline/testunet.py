import torch
from torch import nn
from collections import OrderedDict
from typing import List, Tuple, Union
from testMG2 import MgNO

class Unet2015(nn.Module):
    """Simplified 2D UNet model."""

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()
        
        in_channels = time_history * (n_input_scalar_components + n_input_vector_components * 2)
        out_channels = time_future * (n_output_scalar_components + n_output_vector_components * 2)
        features = hidden_channels

        # Encoder blocks
        self.encoder1 = self._block(in_channels, features, activation)
        self.encoder2 = self._block(features, features * 2, activation)
        self.encoder3 = self._block(features * 2, features * 4, activation)
        self.encoder4 = self._block(features * 4, features * 8, activation)
        self.bottleneck = self._block(features * 8, features * 16, activation)

        # Decoder blocks
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block(features * 8 * 2, features * 8, activation)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 4 * 2, features * 4, activation)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 2 * 2, features * 2, activation)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, activation)

        # Final conv layer
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.view(x.size(0), -1, *x.shape[3:])  # Flatten time and channel dimensions

        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoding path
        dec4 = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        return self.conv(dec1).view(x.size(0), -1, *x.shape[2:])

    @staticmethod
    def _block(in_channels, features, activation):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            activation,
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            activation,
        )


def test_unet2015():
    # Define the parameters for the model
    n_input_scalar_components = 1
    n_input_vector_components = 1
    n_output_scalar_components = 1
    n_output_vector_components = 1
    time_history = 4
    time_future = 2
    hidden_channels = 16
    activation = nn.ReLU()

    # Create a model instance
    model = Unet2015(
        n_input_scalar_components=n_input_scalar_components,
        n_input_vector_components=n_input_vector_components,
        n_output_scalar_components=n_output_scalar_components,
        n_output_vector_components=n_output_vector_components,
        time_history=time_history,
        time_future=time_future,
        hidden_channels=hidden_channels,
        activation=activation
    )

    # Create a random input tensor with shape (batch_size, time_history, channels, height, width)
    batch_size = 2
    height, width = 256, 256
    input_tensor = torch.randn(batch_size, time_history, n_input_scalar_components + n_input_vector_components * 2, height, width)
    print(f"Input shape: {input_tensor.shape}")

    # Pass the input tensor through the model
    output = model(input_tensor)

    # # Define the expected output shape
    # expected_output_shape = (batch_size, time_future, n_output_scalar_components + n_output_vector_components * 2, height, width)

    # # Check if the output shape matches the expected shape
    # assert output.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, but got {output.shape}"
    print(f"Test passed! Output shape is {output.shape}.")




class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu", norm: bool = False, n_groups: int = 1):
        super().__init__()
        # Use simple activation assignment
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity()

        # if norm:
        #     self.norm1 = nn.GroupNorm(n_groups, in_channels)  # Normalize using `in_channels`
        #     self.norm2 = nn.GroupNorm(n_groups, out_channels)  # Normalize using `out_channels`
        # else:
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.conv1(self.activation(self.norm1(x)))
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Down block with ResidualBlock and optional AttentionBlock."""

    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu", norm: bool = False):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, activation=activation, norm=norm)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Up block that combines ResidualBlock and handles concatenated skip connections."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, activation: str = "gelu", norm: bool = False):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResidualBlock(in_channels + skip_channels, out_channels, activation=activation, norm=norm)

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor):
        x = self.up_conv(x)
        x = torch.cat([x, skip_connection], dim=1)  # Concatenate skip connection
        return self.res(x)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

# from .activations import ACTIVATION_REGISTRY
# from .fourier import SpectralConv2d

# Largely based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License
# Copyright (c) 2020 Varuna Jayasiri


class ResidualBlock(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation = self.get_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        # Second convolution layer
        h = self.conv2(self.activation(self.norm2(h)))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class FourierResidualBlock(nn.Module):
    """Fourier Residual Block to be used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes1 (int): Number of modes in the first dimension.
        modes2 (int): Number of modes in the second dimension.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.modes1 = modes1
        self.modes2 = modes2

        self.fourier1 = SpectralConv2d(in_channels, out_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros")
        self.fourier2 = SpectralConv2d(out_channels, out_channels, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, padding_mode="zeros")
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # using pre-norms
        h = self.activation(self.norm1(x))
        x1 = self.fourier1(h)
        x2 = self.conv1(h)
        out = x1 + x2
        out = self.activation(self.norm2(out))
        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2 + self.shortcut(x)
        return out


class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention]

    Args:
        n_channels (int): the number of channels in the input
        n_heads (int): the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups (int): the number of groups for [group normalization][torch.nn.GroupNorm].
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class DownBlock(nn.Module):
    """Down block This combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, activation=activation, norm=norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class FourierDownBlock(nn.Module):
    """Down block This combines [`FourierResidualBlock`][pdearena.modules.twod_unet.FourierResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.res = FourierResidualBlock(
            in_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
            activation=activation,
            norm=norm,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Up block that combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, activation=activation, norm=norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class FourierUpBlock(nn.Module):
    """Up block that combines [`FourierResidualBlock`][pdearena.modules.twod_unet.FourierResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Note:
        We currently don't recommend using this block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = FourierResidualBlock(
            in_channels + out_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
            activation=activation,
            norm=norm,
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, n_channels: int, has_attn: bool = False, activation: str = "gelu", norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, activation=activation, norm=norm)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, activation=activation, norm=norm)

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class Unet(nn.Module):
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Args:
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels.
        hidden_channels (int): Number of channels in the hidden layers.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        ch_mults (list): List of channel multipliers for each resolution.
        is_attn (list): List of booleans indicating whether to use attention blocks.
        mid_attn (bool): Whether to use attention block in the middle block.
        n_blocks (int): Number of residual blocks in each resolution.
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers.
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        hidden_channels: int,
        activation: str,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
    ) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_channels = hidden_channels

        self.activation = self.get_activation(activation)
        
        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = input_channel
        n_channels = hidden_channels
        # Project image into feature map
        if use1x1:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, has_attn=mid_attn, activation=activation, norm=norm)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, has_attn=is_attn[i], activation=activation, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = output_channel
        #
        if use1x1:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(0, 0), stride=2)

        num_iteration = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        resolution = 127
        in_channels = 32
        out_channels = 32
        self.mgno = MgNO(4, out_channels, in_channels, num_iteration, resolution=resolution).to('cuda')

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def print_size(self):
        # print the hidden channel 
        return self.hidden_channels


    def forward(self, x: torch.Tensor):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channel, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_channel, height, width].
        """
        assert x.dim() == 4, f"Expected input to be a 4D tensor, got {x.dim()}D tensor instead."
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        x = self.final(self.activation(self.norm(x))) # 127x127
        
        x = self.mgno(x)
        # corp to 120x120
        x = x[:,4:124,4:124]

        return x


class AltFourierUnet(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        hidden_channels: int,
        activation: str,
        modes1: int = 12,
        modes2: int = 12,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        n_fourier_layers: int = 2,
        mode_scaling: bool = True,
        use1x1: bool = False,
    ) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_channels = hidden_channels

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = input_channel
        n_channels = hidden_channels
        # Project image into feature map
        if use1x1:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            if i < n_fourier_layers:
                for _ in range(n_blocks):
                    down.append(
                        FourierDownBlock(
                            in_channels,
                            out_channels,
                            modes1=max(modes1 // 2**i, 4) if mode_scaling else modes1,
                            modes2=max(modes2 // 2**i, 4) if mode_scaling else modes2,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                        )
                    )
                    in_channels = out_channels
            else:
                # Add `n_blocks`
                for _ in range(n_blocks):
                    down.append(
                        DownBlock(
                            in_channels,
                            out_channels,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                        )
                    )
                    in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, has_attn=mid_attn, activation=activation, norm=norm)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            if i < n_fourier_layers:
                for _ in range(n_blocks):
                    up.append(
                        FourierUpBlock(
                            in_channels,
                            out_channels,
                            modes1=max(modes1 // 2**i, 4) if mode_scaling else modes1,
                            modes2=max(modes2 // 2**i, 4) if mode_scaling else modes2,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                        )
                    )
            else:
                for _ in range(n_blocks):
                    up.append(
                        UpBlock(
                            in_channels,
                            out_channels,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                        )
                    )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, has_attn=is_attn[i], activation=activation, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = output_channel
        if use1x1:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Alternative Fourier U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channel, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_channel, height, width].
        """
        assert x.dim() == 4, f"Expected input to be a 4D tensor, got {x.dim()}D tensor instead."
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        x = self.final(self.activation(self.norm(x)))
        return x


class FourierUnet(nn.Module):
    """Unet with Fourier layers in early downsampling blocks.

    Args:
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels.
        hidden_channels (int): Number of channels in the first layer.
        activation (str): Activation function to use.
        modes1 (int): Number of Fourier modes to use in the first spatial dimension.
        modes2 (int): Number of Fourier modes to use in the second spatial dimension.
        norm (bool): Whether to use normalization.
        ch_mults (list): List of integers to multiply the number of channels by at each resolution.
        is_attn (list): List of booleans indicating whether to use attention at each resolution.
        mid_attn (bool): Whether to use attention in the middle block.
        n_blocks (int): Number of blocks to use at each resolution.
        n_fourier_layers (int): Number of early downsampling layers to use Fourier layers in.
        mode_scaling (bool): Whether to scale the number of modes with resolution.
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layer.
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        hidden_channels: int,
        activation: str,
        modes1: int = 12,
        modes2: int = 12,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        n_fourier_layers: int = 2,
        mode_scaling: bool = True,
        use1x1: bool = False,
    ) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_channels = hidden_channels
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = input_channel
        n_channels = hidden_channels
        # Project image into feature map
        if use1x1:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = nn.Conv2d(insize, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            if i < n_fourier_layers:
                for _ in range(n_blocks):
                    down.append(
                        FourierDownBlock(
                            in_channels,
                            out_channels,
                            modes1=max(modes1 // 2**i, 4) if mode_scaling else modes1,
                            modes2=max(modes2 // 2**i, 4) if mode_scaling else modes2,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                        )
                    )
                    in_channels = out_channels
            else:
                # Add `n_blocks`
                for _ in range(n_blocks):
                    down.append(
                        DownBlock(
                            in_channels,
                            out_channels,
                            has_attn=is_attn[i],
                            activation=activation,
                            norm=norm,
                        )
                    )
                    in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, has_attn=mid_attn, activation=activation, norm=norm)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, has_attn=is_attn[i], activation=activation, norm=norm))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = output_channel
        if use1x1:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Fourier U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channel, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_channel, height, width].
        """
        assert x.dim() == 4, f"Expected input to be a 4D tensor, got {x.dim()}D tensor instead."
        x = self.image_proj(x)

        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)

        x = self.middle(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x)

        x = self.final(self.activation(self.norm(x)))
        return x


if __name__ == "__main__":
    import torch

    # Example usage of the revised Unet class

    # Define the model parameters
    input_channels = 6      # e.g., RGB image
    output_channels = 32     # e.g., Grayscale output
    hidden_channels = 32
    activation = "gelu"
    norm = True
    ch_mults = (1, 2, 2, 4)
    is_attn = (False, False, False, False)
    mid_attn = False
    n_blocks = 2
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
    ).to('cuda')
    # count number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create a dummy input tensor
    dummy_input = torch.randn(8, input_channels, 256, 256).to('cuda')  # [batch_size, channels, height, width]

    # Forward pass
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")  # Expected: [8, output_channels, 256, 256]




