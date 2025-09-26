# -*- coding: utf-8 -*-
"""
Basic script for calculating the number of parameters in the model before and after optimization.

File created on Tue Oct 29 2024

@author: Mason Jones
    
"""

import torch
import torch.nn.functional as F
from torch import nn
        
class Block(nn.Module):
    r""" 
    The primary building block of our surrogate model 
    Based on ConvNeXt Block. ConvNeXt is under the MIT license.
    There are two equivalent implementations:
    (1) DwConv -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C) -> Linear -> GELU -> Linear -> Permute back
    We use (1) for simplicity
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0 WAS NOT TESTED
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        dropout (float): Stochastic dropout rate in linear layers. Default: 0.3
        kernelSize (int): Size of kernel for convolution layers. Default: 7 
        kernelPad (int): Amount of padding to use during convolution layers, keep set to maintain shape. Default: 3
        linearScale (int): Scaling factor for the number of channels in the linear layers. Default: 4
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, dropout = 0.3, kernelSize = 7, kernelPad = 3, linearScale = 4):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernelSize, padding=kernelPad, groups=dim) # depthwise convolutions
        self.pwconv1 = nn.Conv3d(dim, linearScale * dim, kernel_size=1, padding=0, groups=1) # pointwise/1x1 convs, replacing linear layer
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(linearScale * dim, dim, kernel_size=1, padding=0, groups=1) # pointwise/1x1 convs, replacing linear layer
        
        # Scaling factor for skip connection
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1, 1)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
                                    
        # Stochastic layer dropout implementation. Was not tested
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Stochastic dropout
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        
        x = input + self.drop_path(x)
        return x

class Model(nn.Module):
    r""" 
    The Surrogate Model Class
    
    Based on ConvNeXt, which is under the MIT license
    A PyTorch impl of : `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 9
        num_classes (int): Number of classes for classification head. Default: 50
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0. WAS NOT TESTED
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=9, num_classes=51, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.0, 
                 layer_scale_init_value=1e-6, head_init_scale=1., kernel_size = [7,7,7,7], kernel_pad = [3,3,3,3], linear_scale=[4,4,4,4], optim_config=None
                 ):
        super().__init__()
        
        if optim_config is not None:
            depths = [optim_config["Layer1_depth"], optim_config["Layer2_depth"], optim_config["Layer3_depth"], optim_config["Layer4_depth"]]
            dims = [optim_config["Channels1"], optim_config["Channels2"], optim_config["Channels3"], optim_config["Channels4"]]
            kernel_size = [int(optim_config["Layer1_kernel"]*2+1), int(optim_config["Layer2_kernel"]*2+1), int(optim_config["Layer3_kernel"]*2+1), int(optim_config["Layer4_kernel"]*2+1)]
            kernel_pad = [optim_config["Layer1_kernel"], optim_config["Layer2_kernel"], optim_config["Layer3_kernel"], optim_config["Layer4_kernel"]]
            linear_scale = [optim_config["Layer1_scale"], optim_config["Layer2_scale"], optim_config["Layer3_scale"], optim_config["Layer4_scale"]]
            # downsample_kernel = [optim_config["Downsample1_kernel"], int(optim_config["Downsample2_kernel"]*2+1), int(optim_config["Downsample3_kernel"]*2+1), int(optim_config["Downsample4_kernel"]*2+1)]
            # downsample_pad = [optim_config["Downsample1_kernel"], optim_config["Downsample2_kernel"], optim_config["Downsample3_kernel"], optim_config["Downsample4_kernel"]]
                    
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=5, stride=5),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=3, stride=1, padding = 1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        self.depth = 0
        for i in range(4):
            if depths[i] != 0:
                self.depth+=1
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value, kernelSize=kernel_size[i], kernelPad=kernel_pad[i], linearScale=linear_scale[i]) for j in range(depths[i])]
                )
                self.stages.append(stage)
                cur += depths[i]
            else:
                dims[i]=dims[i-1]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
        # This is just a silly way of implementing pooling
        # calculates the mean on a channel by channel basis
        # don't remember my inspiration for doing it this way
        self.mean_conv = nn.Conv3d(in_channels=dims[-1],
                      out_channels=dims[-1],
                      kernel_size=5, padding=0, groups = dims[-1], bias = False) 
        # Set kernel to calculate mean
        mean_kernel_weights = torch.ones((dims[-1],1,5,5,5))/125
        self.mean_conv.weight.data = torch.FloatTensor(mean_kernel_weights)
        self.mean_conv.requires_grad_=False
        
        # Converts channels to number of bins
        self.head = nn.Conv3d(dims[-1], num_classes, kernel_size=1, stride=1, padding = 0)
                
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        
        self.pdf = nn.Softplus()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # Main body of the model
        # Four groupings of blocks, separated by extra convolutions
        # Only very first downsample layer should downsample
        for i in range(self.depth):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        # This is just a silly way of doing average pooling
        x = self.mean_conv(x)
        
        # Do a layer norm
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)
        return x

    def forward(self, x):
        # Main part of model:
        x = self.forward_features(x)
        # Last layer to convert from N channels to N outputs
        x = self.head(x)
        # Activation function to ensure all values positive
        x = self.pdf(x)
        # Normalize values to sum to 1
        x = x/torch.sum(x,dim = 1).unsqueeze(1)
        # Get rid of extra dimension
        x = torch.squeeze(x)
        return x

class LayerNorm(nn.Module):
    r""" 
    Function duplicated from ConvNeXt, which is under the MIT license.
    LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x  


    
       

if __name__ == '__main__':    
    best_so_far = {
    "Layer1_depth": 1,
    "Layer2_depth": 1,
    "Layer3_depth": 3,
    "Layer4_depth": 0,

    "Layer1_kernel": 3,
    "Layer2_kernel": 3,
    "Layer3_kernel": 4,
    "Layer4_kernel": 3,

    "Channels1": 64,
    "Channels2": 96,
    "Channels3": 304,
    "Channels4": 376,

    "Layer1_scale": 7,
    "Layer2_scale": 8,
    "Layer3_scale": 1,
    "Layer4_scale": 3,
    }
    config = best_so_far
    
    # Initialize model
    new_net = Model(optim_config=config)
    old_net = Model()

    num_params_new = sum(p.numel() for p in new_net.parameters() if p.requires_grad)
    num_params_old = sum(p.numel() for p in old_net.parameters() if p.requires_grad)
    delta = num_params_old - num_params_new
    
    print("Num Params before optimization: %d \nNum Params after optimization: %d \nDelta: %d"%(num_params_old,num_params_new,delta))
    

    

