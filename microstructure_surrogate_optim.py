# -*- coding: utf-8 -*-
"""
This is a script for optimizing the hyperparameters of the microstructure surrogate model.
This script is provided primarily for reference purposes and has not been updated to utilize the more advanced data frameworks of the full training and inference scripts, and does not support multi-GPU training.

Created on Tue May 10 09:37:33 2022

@author: mason
"""

import numpy as np
import pickle
rng=np.random.default_rng()
# import cupy as cp
# import matplotlib.pyplot as plt
import time
import datetime
import meshio
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# from optuna.integration import BoTorchSampler
import optuna

from timm.models.layers import trunc_normal_, DropPath

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler
from typing import Iterator, Sequence

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#.cuda.device(0)

# Percentage of data to be set aside for training (the rest will be used for testing)
training_data_split = 0.9

names = ['L1P100V100', 'L1P105V90', 'L1P105V95', 'L1P100V90', 'L1P90V105', 'L3P75V120', 'L4P124V109']#, 'L1P124V109', 'L4P124V109']

microstructure_files = [name+"_dumps_hist.npy" for name in names]
thermal_files = [name+".xmf" for name in names]

files = zip(thermal_files, microstructure_files)


class SurrogateDataset(Dataset):
    
    def __init__(self, files, training_data_split = 0.99, input_width = 15):
        # Calculate how many voxels of space we need arround the central data point
        T_melt = 1723.0
        self.flipcount=0
        self.rotcount=0
        self.sample=0
        self.half_width = int(np.floor((input_width-1)/2))
        
        self.n_samples = 0
        self.n_training = 0
        self.n_test = 0
        self.test_indices = []
        self.train_indices = []
                
        self.data_dict = {}
        self.data_dict["names"] = []
        name_index = np.int32(0)
        
        self.always_sample = None
        self.num_resamples = 1000
                
        for thermal_file,microstructure_file in files:
            print("Loading ", thermal_file)
            
            # add dataset structure to data_dict
            paramset_name = thermal_file.split(".")[0]
            self.data_dict["names"].append(paramset_name)
            self.data_dict[paramset_name] = {val: [] for val in ["arrayT","arrayM","indices","pad"]}
            
            # Load data files
            microstructure_data = np.float32(np.load(microstructure_file))[:,:,:,:]#10]
                        
            with meshio.xdmf.TimeSeriesReader(thermal_file) as reader:
                points, _ = reader.read_points_cells()
                try:_, point_data, _ = reader.read_data(1)
                except:_, point_data, _ = reader.read_data(0)
        
            # Reshape thermal data
            xdim,ydim,zdim = [len(np.unique(points[:,0])),len(np.unique(points[:,1])),len(np.unique(points[:,2]))]
            
            points = None
            
            for name in point_data: point_data[name] = point_data[name].reshape((xdim,ydim,zdim))
            
            point_data['Max Reheat'][point_data['Melt Count']<1] = 0.0
            
            # check z dimension compatibility of data, remove top layer from microstructure data if applicable
            if microstructure_data.shape[2] > zdim:
                zdiff = microstructure_data.shape[2] - zdim
                if zdiff == 1:
                    microstructure_data = microstructure_data[:,:,:-1,:]
                    # microstructure_data = np.delete(microstructure_data,zdiff,2)
                else:
                    raise Exception("Something went wrong: Microstructure data larger than thermal data")
            elif microstructure_data.shape[2] < zdim:
                raise Exception("Something went wrong: Thermal data larger than microstructure data")
            
            # check x,y dimension compatibility
            if not microstructure_data.shape[0] == xdim:
                raise Exception("Something went wrong: Incompatible x-dimension")
            if not microstructure_data.shape[1] == ydim:
                raise Exception("Something went wrong: Incompatible y-dimension")
                        
            for lowest_melt in range(zdim):
                if np.any(point_data["Melt Count"][:,:,lowest_melt]): break
            
            # Check how much padding is needed on the bottom. Top always gets max padding
            if lowest_melt < self.half_width: 
                z_min_pad = self.half_width - lowest_melt
            else:
                z_min_pad = 0 # if no padding is needed set zero
            
            # Check how much padding is needed on each side in x dimension
            x_melt = []
            for i in range(xdim):
                if np.any(point_data["Melt Count"][i,:,:]):
                    x_melt.append(i)
            min_x_melt = np.amin(x_melt)
            max_x_melt = np.amax(x_melt)
            
            if min_x_melt < self.half_width:
                x_min_pad = self.half_width - min_x_melt
            else:
                x_min_pad = 0
                
            if (xdim - max_x_melt - 1) < self.half_width:
                x_max_pad = self.half_width - (xdim - max_x_melt - 1)
            else:
                x_max_pad = 0
            
            # Check how much padding is needed on each side in y dimension
            y_melt = []
            for i in range(ydim):
                if np.any(point_data["Melt Count"][:,i,:]):
                    y_melt.append(i)
            min_y_melt = np.amin(y_melt)
            max_y_melt = np.amax(y_melt)
            
            if min_y_melt < self.half_width:
                y_min_pad = self.half_width - min_y_melt
            else:
                y_min_pad = 0
                
            if (ydim - max_y_melt - 1) < self.half_width:
                y_max_pad = self.half_width - (ydim - max_y_melt - 1)
            else:
                y_max_pad = 0
                
            # if any melt pools extend to the edge of the domain warn the user
            if not np.all([min_x_melt,(xdim - max_x_melt - 1),min_y_melt,(ydim - max_y_melt - 1)]):
                print("Warning: melt pool touching edge. Suggest using larger domain for these parameters.")

            # Normalize the data
            point_data["Max Reheat"] = point_data["Max Reheat"]/T_melt
            point_data["Melt Count"] = point_data["Melt Count"]/5.0 # 10 picked at random, seems unlikely that any spot will melt more than 10 times
            point_data["Nucleation Time"] = point_data["Nucleation Time"]/500.0
            point_data["Solidus Time"] = point_data["Solidus Time"]/4000.0
            point_data["Annealing Time"] = point_data["Annealing Time"]/10000.0
            point_data["X Grad"] = point_data["X Grad"]/150.0
            point_data["Y Grad"] = point_data["Y Grad"]/150.0
            point_data["Z Grad"] = point_data["Z Grad"]/150.0
            point_data["Cooling Rate"] = point_data["Cooling Rate"]/10000000
            

            # Convert point_data to 4 dimensional array          
            num_thermal_params = len(point_data)
            thermal_data = np.zeros((xdim, ydim, zdim, num_thermal_params), dtype = np.float32)
            
            param_names = []
            i = 0
            for parameter in point_data:
                thermal_data[:,:,:,i] = np.float32(point_data[parameter])
                param_names.append(parameter)
                i+=1
            
            # Add the appropriate padding to the thermal data and save to data_dict 
            self.data_dict[paramset_name]["arrayT"] = np.pad(thermal_data, ((x_min_pad, x_max_pad), (y_min_pad, y_max_pad), (z_min_pad, self.half_width), (0,0)))
            self.data_dict["DataNames"] = param_names
            
            # Process the microstructure data
            num_runs = np.sum(microstructure_data[0,0,0,:])
            fractional_microdata = microstructure_data/num_runs
            fractional_microdata[np.isnan(fractional_microdata)]=0
            fractional_microdata = np.float32(fractional_microdata)

            # Save the microstructure data and offset-to-origin pads to data_dict
            self.data_dict[paramset_name]["arrayM"] = fractional_microdata
            self.data_dict[paramset_name]["pad"] = np.array([x_min_pad, y_min_pad, z_min_pad])
            
            # Split the data
            melted_ind = np.array(np.where(point_data["Melt Count"] > 0)).T
            num_melted = len(melted_ind)
            print("num melted:"+str(num_melted))
            self.n_samples += num_melted
            split_ind = int(np.floor(num_melted*training_data_split))
            # print("split_ind:"+str(split_ind))
                        
            rng.shuffle(melted_ind) 
            
            self.n_training += split_ind
            print("num training:"+str(self.n_training))
            self.n_test += (num_melted - split_ind)
            print("num test:"+str(self.n_test), "\n")
            self.data_dict[paramset_name]["indices"] = np.int32(melted_ind)
            
            for i in range(split_ind):
                self.train_indices.append([name_index, i, True])
                
            for i in range(split_ind, num_melted):
                self.test_indices.append([name_index, i, False])
            name_index += np.int32(1)
            
                   
    def __getitem__(self,indexes, transform=False):
        paramset_ind, index, transform = indexes
        paramset = self.data_dict["names"][paramset_ind]
        indices = self.data_dict[paramset]["indices"][index]
        thermal_range_upper = indices + self.half_width + self.data_dict[paramset]["pad"] + 1
        thermal_range_lower = indices - self.half_width + self.data_dict[paramset]["pad"]
        thermal_sample = np.copy(self.data_dict[paramset]["arrayT"][thermal_range_lower[0]:thermal_range_upper[0], 
                                           thermal_range_lower[1]:thermal_range_upper[1], 
                                           thermal_range_lower[2]:thermal_range_upper[2], 
                                           :])
        i,j,k = indices
        sample_p=0
        
        if transform:
            rotate_p = torch.randint(4,(3,))
            flip_p = torch.randint(2,(3,))
            if self.always_sample is None:
                sample_p = torch.randint(2,(1,))[0].item()
            else:
                sample_p = 1
            
            thermal_sample = np.rot90(thermal_sample, k = rotate_p[0], axes = (1,0))
            if rotate_p[0] == 1:
                new_x = thermal_sample[:,:,:,4]
                new_y = -thermal_sample[:,:,:,3]
                thermal_sample[:,:,:,3] = new_x
                thermal_sample[:,:,:,4] = new_y
                
            elif rotate_p[0] == 2:
                new_x = -thermal_sample[:,:,:,3]
                new_y = -thermal_sample[:,:,:,4]
                thermal_sample[:,:,:,3] = new_x
                thermal_sample[:,:,:,4] = new_y
                
            elif rotate_p[0] == 3:
                new_x = -thermal_sample[:,:,:,4]
                new_y = thermal_sample[:,:,:,3]
                thermal_sample[:,:,:,3] = new_x
                thermal_sample[:,:,:,4] = new_y
    
            
            thermal_sample = np.rot90(thermal_sample, k = rotate_p[1], axes = (2,0))
            if rotate_p[1] == 1:
                new_x = thermal_sample[:,:,:,5]
                new_z = -thermal_sample[:,:,:,3]
                thermal_sample[:,:,:,3] = new_x
                thermal_sample[:,:,:,5] = new_z
                
            elif rotate_p[1] == 2:
                new_x = -thermal_sample[:,:,:,3]
                new_z = -thermal_sample[:,:,:,5]
                thermal_sample[:,:,:,3] = new_x
                thermal_sample[:,:,:,5] = new_z
                
            elif rotate_p[1] == 3:
                new_x = -thermal_sample[:,:,:,5]
                new_z = thermal_sample[:,:,:,3]
                thermal_sample[:,:,:,3] = new_x
                thermal_sample[:,:,:,5] = new_z
            
            thermal_sample = np.rot90(thermal_sample, k = rotate_p[2], axes = (2,1))
            if rotate_p[2] == 1:
                new_y = thermal_sample[:,:,:,5]
                new_z = -thermal_sample[:,:,:,4]
                thermal_sample[:,:,:,4] = new_y
                thermal_sample[:,:,:,5] = new_z
                
            elif rotate_p[2] == 2:
                new_y = -thermal_sample[:,:,:,4]
                new_z = -thermal_sample[:,:,:,5]
                thermal_sample[:,:,:,4] = new_y
                thermal_sample[:,:,:,5] = new_z
                
            elif rotate_p[2] == 3:
                new_y = -thermal_sample[:,:,:,5]
                new_z = thermal_sample[:,:,:,4]
                thermal_sample[:,:,:,4] = new_y
                thermal_sample[:,:,:,5] = new_z
                
            if flip_p[0] == 1:
                thermal_sample = np.flip(thermal_sample, axis=0)
                thermal_sample[:,:,:,3] = -thermal_sample[:,:,:,3]
            if flip_p[1] == 1:
                thermal_sample = np.flip(thermal_sample, axis=1)
                thermal_sample[:,:,:,4] = -thermal_sample[:,:,:,4]
            if flip_p[2] == 1:
                thermal_sample = np.flip(thermal_sample, axis=2)
                thermal_sample[:,:,:,5] = -thermal_sample[:,:,:,5]
                    
        
        
        if sample_p == 1:
            probs = self.data_dict[paramset]["arrayM"][i,j,k,:]
            sample = np.random.choice(50,self.num_resamples,p=probs)
            microstructure_sample = np.bincount(sample,minlength=50)
            microstructure_sample = torch.as_tensor(microstructure_sample/np.sum(microstructure_sample),dtype=torch.float32) + torch.rand(50)*1E-40
            self.sample+=1
        else:
            microstructure_sample = torch.as_tensor(np.copy(self.data_dict[paramset]["arrayM"][i,j,k,:]))+torch.rand(50)*1E-40
        
        thermal_sample = torch.as_tensor(thermal_sample.copy()).permute(3,0,1,2)
                
        return thermal_sample, microstructure_sample
        
    def __len__(self):
        return self.n_samples        

class InferenceDataset(Dataset):
    
    def __init__(self, files, input_width = 15):
        # Calculate how many voxels of space we need arround the central data point
        T_melt = 1723.0
        
        self.half_width = int(np.floor((input_width-1)/2))
        
        self.n_samples = 0
        self.n_training = 0
        self.n_test = 0
        self.indices = []
                
        self.data_dict = {}
        self.data_dict["names"] = []
        name_index = np.int32(0)
                
        for thermal_file in files:
            print("Loading ", thermal_file, "\n")
            
            # add dataset structure to data_dict
            paramset_name = thermal_file.split(".")[0]
            self.data_dict["names"].append(paramset_name)
            self.data_dict[paramset_name] = {val: [] for val in ["arrayT","arrayM","indices","pad"]}          
                        
            with meshio.xdmf.TimeSeriesReader(thermal_file) as reader:
                points, _ = reader.read_points_cells()
                _, point_data, _ = reader.read_data(1)
        
            
            # Reshape thermal data
            xdim,ydim,zdim = [len(np.unique(points[:,0])),len(np.unique(points[:,1])),len(np.unique(points[:,2]))]
            
            points = None
            
            for name in point_data: point_data[name] = point_data[name].reshape((xdim,ydim,zdim))
            
            point_data['Max Reheat'][point_data['Melt Count']<1] = 0.0
            
            for lowest_melt in range(zdim):
                if np.any(point_data["Melt Count"][:,:,lowest_melt]): break
            
            # Check how much padding is needed on the bottom. Top always gets max padding
            if lowest_melt < self.half_width: 
                z_min_pad = self.half_width - lowest_melt
            else:
                z_min_pad = 0 # if no padding is needed set zero
            
            # Check how much padding is needed on each side in x dimension
            x_melt = []
            for i in range(xdim):
                if np.any(point_data["Melt Count"][i,:,:]):
                    x_melt.append(i)
            min_x_melt = np.amin(x_melt)
            max_x_melt = np.amax(x_melt)
            
            if min_x_melt < self.half_width:
                x_min_pad = self.half_width - min_x_melt
            else:
                x_min_pad = 0
                
            if (xdim - max_x_melt - 1) < self.half_width:
                x_max_pad = self.half_width - (xdim - max_x_melt - 1)
            else:
                x_max_pad = 0
            
            # Check how much padding is needed on each side in y dimension
            y_melt = []
            for i in range(ydim):
                if np.any(point_data["Melt Count"][:,i,:]):
                    y_melt.append(i)
            min_y_melt = np.amin(y_melt)
            max_y_melt = np.amax(y_melt)
            
            if min_y_melt < self.half_width:
                y_min_pad = self.half_width - min_y_melt
            else:
                y_min_pad = 0
                
            if (ydim - max_y_melt - 1) < self.half_width:
                y_max_pad = self.half_width - (ydim - max_y_melt - 1)
            else:
                y_max_pad = 0
                
            # if any melt pools extend to the edge of the domain warn the user
            if not np.all([min_x_melt,(xdim - max_x_melt - 1),min_y_melt,(ydim - max_y_melt - 1)]):
                print("Warning: melt pool touching edge. Suggest using larger domain for these parameters.")

            # Normalize the data
            point_data["Max Reheat"] = point_data["Max Reheat"]/T_melt
            point_data["Melt Count"] = point_data["Melt Count"]/5.0 # 10 picked at random, seems unlikely that any spot will melt more than 10 times
            point_data["Nucleation Time"] = point_data["Nucleation Time"]/500.0
            point_data["Solidus Time"] = point_data["Solidus Time"]/4000.0
            point_data["Annealing Time"] = point_data["Annealing Time"]/10000.0
            point_data["X Grad"] = point_data["X Grad"]/150.0
            point_data["Y Grad"] = point_data["Y Grad"]/150.0
            point_data["Z Grad"] = point_data["Z Grad"]/150.0
            point_data["Cooling Rate"] = point_data["Cooling Rate"]/10000000
            

            # Convert point_data to 4 dimensional array          
            num_thermal_params = len(point_data)
            thermal_data = np.zeros((xdim, ydim, zdim, num_thermal_params), dtype = np.float32)
            
            param_names = []
            i = 0
            for parameter in point_data:
                thermal_data[:,:,:,i] = np.float32(point_data[parameter])
                param_names.append(parameter)
                i+=1
            
            # Add the appropriate padding to the thermal data and save to data_dict 
            '''note: need to convert to tensor still'''
            self.data_dict[paramset_name]["arrayT"] = np.pad(thermal_data, ((x_min_pad, x_max_pad), (y_min_pad, y_max_pad), (z_min_pad, self.half_width), (0,0)))
            self.data_dict["DataNames"] = param_names
            
            
            # Save the microstructure data and offset-to-origin pads to data_dict
            self.data_dict[paramset_name]["pad"] = np.array([x_min_pad, y_min_pad, z_min_pad])
            
            # Split the data
            melted_ind = np.array(np.where(point_data["Melt Count"] > 0)).T
            num_melted = len(melted_ind)
            print("num melted:"+str(num_melted))
            self.n_samples += num_melted

            self.data_dict[paramset_name]["indices"] = np.int32(melted_ind)
            
            for i in range(num_melted):
                self.indices.append([name_index, i])

            name_index += np.int32(1)
            
            

                   
    def __getitem__(self,indexes, transform=False):
        paramset_ind, index = indexes
        paramset = self.data_dict["names"][paramset_ind]
        indices = self.data_dict[paramset]["indices"][index]
        thermal_range_upper = indices + self.half_width + self.data_dict[paramset]["pad"] + 1
        thermal_range_lower = indices - self.half_width + self.data_dict[paramset]["pad"]
        thermal_sample = self.data_dict[paramset]["arrayT"][thermal_range_lower[0]:thermal_range_upper[0], 
                                           thermal_range_lower[1]:thermal_range_upper[1], 
                                           thermal_range_lower[2]:thermal_range_upper[2], 
                                           :]
        
        thermal_sample = torch.as_tensor(thermal_sample.copy()).permute(3,0,1,2)
                
        return thermal_sample, indices
        
    def __len__(self):
        return self.n_samples        

        
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, dropout = 0.3, kernelSize = 7, kernelPad = 3, linearScale = 4):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernelSize, padding=kernelPad, groups=dim) # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv3d(dim, linearScale * dim, kernel_size=1, padding=0, groups=1) # pointwise/1x1 convs, replacing linear layer
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(linearScale * dim, dim, kernel_size=1, padding=0, groups=1) # pointwise/1x1 convs, replacing linear layer
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1, 1)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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

class ConvNeXt(nn.Module):
    r""" ConvNeXt A PyTorch impl of : `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 9
        num_classes (int): Number of classes for classification head. Default: 50
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=9, num_classes=50, 
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
            nn.Conv3d(in_chans, dims[0], kernel_size=3, stride=3),
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
        self.mean_conv = nn.Conv3d(in_channels=dims[-1],
                      out_channels=dims[-1],
                      kernel_size=5, padding=0, groups = dims[-1], bias = False)
        # Set kernel to calculate mean
        mean_kernel_weights = torch.ones((dims[-1],1,5,5,5))/125
        # print(self.mean_conv.weight.data.shape)
        self.mean_conv.weight.data = torch.FloatTensor(mean_kernel_weights)
        self.mean_conv.requires_grad_=False
        
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
        for i in range(self.depth):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        # print(x.shape)
        x = self.mean_conv(x)
        # print(x.shape)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)
        # print(x.shape)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        # print(x.shape)
        x = self.pdf(x)
        # print(x.shape)
        # print(torch.sum(x,dim = 1).shape)
        x = x/torch.sum(x,dim = 1).unsqueeze(1)
        x = torch.squeeze(x)
        # print(x.shape)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
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


def test(net, testloader):
    sum_loss = 0.0  
    count = 0
    net.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(testloader)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # return x,y
            count += y.shape[0]
            runtime=0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.amp.autocast(device_type='cuda', enabled = False):
                start.record()
                outputs = net(x)
                end.record()
                diff = torch.abs(outputs - y)
                diff_square = torch.square(diff)
                loss = torch.sum(diff_square, dim=1)/50
                sum_loss += torch.sum(loss).item()
                torch.cuda.synchronize()
                runtime = runtime + (start.elapsed_time(end))

    mean_loss = sum_loss/count 
    return mean_loss, runtime/1000


def inference(net, infloader):
    microstructure = np.zeros([350,250,22,50])#[234,376,34,50])#
    with torch.no_grad():
        for i, (x, indices) in enumerate(tqdm(infloader)):
            x = x.to(device)
            outputs = net(x)
            outputs[outputs<1e-5] = 0.0
            
            for ind in range(len(indices)):
                i,j,k = indices[ind]
                microstructure[i,j,k,:] = outputs[ind].cpu().numpy()
    return microstructure

class InfSampler(Sampler[int]):
    r"""Sampler for inference, will iterate over all indices in list.
    
    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        for i in np.arange(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

def train_net(net, optimizer, scheduler, train_loader,trial):  
    scaler = torch.cuda.amp.GradScaler(growth_interval=10,backoff_factor=.9375,growth_factor=1.125, enabled=False)
    min_scale=512
    scaler_reset = 0
    iters = len(train_loader)
    loss_Func = nn.MSELoss()
    epochs =  1
    j=0
    for epoch in range(epochs):
        epochstart = time.time()
        running_loss = 0.0
        epoch_loss = 0.0
        
        for i, (x, y) in enumerate(tqdm(train_loader)): 
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', enabled=False):#, dtype=torch.float32),enabled=False
                outputs = net(x)
                loss = loss_Func(outputs, y)
            scaler.scale(loss).backward()
            #loss.backward()
            scaler.step(optimizer)
            #optimizer.step()
            scaler.update()
            scheduler.step(epoch + i / iters)
            #if scaler._scale<min_scale:
             #   scaler_reset+=1
            #    scaler._scale = torch.tensor(min_scale).to(scaler._scale)
            running_loss += loss.item()
                    
            if (i+1) % 4000 == 0:
                j = np.max([i+1, j])
                epoch_loss += running_loss
                print("batch:", i+1, "batch_loss:", running_loss/4000)#, "scaler_resets:", scaler_reset)
                running_loss = 0.0

        j = i+1
        epoch_loss += running_loss
        epoch_time = datetime.timedelta(seconds=time.time()-epochstart)

        print("batch:", i+1, "batch_loss:", running_loss/((i+1)%4000))
        
        print('Epoch took: %s' %epoch_time)

rate = 0.001

def objective(trial, dataset):#, train_loader, test_loader
    # torch.backends.cudnn.benchmark = True
    print("objective started")
    dataset.always_sample = False
    dataset.num_resamples = trial.suggest_int("Number of Resamples", 1, 2000)
    train_sampler = SubsetRandomSampler(dataset.train_indices)
    test_sampler = SubsetRandomSampler(dataset.test_indices)
    print("samplers setup")
    
    train_loader = DataLoader(dataset, batch_size = 128, sampler=train_sampler, num_workers=4, pin_memory=True)
    print("train loader setup")
    # best_so_far = {
    # "Layer1_depth": 1,
    # "Layer2_depth": 1,
    # "Layer3_depth": 3,
    # "Layer4_depth": 0,

    # "Layer1_kernel": 3,
    # "Layer2_kernel": 3,
    # "Layer3_kernel": 4,
    # "Layer4_kernel": 3,

    # "Channels1": 64,
    # "Channels2": 96,
    # "Channels3": 304,
    # "Channels4": 376,

    # "Layer1_scale": 7,
    # "Layer2_scale": 8,
    # "Layer3_scale": 1,
    # "Layer4_scale": 3,
    # }
    config = {
    "Layer1_depth": trial.suggest_int("Layer1_depth", 1, 6),
    "Layer2_depth": trial.suggest_int("Layer2_depth", 0, 6),
    "Layer3_depth": trial.suggest_int("Layer3_depth", 0, 11),
    "Layer4_depth": trial.suggest_int("Layer4_depth", 0, 6),

    "Layer1_kernel": trial.suggest_int("Layer1_kernel", 1, 4),
    "Layer2_kernel": trial.suggest_int("Layer2_kernel", 1, 4),
    "Layer3_kernel": trial.suggest_int("Layer3_kernel", 1, 4),
    "Layer4_kernel": trial.suggest_int("Layer4_kernel", 1, 4),

    "Channels1": trial.suggest_int("Channels1", 16, 400, step=8),
    "Channels2": trial.suggest_int("Channels2", 32, 352, step=8),
    "Channels3": trial.suggest_int("Channels3", 64, 560, step=8),
    "Channels4": trial.suggest_int("Channels4", 192, 896, step=8),

    "Layer1_scale": trial.suggest_int("Layer1_scale", 1, 11),
    "Layer2_scale": trial.suggest_int("Layer2_scale", 1, 11),
    "Layer3_scale": trial.suggest_int("Layer3_scale", 1, 11),
    "Layer4_scale": trial.suggest_int("Layer4_scale", 1, 11),
    }
    #config = best_so_far
    net = ConvNeXt(optim_config=config).to(device)
    print("net setup")
    
    optimizer = optim.AdamW(net.parameters(), lr = 0.001)
    print("optimizer setup")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4, eta_min =0.5E-5, T_mult = 2)
    print("scheduler setup")
    
    train_net(net, optimizer, scheduler, train_loader,trial)  # Train the model
    print("training finished")
    train_loader = None
    print("trainloader emptied")
    
    test_loader = DataLoader(dataset, batch_size = 512, sampler=test_sampler, num_workers=16, pin_memory=True)
    print("test loader setup")
    print("initial test run")
    loss = test(net, test_loader)  # let cudnn see test run before timing actual test run
    print("test starting")
    num_runs = 4
    best_time = 10000.0
    for run in range(num_runs):
        #starttime = time.time()
        loss, runtime = test(net, test_loader)  # Compute test accuracy
        #runtime=time.time()-starttime
        best_time = min(runtime,best_time)
    print("test finished")
    performance = len(dataset.test_indices)/best_time
    print("loss [MSE]= ", loss)
    print('performance [points/s] = ', performance)
    

    return performance,loss

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
        
if __name__ == '__main__':    

    dataset = SurrogateDataset(files)
    print("finished loading")
    func = lambda trial: objective(trial, dataset)
    print("lambda function created")
    sampler = optuna.integration.BoTorchSampler(n_startup_trials=4)
    study = optuna.create_study(directions=["maximize", "minimize"], study_name = "multilayer_tk3", storage = "sqlite:///optimization3.db", load_if_exists=True)
    study.optimize(func, n_trials=10)
    #plot = optuna.visualization.plot_pareto_front(study, targets=lambda t: (t.values[1],t.values[0]/1000), target_names=["Loss [MSE]","Performance [points/s]"])
    #plot = optuna.visualization.plot_pareto_front(study, targets=lambda t: (1/(t.values[0]/1000),t.values[1]), target_names=["Performance [s/point]","MSE Loss"])
    #plot = optuna.visualization.plot_pareto_front(study, targets=lambda t: ((t.values[0]/1000),t.values[1]), target_names=["Performance [points/s]","Loss [MSE]"])
    plot = optuna.visualization.plot_pareto_front(study, targets=lambda t: ((t.values[0]/1000),t.values[1]), target_names=["Performance [points/second]","MSE Loss"])
    plot.update_yaxes(mirror=True,linewidth=2, gridwidth=2, range=[.0001,0.00015])
    plot.update_xaxes(mirror=True,linewidth=2, gridwidth=2)
    plot.update_layout(title=dict(text='Pareto Front Plot of Speed vs Accuracy', font=dict(size=40), automargin=True, yref='container', xanchor='center', x=0.5, yanchor = 'middle',y=0.95),plot_bgcolor='#eeeeee')
    plot.update_layout(xaxis=dict(titlefont=dict(size=35),tickfont=dict(size=25)),yaxis=dict(titlefont=dict(size=35),tickfont=dict(size=25)))
    plot.update_traces(marker=dict(size=12, line=dict(width = 2, color='DarkSlateGrey')))
    plot.update_traces(marker=dict(showscale=False))
    plot.update_layout(width=1000, height=800)
    plot.write_image("test_pareto.png")
    #plot.update_yaxes(gridcolor='LightGrey',linecolor='LightGrey',mirror=True,linewidth=2, gridwidth=2, range=[.0001,0.00015])
    #plot.update_xaxes(gridcolor='LightGrey',linecolor='LightGrey',mirror=True,linewidth=2, gridwidth=2)    
    #plot.update_layout(title=dict(text='Pareto Front Plot of Speed vs Accuracy', font=dict(size=40), automargin=True, yref='container', xanchor='center', x=0.5, yanchor = 'middle',y=0.95),plot_bgcolor='rgba(0,0,0,0)')
    
    #plot.update_traces(marker=dict(size=12, line=dict(width = 2, color='DarkSlateGrey'))),selector=dict(mode='markers'))
    #optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="Performance")
    #optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1], target_name="Loss")
    #file = open("study1.pkl", 'wb')
    #pickle.dump(study, file)
    # data.to_pickle('long_full.pkl')
    # data.plot.scatter('loss','performance',c='index',colormap='viridis')
    # print("Best config is:", results.get_best_result().config)
    
plot = optuna.visualization.plot_pareto_front(study, targets=lambda t: ((t.values[0]/1000),t.values[1]), target_names=["Performance [points/second]","MSE Loss"])
plot.update_yaxes(mirror=True,linewidth=2, gridwidth=2, range=[.0001,0.00015])
plot.update_xaxes(mirror=True,linewidth=2, gridwidth=2)
plot.update_layout(title=dict(text='Pareto Front Plot of Speed vs Accuracy', font=dict(size=40), automargin=True, yref='container', xanchor='center', x=0.5, yanchor = 'middle',y=0.95),plot_bgcolor='#eeeeee')
plot.update_layout(xaxis=dict(titlefont=dict(size=35),tickfont=dict(size=25)),yaxis=dict(titlefont=dict(size=35),tickfont=dict(size=25)))
plot.update_traces(marker=dict(size=12, line=dict(width = 2, color='DarkSlateGrey')))
plot.update_traces(marker=dict(showscale=False))
plot.update_layout(width=1000, height=800)
plot.write_image("test_pareto.png")

