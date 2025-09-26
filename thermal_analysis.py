# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 2024

@author: mason

load in the same thermal files used for training surrogate model and perform statistics on the thermal characteristics/plot histograms of the thermal characteristics
    
"""
#import pdb
import os

import numpy as np
import cupy as cp
from cupyx.scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
#import pandas as pd
import pickle


import time
import datetime
import meshio
from tqdm import tqdm
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from tensordict.tensordict import TensorDict
import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler

from typing import Iterator, Sequence
#from numba import njit
distances_file = None
average_therm_file = None

average_therm_file = 'Average_thermal_characteristics.npy'
#distances_file = 'Correlation_from_mean.csv'
distances_file = 'MSE_from_mean.csv'

# Provide name of csv file containing the metadata for microstructure and thermal training data:
# Include .csv extension. Default: "data_points_metadata.csv"
metadata_filename = "data_points_metadata.csv"
        
# Options for telling the model which data to use
# Set testing to True to manually set specific data files to use inside get_data_filenames function
# Leave set to False to use data_points_metadata.csv for auto selection of data
Testing = False
# Tell the model if it should only use the "training" portion of the available dataset, as defined in data_points_metadata.csv, or all of it
# Set training to true when only using part of your available data, else use "All"
Training = "All"  # Options: True, "All" 

# Lets you use a scratch directory for the microstructure data if dataset is not in the current working directory.
# Currently assumes scratch base path is /scratch/ followed by <scratch_dir> given here
# Implemented for use with memmaps: should have reduced access latency compared to network attached storage.
use_scratch = True
scratch_dir = 'mydir' # 'masjone' #'mydir'

# Disable the pre-check to see if the raw data files exist:
# Only set to true if the data dict has already been populated (and the raw files are gone)
# In the future it would be better to check if the data is already in the TensorDict before checking for the raw files.
DISABLE_RAW_FILE_CHECK = True 

rng=np.random.default_rng()


def get_data_filenames(Testing, Training, scratch_path, metadata_filename):
    r"""
    Automatically generates a list of file names for PreloadData to use by reading metadata from metadata csv
    
    Some of the checks in here are a bit convoluted and might miss problems.
    """

    # Load data metadata
    data_metadata = pd.read_csv(metadata_filename, keep_default_na = False, dtype={'ModelLayersThermal': str, 'LargestEnsemble': str})

    # Separate out the data we want to train on
    # Uglier than necessary because using boolean and string
    if Training == True:
        training_metadata = data_metadata.loc[data_metadata.loc[:,'TrainingData']==Training,:]
    elif Training.lower() == "all":
        training_metadata = data_metadata
    else:
        raise Exception("Invalid value for 'Training': use 'All' or True")

    # If not testing the algorithm, set up the list of datafiles corresponding to the chosen settings and metadata_filename: "data_points_metadata.csv"
    if Testing == False:
        training_files_thermal = []
        
        for index, row in training_metadata.iterrows():
            
             
            # If we pass the first check, deal with the single layer thermal layers = None condition
            if row['ModelLayersThermal'] == 'None':
                print('Single layer: '+row['Thermal_name'])
                # Add filenames to list
                training_files_thermal.append('thermal_data/'+row['Thermal_name']+'.xmf')
            
            # Know that thermal layers not equal to None
            # Check if thermal layers match MS layers, if not, skip
            else:
                thermal_layers = [int(ele) for ele in row['ModelLayersThermal'].split(',')]
                
                # Multi-layer case
                if len(thermal_layers) > 1:
                    for layer in thermal_layers:
                        training_files_thermal.append('thermal_data/'+row['Thermal_name']+'_L'+str(int(layer))+'.xmf')
                
                # Single layer case
                elif len(thermal_layers) == 1:                    
                    layer = thermal_layers[0]
                
                    # Add filenames to list
                    training_files_thermal.append('thermal_data/'+row['Thermal_name']+'_L'+str(int(layer))+'.xmf')


                    
                else:
                    print('Warning: Skipping '+row['Thermal_name']+"Something went wrong, number of layers less than one")


        i = 0      
        if not DISABLE_RAW_FILE_CHECK:        
            while i < (len(training_files_thermal)):
                if os.path.isfile(training_files_thermal[i]) == False:
                    print("Warning: " + training_files_thermal[i] + " not found, removing from list")
                    del training_files_thermal[i]                    
                    i = i-1
                    if i >= len(training_files_thermal):
                        break

                    
                i+=1
        else:
            print('DISABLE_RAW_FILE_CHECK set to True, not checking for data files. \nMay cause problems if data not already in TensorDict.') 

        thermal_files = training_files_thermal


    if Testing == True:

        thermal_files = [
                        'L1P90V105.xmf',
                        'L1P100V100.xmf'
                        ]

        thermal_files = ["thermal_data/" + thermal_file for thermal_file in thermal_files]

    
    files = thermal_files
    
    return files

def PreloadData(files, path = '', input_width = 25, SAVE_DICT = False):
    '''
    Checks for a pre-saved TensorDict and loads it if it exists, otherwise creates it.
    In both cases checks if data is populated, and attempts to add it if it is not
    
    Path is the location to check for/save the TensorDict
    SAVE_DICT defaults to false to avoid accidental deletion of MS data (not sure if saving would delete existing data or not)
    '''

    # Check if TensorDict already exists
    try:
        data_dict = TensorDict.load_memmap(path)
        TensorDict_exists = True
    except:
        print('No matching TensorDict found at ', path,', creating from files.')
        data_dict = {}
        data_dict["names"] = []
        TensorDict_exists = False

    
    changes_made = False
    
    training_names = []
    training_sizes = []
    
    for thermal_file in files:
    
        paramset_name = thermal_file.split(".npy")[0].split("/")[1]
        training_names.append(paramset_name)
        
        ### START WITH THERMAL DATA ###
        # Check if paramset has already been populated into tensordict
        if paramset_name in data_dict["names"]:
            print(paramset_name+" pre-populated, skipping thermal file")
            #breakpoint()
            xdim,ydim,zdim, channels = data_dict[paramset_name]["arrayT"].shape
            new_paramset = False
            
        # If it hasn't then add it
        else:
            new_paramset = True
            changes_made = True
            print("Loading ", thermal_file)
            T_melt = 1723.0
            half_width = int(np.floor((input_width-1)/2))
            # New paramset, so set up dict for it
            data_dict["names"].append(paramset_name)
            data_dict[paramset_name] = {val: [] for val in ["arrayT","arrayM","indices","pad"]}
            data_dict[paramset_name]["arrayM"] = {}
           
            # Read thermal file
            with meshio.xdmf.TimeSeriesReader(thermal_file) as reader:
                points, _ = reader.read_points_cells()
                try:_, point_data, _ = reader.read_data(1)
                except:_, point_data, _ = reader.read_data(0)
    
            # Reshape thermal data
            xdim,ydim,zdim = [len(np.unique(points[:,0])),len(np.unique(points[:,1])),len(np.unique(points[:,2]))]
            print("thermal dimensions:", xdim, ydim, zdim)
            points = None
            
            for name in point_data: point_data[name] = point_data[name].reshape((xdim,ydim,zdim))
            
            point_data['Max Reheat'][point_data['Melt Count']<1] = 0.0
            
            for lowest_melt in range(zdim):
                if np.any(point_data["Melt Count"][:,:,lowest_melt]): break
        
            # Check how much padding is needed on the bottom. Top always gets max padding
            if lowest_melt < half_width: 
                z_min_pad = half_width - lowest_melt
            else:
                z_min_pad = 0 # if no padding is needed set zero
            
            # Check how much padding is needed on each side in x dimension
            x_melt = []
            for i in range(xdim):
                if np.any(point_data["Melt Count"][i,:,:]):
                    x_melt.append(i)
            min_x_melt = np.amin(x_melt)
            max_x_melt = np.amax(x_melt)
            
            if min_x_melt < half_width:
                x_min_pad = half_width - min_x_melt
            else:
                x_min_pad = 0
                
            if (xdim - max_x_melt - 1) < half_width:
                x_max_pad = half_width - (xdim - max_x_melt - 1)
            else:
                x_max_pad = 0
            
            # Check how much padding is needed on each side in y dimension
            y_melt = []
            for i in range(ydim):
                if np.any(point_data["Melt Count"][:,i,:]):
                    y_melt.append(i)
            min_y_melt = np.amin(y_melt)
            max_y_melt = np.amax(y_melt)
            
            if min_y_melt < half_width:
                y_min_pad = half_width - min_y_melt
            else:
                y_min_pad = 0
                
            if (ydim - max_y_melt - 1) < half_width:
                y_max_pad = half_width - (ydim - max_y_melt - 1)
            else:
                y_max_pad = 0
                
            # if any melt pools extend to the edge of the domain warn the user
            if not np.all([min_x_melt,(xdim - max_x_melt - 1),min_y_melt,(ydim - max_y_melt - 1)]):
                print("Warning: melt pool touching edge. Suggest using larger domain for these parameters.")

            # Scale the thermal data to fall closer to 0
            # These were best guesses for scaling factors
            # Adding this to the first layer of the model with learnable scaling factors may be better
            # But this way the scaled values are pre-computed
            point_data["Max Reheat"] = point_data["Max Reheat"]/T_melt
            point_data["Melt Count"] = point_data["Melt Count"]/5.0 
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
            data_dict["DataNames"] = param_names
            
            # Add the appropriate padding to the thermal data and save to data_dict 
            data_dict[paramset_name]["arrayT"] = np.pad(thermal_data, ((x_min_pad, x_max_pad), (y_min_pad, y_max_pad), (z_min_pad, half_width), (0,0)))
            data_dict[paramset_name]["pad"] = np.array([x_min_pad, y_min_pad, z_min_pad])
            
            # Create list of melted indices
            melted_ind = np.array(np.where(point_data["Melt Count"] > 0)).T
            rng.shuffle(melted_ind) 
            data_dict[paramset_name]["indices"] = np.int32(melted_ind)

        
    # If there wasn't already a tensordict, convert the dict to one
    if not TensorDict_exists:
        tensor_dataDict = TensorDict.from_dict(data_dict)
        tensor_dataDict.set_non_tensor("names",data_dict["names"])
        tensor_dataDict.set_non_tensor("DataNames",data_dict["DataNames"])
        # Save the tensordisct as a memmap
        tensor_dataDict.memmap(path, num_threads=32)
        
    # If it alredy existed, and changes were made, save changes inplace
    elif changes_made and SAVE_DICT:
        data_dict.memmap_(path, num_threads=32)
        tensor_dataDict = data_dict
        
    # If it already existed and nothing was changed, don't do anything
    else:
        tensor_dataDict = data_dict
        
    # If it already existed and nothing was changed, don't do anything
    
    return (tensor_dataDict, training_names)


class Thermal_analysis():
    
    def __init__(self, files):
        # Calculate how many voxels of space we need arround the central data point
        T_melt = 1723.0

        self.train_indices = []
                
        self.data_dict = {}
        self.data_dict["names"] = []
        self.data_dict["max"] = {}
        self.data_dict["min"] = {}
        self.data_dict["data"] = {}
        
        self.data_dict["Parameter_units"] = {}
        self.data_dict["Parameter_units"]["Max Reheat"] = "Temperature [k]"
        self.data_dict["Parameter_units"]["Melt Count"] = "Count"
        self.data_dict["Parameter_units"]["Nucleation Time"] = "Time [us]"
        self.data_dict["Parameter_units"]["Solidus Time"] = "Time [us]"
        self.data_dict["Parameter_units"]["Annealing Time"] = "Time [us]"
        self.data_dict["Parameter_units"]["X Grad"] = "Temperature Gradient [K/um]"
        self.data_dict["Parameter_units"]["Y Grad"] = "Temperature Gradient [K/um]"
        self.data_dict["Parameter_units"]["Z Grad"] = "Temperature Gradient [K/um]"
        self.data_dict["Parameter_units"]["Cooling Rate"] = "Temperature Rate [-K/us]"
                       
        for thermal_file in files:
            print("Loading ", thermal_file)
            
            # add dataset structure to data_dict
            paramset_name = thermal_file.split(".")[0]
            self.data_dict["names"].append(paramset_name)
            #self.data_dict[paramset_name] = {val: [] for val in ["arrayT","arrayM","indices","pad"]}

                        
            with meshio.xdmf.TimeSeriesReader(thermal_file) as reader:
                points, _ = reader.read_points_cells()
                try:_, point_data, _ = reader.read_data(1)
                except:_, point_data, _ = reader.read_data(0)
        
            # Reshape thermal data
            xdim,ydim,zdim = [len(np.unique(points[:,0])),len(np.unique(points[:,1])),len(np.unique(points[:,2]))]
            print("thermal dimensions:", xdim, ydim, zdim)
            points = None
            
            size = np.array([xdim,ydim,zdim])
            
            for name in point_data: point_data[name] = point_data[name].reshape((xdim,ydim,zdim))
            
            point_data['Max Reheat'][point_data['Melt Count']<1] = 0.0


            # # Normalize the data
            # point_data["Max Reheat"] = point_data["Max Reheat"]/T_melt
            # point_data["Melt Count"] = point_data["Melt Count"]/5.0 # 10 picked at random, seems unlikely that any spot will melt more than 10 times
            # point_data["Nucleation Time"] = point_data["Nucleation Time"]/500.0
            # point_data["Solidus Time"] = point_data["Solidus Time"]/4000.0
            # point_data["Annealing Time"] = point_data["Annealing Time"]/10000.0
            # point_data["X Grad"] = point_data["X Grad"]/150.0
            # point_data["Y Grad"] = point_data["Y Grad"]/150.0
            # point_data["Z Grad"] = point_data["Z Grad"]/150.0
            # point_data["Cooling Rate"] = point_data["Cooling Rate"]/10000000
            
            melted_ind = np.where(point_data["Melt Count"] > 0)
            #print("max melt: ", np.max(np.array(melted_ind)[0,:]), np.max(np.array(melted_ind)[1,:]), np.max(np.array(melted_ind)[2,:]))
            #print("max melt: ", np.max(melted_ind[:,0]), np.max(melted_ind[:,1]), np.max(melted_ind[:,2]))
            #max_ind = np.array([np.max(np.array(melted_ind)[0,:]), np.max(np.array(melted_ind)[1,:]), np.max(np.array(melted_ind)[2,:])])
            
            # for row in melted_ind:
                # if np.any(row>size):
                    # print("out of bounds: ",row)
            
            self.data_dict['channel_names'] = []
            for parameter in point_data:
                #print(parameter)
                #print(point_data[parameter].shape)
                #breakpoint()
                #print(point_data[parameter][tuple(max_ind)])
                
                if parameter not in self.data_dict["max"]:
                    self.data_dict["max"][parameter] = np.max(point_data[parameter][melted_ind])
                else:
                    self.data_dict["max"][parameter] = np.max([np.max(point_data[parameter][melted_ind]),self.data_dict["max"][parameter]])
                if parameter not in self.data_dict["min"]:
                    self.data_dict["min"][parameter] = np.min(point_data[parameter][melted_ind])
                else:
                    self.data_dict["min"][parameter] = np.min([np.min(point_data[parameter][melted_ind]),self.data_dict["min"][parameter]])
                
                self.data_dict['channel_names'].append(parameter)
                
                if parameter not in self.data_dict["data"]:
                    self.data_dict["data"][parameter] = {}
                self.data_dict["data"][parameter][paramset_name] = np.copy(point_data[parameter][melted_ind])
        
            
            
        #if "bins" not in self.data_dict:
        self.data_dict["bins"] = {}
        self.data_dict["binned_data"] = {key: {} for key in self.data_dict['channel_names']}
        
        # for paramset_name in self.data_dict["DataNames"]:
            # self.data_dict["binned_data"][channel][paramset_name] = np.histogram(self.data_dict["data"][channel][paramset_name][melted_ind], bins = self.data_dict["bins"][channel])
        
        plt.style.use('ggplot')
        '''All Input Channels'''
        # fig, axes = plt.subplots(3,3, sharex = False, figsize=(10,10))
        # subplot = 0
        # for channel in self.data_dict['channel_names']:
            # print(channel + ": [" + str(self.data_dict["min"][channel]) + "," + str(self.data_dict["max"][channel]) + "]")
            # param_range = self.data_dict["max"][channel] - self.data_dict["min"][channel]
            # #self.data_dict["bins"][channel] = np.linspace(self.data_dict["min"][channel],self.data_dict["max"][channel],200)
            # #self.data_dict["bins"]['Melt Count'] = np.linspace(self.data_dict["min"]['Melt Count'],self.data_dict["max"]['Melt Count'],9)
            
            # channel_data = np.concatenate([data for data in self.data_dict["data"][channel].values()])
            # #_, self.data_dict["bins"][channel]= histogram(channel_data, bins = 'knuth')
            # #_, self.data_dict["bins"][channel]= knuth_bin_width(channel_data,return_bins = True)
            # #self.data_dict["bins"][channel]= bayesian_blocks(channel_data)
            # # _, self.data_dict["bins"][channel]= scott_bin_width(channel_data,return_bins = True)
            # _, self.data_dict["bins"][channel]= freedman_bin_width(channel_data,return_bins = True)
            
            # # for paramset_name in self.data_dict["names"]:
                # # self.data_dict["binned_data"][channel][paramset_name],self.data_dict["bins"][channel] = hist(self.data_dict["data"][channel][paramset_name], bins = self.data_dict["bins"][channel])
            
            
            # #channel_data = [data for data in self.data_dict["binned_data"][channel].values()]
            # channel_data = [data for data in self.data_dict["data"][channel].values()]
            # #breakpoint()
            # #plt.hist(channel_data, bins = self.data_dict["bins"][channel], histtype='step', stacked=True, label = self.data_dict["names"])
            # #plt.show()
            # #breakpoint()
            
            # axes[int(np.floor(subplot/3)),subplot%3].set_title(channel)
            # axes[int(np.floor(subplot/3)),subplot%3].set_box_aspect(.9)
            # axes[int(np.floor(subplot/3)),subplot%3].hist(channel_data, bins = self.data_dict["bins"][channel], histtype='barstacked', label = self.data_dict["names"])
            # #axes[int(np.floor(subplot/3)),subplot%3].legend()
            
            # subplot += 1
            
        # fig.suptitle('Distributions of all Thermal Channels')
        # fig.tight_layout()
        # plt.show()
        
        # analyze all gradient options (x,y,z, abs/non-abs, magnitude with directions, combined)
        '''Absolute Values of thermal gradients'''
        fig, axes = plt.subplots(3,1, sharex = False, figsize=(10,10))
        subplot = 0
        for channel in ["X Grad", "Y Grad", "Z Grad"]:
            print(channel + ": [" + str(self.data_dict["min"][channel]) + "," + str(self.data_dict["max"][channel]) + "]")
            param_range = self.data_dict["max"][channel] - self.data_dict["min"][channel]
            #self.data_dict["bins"][channel] = np.linspace(self.data_dict["min"][channel],self.data_dict["max"][channel],200)
            #self.data_dict["bins"]['Melt Count'] = np.linspace(self.data_dict["min"]['Melt Count'],self.data_dict["max"]['Melt Count'],9)
            
            channel_data = np.concatenate([np.abs(data) for data in self.data_dict["data"][channel].values()])
            #_, self.data_dict["bins"][channel]= histogram(channel_data, bins = 'knuth')
            #_, self.data_dict["bins"][channel]= knuth_bin_width(channel_data,return_bins = True)
            #self.data_dict["bins"][channel]= bayesian_blocks(channel_data)
            # _, self.data_dict["bins"][channel]= scott_bin_width(channel_data,return_bins = True)
            _, self.data_dict["bins"][channel]= freedman_bin_width(channel_data,return_bins = True)
            
            # for paramset_name in self.data_dict["names"]:
                # self.data_dict["binned_data"][channel][paramset_name],self.data_dict["bins"][channel] = hist(self.data_dict["data"][channel][paramset_name], bins = self.data_dict["bins"][channel])
            
            
            #channel_data = [data for data in self.data_dict["binned_data"][channel].values()]
            channel_data = [np.abs(data) for data in self.data_dict["data"][channel].values()]
            #breakpoint()
            #plt.hist(channel_data, bins = self.data_dict["bins"][channel], histtype='step', stacked=True, label = self.data_dict["names"])
            #plt.show()
            #breakpoint()
            axes[subplot].set_xlim((0,50))
            axes[subplot].set_title(channel)
            axes[subplot].set_box_aspect(.9)
            axes[subplot].hist(channel_data, bins = self.data_dict["bins"][channel], histtype='barstacked', label = self.data_dict["names"])
            #axes[int(np.floor(subplot/3)),subplot%3].legend()
            
            subplot += 1
            
        fig.suptitle('Distribution of Directional Gradient Magnitudes')
        fig.tight_layout()
        plt.show()
        
        
class Thermal_Dataset(Dataset):
    
    def __init__(self, files, input_width = 25, path = ''):
        self.find_average = True

        self.half_width = int(np.floor((input_width-1)/2))
        self.input_width = input_width
        
        self.n_voxels = 0
        self.indices = []
                
        self.data_dict, self.data_names = PreloadData(files, path = path, input_width = input_width)
        
        self.name_index = np.int32(0)
        
        self.average_therm = np.zeros([input_width,input_width,input_width,9])
                
        for data_name in self.data_names:
            
            # Get the number of data points for this file
            num_melted = len(self.data_dict[data_name]["indices"])
            #print("num melted:"+str(num_melted))

            self.n_voxels += num_melted       
            
            for i in range(num_melted):
                self.indices.append([self.name_index, i])
             
            self.name_index += np.int32(1)
            
        print("num voxels:"+str(self.n_voxels)+"\n")
        
        self.indices = np.array(self.indices, dtype=np.int32)
   
        
    def mean_avg_therm(self):
        self.average_therm /= self.n_voxels
                   
    def __getitem__(self, indexes):
        paramset_ind, index = indexes
        paramset = self.data_dict["names"][paramset_ind]
        indices = self.data_dict[paramset]["indices"][index]
        # try:
            # indices = self.data_dict[paramset]["indices"][index]
        # except:
            # print("max: "+str(len(self.data_dict[paramset]["indices"]))+" tried: "+str(index))
            # breakpoint()
        thermal_range_upper = indices + self.half_width + self.data_dict[paramset]["pad"] + 1
        thermal_range_lower = indices - self.half_width + self.data_dict[paramset]["pad"]
        thermal_sample = self.data_dict[paramset]["arrayT"][thermal_range_lower[0]:thermal_range_upper[0], 
                                           thermal_range_lower[1]:thermal_range_upper[1], 
                                           thermal_range_lower[2]:thermal_range_upper[2], 
                                           :].detach().clone()
        i,j,k = indices
        
                
        #return torch.as_tensor(paramset_ind), torch.as_tensor(thermal_sample)
        return paramset_ind, thermal_sample
        
    def __len__(self):
        return self.n_voxels        
        
class Distance_Calc(nn.Module):
    def __init__(self, thermal_mean):
        super().__init__()
        self.mean_array = torch.as_tensor(thermal_mean)
        self.mean_array = nn.Parameter(thermal_mean, requires_grad=False)        
        
    def forward(self, x):
        #breakpoint()
        return torch.sum((x-self.mean_array)**2, dim=[1,2,3])/(25**3)
        
class Correlation_Distance_Calc(nn.Module):
    def __init__(self, thermal_mean):
        super().__init__()
        self.mean_array = torch.as_tensor(thermal_mean)
        self.mean_array = nn.Parameter(thermal_mean, requires_grad=False)
        self.mean_mean = nn.Parameter(torch.sum(self.mean_array, dim=[0,1,2])/(25**3), requires_grad=False)
        #breakpoint()
        self.mean_array_norm = nn.Parameter(self.mean_array-self.mean_mean[None,None,None,:], requires_grad=False)
        self.mean_array_2norm = nn.Parameter(torch.sqrt(torch.sum(self.mean_array_norm**2, dim=[0,1,2])), requires_grad=False)
        
        
    def forward(self, x):
        x_mean = torch.sum(x, dim=[1,2,3])/(25**3)
        x_norm = x-x_mean[:,None,None,None,:]
        x_2norm = torch.sqrt(torch.sum(x_norm**2, dim=[1,2,3]))
        return 1-((x_norm*self.mean_array_norm).sum(dim=[1,2,3])/(x_2norm*self.mean_array_2norm))
        
class BatchTransform():
    def rotate_x(self,sample):
        sample = np.rot90(sample, k = 1, axes = (2,1))
        new_x = sample[:,:,:,:,4]
        new_y = -sample[:,:,:,:,3]
        sample[:,:,:,:,3] = new_x
        sample[:,:,:,:,4] = new_y
        
        return sample
    
    def rotate_y(self,sample):
        sample = np.rot90(sample, k = 1, axes = (3,1))
        new_x = sample[:,:,:,:,5]
        new_z = -sample[:,:,:,:,3]
        sample[:,:,:,:,3] = new_x
        sample[:,:,:,:,5] = new_z
        
        return sample
        
    def flip(self,sample):
        sample = np.flip(sample, axis=1)
        sample[:,:,:,:,3] = -sample[:,:,:,:,3]
        
        return sample
    
    def find_avg(self, sample):
    
        temp_avg = np.zeros_like(sample)
                
        rotate1 = np.copy(sample)
        for first_rotate in range(4):
            if first_rotate > 0: # This if statement just makes it start with the initial orientation, removing it would make the 4th loop the initial orientation instead (it will have rotated 360 deg)
                rotate1 = self.rotate_y(rotate1)
            #breakpoint()
            temp_avg += (rotate1 + self.flip(rotate1))
            
            rotate2 = np.copy(rotate1)
            for sec_rotate in range(3):
                rotate2 = self.rotate_x(rotate2)
                temp_avg += (rotate2 + self.flip(rotate2))
                
                if sec_rotate == 0 or sec_rotate == 2:
                    rotate3 = np.copy(rotate2)
                    rotate3 = self.rotate_y(rotate3)
                    temp_avg += (rotate3 + self.flip(rotate3))
                
                    
        return (temp_avg/48)

class SingleTransform():
    def rotate_x(self,sample):
        sample = np.rot90(sample, k = 1, axes = (1,0))
        new_x = sample[:,:,:,4]
        new_y = -sample[:,:,:,3]
        sample[:,:,:,3] = new_x
        sample[:,:,:,4] = new_y
        
        return sample
    
    def rotate_y(self,sample):
        sample = np.rot90(sample, k = 1, axes = (2,0))
        new_x = sample[:,:,:,5]
        new_z = -sample[:,:,:,3]
        sample[:,:,:,3] = new_x
        sample[:,:,:,5] = new_z
        
        return sample
        
    def flip(self,sample):
        sample = np.flip(sample, axis=0)
        sample[:,:,:,3] = -sample[:,:,:,3]
        
        return sample
    
    def find_avg(self, sample):
    
        temp_avg = np.zeros_like(sample)
                
        rotate1 = np.copy(sample)
        for first_rotate in range(4):
            if first_rotate > 0: # This if statement just makes it start with the initial orientation, removing it would make the 4th loop the initial orientation instead (it will have rotated 360 deg)
                rotate1 = self.rotate_y(rotate1)
            #breakpoint()
            temp_avg += (rotate1 + self.flip(rotate1))
            
            rotate2 = np.copy(rotate1)
            for sec_rotate in range(3):
                rotate2 = self.rotate_x(rotate2)
                temp_avg += (rotate2 + self.flip(rotate2))
                
                if sec_rotate == 0 or sec_rotate == 2:
                    rotate3 = np.copy(rotate2)
                    rotate3 = self.rotate_y(rotate3)
                    temp_avg += (rotate3 + self.flip(rotate3))
                
                    
        return (temp_avg/48)
   
        
class FullSampler(Sampler[int]):
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

        
if __name__ == '__main__':   
    
    # Check if set to use the scratch directory of the node, and set the appropriate directory.
    # For convenience, doesn't change behavior. Recommend using scratch drive if available to avoid networking overhead.
    if use_scratch:
        scratch_path = '/scratch/'+scratch_dir+'/'
    else:
        scratch_path = ''
    
    # Automatically generate list of files
    files = get_data_filenames(Testing, Training, scratch_path, metadata_filename)
    
    if average_therm_file is None or distances_file is None:
        transform = SingleTransform()
        dataset = Thermal_Dataset(files, path= '/scratch/'+scratch_dir+'/data_tensordict')
        sampler = FullSampler(dataset.indices)
    else:
        #breakpoint()
        dataset = Thermal_Dataset(files, path= '/scratch/'+scratch_dir+'/data_tensordict')
    
    if average_therm_file is None:
    
        loader = DataLoader(dataset, batch_size = 8192, sampler=sampler, num_workers=64, pin_memory=True)

        with torch.no_grad():
            temp_therm_sum = torch.zeros([25,25,25,9]).to(device)
            for x in enumerate(tqdm(loader)):
                x = x[1][1].to(device)
                temp_therm_sum += torch.sum(x, dim = 0)
        
        dataset.average_therm = temp_therm_sum.cpu().numpy()
        dataset.average_therm = transform.find_avg(sample=dataset.average_therm)
        dataset.mean_avg_therm()
        np.save('Average_thermal_characteristics.npy', dataset.average_therm)
        #plt.figure()
        #plt.imshow(dataset.average_therm[:,:,12,1])
        
    else:
        print("Loading existing mean")
        dataset.average_therm = np.load(average_therm_file)
    
    if distances_file is None:
        Distance_func_corr = Correlation_Distance_Calc(torch.as_tensor(dataset.average_therm)).to(device)
        Distance_func_MSE = torch.jit.script(Distance_Calc(torch.as_tensor(dataset.average_therm)).to(device))
        loader = DataLoader(dataset, batch_size = 8192, sampler=sampler, num_workers=32, pin_memory=True)
        
        dataset.find_average = False
        outputs_corr = []
        outputs_MSE = []
        Distance_func_corr.eval()
        Distance_func_MSE.eval()
        with torch.no_grad():
            for paramset, x in enumerate(tqdm(loader)):
                # breakpoint()
                paramset = x[0]
                x = x[1].to(device)
                paramset = paramset[:,None]
                out_corr = Distance_func_corr(x).cpu().numpy()
                out_MSE = Distance_func_MSE(x).cpu().numpy()
                #out = np.hstack((paramset,out))
                #breakpoint()
                outputs_corr.append(np.hstack((paramset,out_corr)))
                outputs_MSE.append(np.hstack((paramset,out_MSE)))
        
        
        header_txt = 'Param_ind, ' + ','.join(dataset.data_dict["DataNames"])
        outputs_corr = np.concatenate(outputs_corr, axis=0)
        outputs_MSE = np.concatenate(outputs_MSE, axis=0)
        np.savetxt('Correlation_from_mean.csv', outputs_corr, delimiter = ',', header = header_txt)
        np.savetxt('MSE_from_mean.csv', outputs_MSE, delimiter = ',', header = header_txt)
        
        outputs = outputs_MSE
    
    else:
        print("Loading existing distances")
        outputs = np.loadtxt(distances_file, delimiter = ',', skiprows = 1)
        print("Data finished loading")
    
    
    distances = outputs[:,1:]
    paramsets = outputs[:,0].astype(int)
    
    with open("CaseNames.txt","w") as file:
        for paramset in dataset.data_dict["names"]:
            file.write(paramset+"\n")
    
        
    
    #print(outputs.shape)
    breakpoint()
    
    sampled_data = {paramset: None for paramset in dataset.data_dict["names"]}
    prev_paramset_ind = 0
    paramset_no = 0
    for paramset in sampled_data:
        paramset_ind = np.max(np.where(paramsets == paramset_no))
        paramset_dataset = distances[prev_paramset_ind:paramset_ind, :]
        sampled_data[paramset] = rng.choice(paramset_dataset,size = 4000, replace=False)
        prev_paramset_ind = paramset_ind
        paramset_no += 1
    
    load_distances = False
    save_distances = True
    SIMILARITY = True
    SCATTER = False
    Normalize_sample_channels = True
    
  
    Title_size = 20
    Label_size = 16
    
    if Normalize_sample_channels:
        sample_max = np.zeros(9)
        for paramset in sampled_data:
            temp_max = np.max(sampled_data[paramset][:,:],axis=0)
            sample_max = np.maximum(sample_max,temp_max)
        for paramset in sampled_data:
            sampled_data[paramset] = sampled_data[paramset][:,:]/sample_max
    
    max_abs_val0 = 0
    max_abs_val1 = 0
    max_abs_val2 = 0

    if not load_distances:
        # create a nested dict which has the 9 channels X the 9 channels, with arrays for storing the similarity between all paramsets for each channel combination
        # 9 channels
        # -> 9 channels
        #    -> NxN array, where N is the number of simulations/paramsets
        # Each element in each array is the similarity between the corresponding paramsets, of the corresponding channel1xchannel 2 relationship
        # Layer zero of the array [x,x,0] is the similarity as measured with the average, layer 1 [x,x,1] is using the max (ugly, but easiest way to add other measures)
        similarity_dict = {channel1: {channel2: np.zeros([np.max(paramsets)+1,np.max(paramsets)+1, 3]) for channel2 in dataset.data_dict["DataNames"]} for channel1 in dataset.data_dict["DataNames"]}
        similarity_dict["full"] = np.zeros([np.max(paramsets)+1,np.max(paramsets)+1,3])
    
    if not load_distances:
        for i in range(distances.shape[-1]):
            ChannelName1 = dataset.data_dict["DataNames"][i]

            for j in range(distances.shape[-1]):
                if j>i:
                    continue
                    
                    
                ChannelName2 = dataset.data_dict["DataNames"][j]
                
                print("Calulating: " + ChannelName1 + ", " + ChannelName2)
                
                paramset1_no = 0
                for paramset1 in sampled_data:
                #for paramset1 in range(np.max(paramsets)+1):
                    sample1 = cp.asarray(sampled_data[paramset1][:,[i,j]]) # extract the data from the 2 current channels of the current paramset sample, move it to GPU
                    self_distance1_array = squareform(pdist(sample1).get())
                    self_distancemax1 = cp.amax(self_distance1_array, axis = 0)#, where = self_distance1_array>0, initial = 0.0)
                    self_distancemin1 = cp.asarray(np.min(self_distance1_array, axis = 0, where = self_distance1_array>0, initial = 1000))
                    
                    self_distance1_array = 0
                    self_distance1_avg = cp.average(self_distancemin1).get()
                    self_distance1_max = cp.amax(self_distancemin1).get()
                    self_distance1_avgmax = cp.average(self_distancemax1).get()
                    
                    paramset2_no = 0
                    for paramset2 in sampled_data:
                        # Calculate simularity of paramset 1 data to paramset 2 data for the channel1xchannel2 plot
                        if paramset1_no == paramset2_no: # This is always going to be self similar
                            similarity_dict[ChannelName1][ChannelName2][paramset1_no, paramset2_no, 0] = 0
                            similarity_dict[ChannelName1][ChannelName2][paramset1_no, paramset2_no, 1] = 0
                            similarity_dict[ChannelName1][ChannelName2][paramset1_no, paramset2_no, 2] = 0
                            continue
                            
                        if paramset2_no > paramset1_no: # Only calculate if in the lower triangular, to avoid calculating the distances for every pair twice
                            continue
                            
                        sample2 = cp.asarray(sampled_data[paramset2][:,[i,j]]) # extract the data from the 2 current channels of the 2nd current paramset sample, move it to GPU
                        self_distance2_array = squareform(pdist(sample2).get())
                        self_distancemax2 = cp.amax(self_distance2_array, axis = 0)#, where = self_distance2_array>0, initial = 0.0)
                        self_distancemin2 = cp.asarray(np.min(self_distance2_array, axis = 0, where = self_distance2_array>0, initial = 1000))
                        
                        self_distance2_array = 0
                        self_distance2_avg = cp.average(self_distancemin2).get()
                        self_distance2_max = cp.amax(self_distancemin2).get()
                        self_distance2_avgmax = cp.average(self_distancemax2).get()
                        
                        cross_distance = cdist(sample1,sample2)
                        cross_distancemin1 = cp.amin(cross_distance, axis = 0)
                        cross_distancemax1 = cp.amax(cross_distance, axis = 0)
                        cross_distancemin2 = cp.amin(cross_distance, axis = 1)
                        cross_distancemax2 = cp.amax(cross_distance, axis = 1)
                        cross_distance = 0
                        
                        cross_distance1_avg = cp.average(cross_distancemin1).get()
                        cross_distance1_max = cp.amax(cross_distancemin1).get()
                        
                        cross_distance1_avgmax = cp.average(cross_distancemax1).get()
                        
                        cross_distance2_avg = cp.average(cross_distancemin2).get()
                        cross_distance2_max = cp.amax(cross_distancemin2).get()
                        
                        cross_distance2_avgmax = cp.average(cross_distancemax2).get()
                        
                        similarity1_avg = cross_distance1_avg/self_distance1_avg
                        similarity2_avg = cross_distance2_avg/self_distance2_avg
                        
                        similarity1_max = cross_distance1_max/self_distance1_max
                        similarity2_max = cross_distance2_max/self_distance2_max
                        
                        similarity1_avgmax = cross_distance1_avgmax/self_distance1_avgmax
                        similarity2_avgmax = cross_distance2_avgmax/self_distance2_avgmax
                        
                        # Lower triangular is axis 0 WRT axis 1
                        # Upper triangular is axis 1 WRT axis 0
                        similarity_dict[ChannelName1][ChannelName2][paramset1_no, paramset2_no, 0] = similarity1_avg
                        similarity_dict[ChannelName1][ChannelName2][paramset2_no, paramset1_no, 0] = similarity2_avg
                        
                        similarity_dict[ChannelName1][ChannelName2][paramset1_no, paramset2_no, 1] = similarity1_max
                        similarity_dict[ChannelName1][ChannelName2][paramset2_no, paramset1_no, 1] = similarity2_max
                        
                        similarity_dict[ChannelName1][ChannelName2][paramset1_no, paramset2_no, 2] = similarity1_avgmax
                        similarity_dict[ChannelName1][ChannelName2][paramset2_no, paramset1_no, 2] = similarity2_avgmax
                        
                        similarity_dict[ChannelName2][ChannelName1][paramset1_no, paramset2_no, 0] = similarity1_avg
                        similarity_dict[ChannelName2][ChannelName1][paramset2_no, paramset1_no, 0] = similarity2_avg
                        
                        similarity_dict[ChannelName2][ChannelName1][paramset1_no, paramset2_no, 1] = similarity1_max
                        similarity_dict[ChannelName2][ChannelName1][paramset2_no, paramset1_no, 1] = similarity2_max
                        
                        similarity_dict[ChannelName2][ChannelName1][paramset1_no, paramset2_no, 2] = similarity1_avgmax
                        similarity_dict[ChannelName2][ChannelName1][paramset2_no, paramset1_no, 2] = similarity2_avgmax
                        
                        
                        paramset2_no += 1
                        
                    paramset1_no += 1
                       
        #if save_distances:
        #    with open('similarity_dict.pkl', 'wb') as f:
        #        pickle.dump(similarity_dict,f)
    
    elif load_distances:
        with open('similarity_dict.pkl','rb') as f:
            similarity_dict = pickle.load(f)

    full_names = dataset.data_names

    if SCATTER:
        plt.style.use('ggplot')
        sample_max = np.ones(9)
        # sample_max = np.zeros(9)
        # for paramset in sampled_data:
            # temp_max = np.max(sampled_data[paramset][:,:],axis=0)
            # sample_max = np.maximum(sample_max,temp_max)
        
        ## Compute and plot the 1 vs all others distance
        
        fig, axes = plt.subplots(3,3, sharex = False, figsize=(20,20))
        subplot = 0
        
        
        for i in range(distances.shape[-1]):
            ChannelName = dataset.data_dict["DataNames"][i]
            axes[int(np.floor(subplot/3)),subplot%3].set_title(ChannelName)
            axes[int(np.floor(subplot/3)),subplot%3].set_box_aspect(.9)
            
            combined_channels = np.sum(distances, axis = -1) - distances[:,i]
            #print(combined_channels.shape)
            #breakpoint()
            # combined_data = np.vstack((distances[:,i], combined_channels)).T
            # prev_paramset_ind = 0
            # for paramset in range(np.max(paramsets)+1):
            for paramset in sampled_data:
                sample1 = sampled_data[paramset][:,i]
                sample2 = np.linalg.norm(np.delete(sampled_data[paramset][:,:],i,axis = 1),axis=1)
                axes[int(np.floor(subplot/3)),subplot%3].scatter(sample1, sample2,label = str(paramset))
            
            subplot += 1
                
        #fig.suptitle('Scatter of combined distance of remaining channels vs selected channel distance')
        #fig.tight_layout()
        plt.savefig('All_channel_scatter.png')
        
        ###
        
        
        for i in range(distances.shape[-1]):
            fig, axes = plt.subplots(2,4, sharex = False, figsize=(20,8))
            
            PlotChannelName = dataset.data_dict["DataNames"][i]
            fig.suptitle(PlotChannelName)
            
            similarity_dict[PlotChannelName] = {}
            
            for paramset in sampled_data:
                sample1 = sampled_data[paramset][:,i]
                
                subplot=0
                for j in range(distances.shape[-1]):
                    if i==j:
                        continue
                    sample2 = sampled_data[paramset][:,j]
                    SubplotChannelName = dataset.data_dict["DataNames"][j]
                    #axes[int(np.floor(subplot/4)),subplot%4].set_title(SubplotChannelName)
                    axes[int(np.floor(subplot/4)),subplot%4].set_ylabel(SubplotChannelName)

                    axes[int(np.floor(subplot/4)),subplot%4].set_box_aspect(.9)
                    
                    axes[int(np.floor(subplot/4)),subplot%4].scatter(sample1, sample2,label = str(paramset))
                    #combined_channels = np.sum(distances, axis = -1) - distances[:,i]
                    # breakpoint()
                    #axes[int(np.floor(subplot/3)),subplot%3].scatter(sampled_data[:,0], sampled_data[:,1])
                    #axes[int(np.floor(subplot/3)),subplot%3].legend()
                    subplot+=1
                
            #fig.suptitle('Scatter comparing all channels')
            plt.savefig(PlotChannelName+'_scatter.png')
        
       
        
        fig, axes = plt.subplots(3,3, sharex = False, figsize=(20,20))
        subplot = 0
        
        
        
        for i in range(distances.shape[-1]):
            ChannelName = dataset.data_dict["DataNames"][i]
            axes[int(np.floor(subplot/3)),subplot%3].set_title(ChannelName)
            axes[int(np.floor(subplot/3)),subplot%3].set_box_aspect(.9)
            
            combined_channels = np.sum(distances, axis = -1) - distances[:,i]
            #print(combined_channels.shape)
            #breakpoint()
            # combined_data = np.vstack((distances[:,i], combined_channels)).T
            # prev_paramset_ind = 0
            # for paramset in range(np.max(paramsets)+1):
            for paramset in sampled_data:
                sample1 = sampled_data[paramset][:,i]/sample_max[i]
                sample2 = np.linalg.norm(np.delete(sampled_data[paramset][:,:]/sample_max,i,axis = 1),axis=1)
                axes[int(np.floor(subplot/3)),subplot%3].scatter(sample1, sample2,label = str(paramset))
            
            subplot += 1
                
        #fig.suptitle('Scatter of combined distance of remaining channels vs selected channel distance')
        #fig.tight_layout()
        plt.savefig('All_channel_scatter_normalized.png')
        
        ###
        
        
        for i in range(distances.shape[-1]):
            fig, axes = plt.subplots(2,4, sharex = False, figsize=(20,8))
            
            PlotChannelName = dataset.data_dict["DataNames"][i]
            fig.suptitle(PlotChannelName)
            
            similarity_dict[PlotChannelName] = {}
            
            for paramset in sampled_data:
                sample1 = sampled_data[paramset][:,i]/sample_max[i]
                
                subplot=0
                for j in range(distances.shape[-1]):
                    if i==j:
                        continue
                    sample2 = sampled_data[paramset][:,j]/sample_max[j]
                    SubplotChannelName = dataset.data_dict["DataNames"][j]
                    #axes[int(np.floor(subplot/4)),subplot%4].set_title(SubplotChannelName)
                    axes[int(np.floor(subplot/4)),subplot%4].set_ylabel(SubplotChannelName)

                    axes[int(np.floor(subplot/4)),subplot%4].set_box_aspect(.9)
                    
                    axes[int(np.floor(subplot/4)),subplot%4].scatter(sample1, sample2,label = str(paramset))
                    #combined_channels = np.sum(distances, axis = -1) - distances[:,i]
                    # breakpoint()
                    #axes[int(np.floor(subplot/3)),subplot%3].scatter(sampled_data[:,0], sampled_data[:,1])
                    #axes[int(np.floor(subplot/3)),subplot%3].legend()
                    subplot+=1
                
            #fig.suptitle('Scatter comparing all channels')
            plt.savefig(PlotChannelName+'_scatter_normalized.png')
        
        
    if SIMILARITY:    
        
        for i in range(distances.shape[-1]):
            ChannelName1 = dataset.data_dict["DataNames"][i]
            for j in range(distances.shape[-1]):
                ChannelName2 = dataset.data_dict["DataNames"][j]
                temp_max0 = np.max(similarity_dict[ChannelName1][ChannelName2][:,:,0])
                if max_abs_val0<temp_max0:
                    print("New maximum at Channel1: "+ChannelName1+" Channel2: "+ChannelName2+" max: "+str(temp_max0))
                    max_abs_val0 = temp_max0
                #max_abs_val0 = np.maximum(max_abs_val0,temp_max0)
                temp_max1 = np.max(similarity_dict[ChannelName1][ChannelName2][:,:,1])
                max_abs_val1 = np.maximum(max_abs_val1,temp_max1)
                temp_max2 = np.max(similarity_dict[ChannelName1][ChannelName2][:,:,2])
                max_abs_val2 = np.maximum(max_abs_val1,temp_max2)
                
        #colormap_high = np.flip(mpl.colormaps["viridis"](np.linspace(0.,1,256)),axis=0)
        colormap_high = mpl.colormaps["viridis"](np.linspace(0.,1.,256))
        colormap_low = np.flip(mpl.colormaps["magma"](np.linspace(.5,1.,256)),axis=0)
        colormap_new = np.flip(np.vstack((colormap_high,colormap_low)),axis=0)
        
        my_colormap = colors.LinearSegmentedColormap.from_list("my_colormap",colormap_new)
        
        
        
        fig, axes = plt.subplots(9,9, sharex = True, sharey = True, figsize=(20,20))

        
        for i in range(distances.shape[-1]):
            ChannelName1 = dataset.data_dict["DataNames"][i]

            for j in range(distances.shape[-1]):
                
                ChannelName2 = dataset.data_dict["DataNames"][j]
                plot_name1 = ChannelName1.replace(" ","\n")
                plot_name2 = ChannelName2.replace(" ","\n")
                if i == 0:
                    axes[i,j].set_title(plot_name2,size=Label_size)#,rotation=-30, ha="right", rotation_mode="anchor")
                if j == 0:
                    axes[i,j].set_ylabel(plot_name1,size=Label_size)#,rotation=60, ha="right", rotation_mode="anchor")
                    
                
                
                if j!=0:
                    if i!=(distances.shape[-1]-1):
                        axes[i,j].tick_params(top=False, bottom=False, left=False, right=False, labeltop=False, labelbottom=False, labelleft=False, labelright=False)
                    else:
                        axes[i,j].tick_params(top=False, bottom=True, left=False, right=False, labeltop=False, labelbottom=True, labelleft=False, labelright=False)
                        axes[i,j].set_xticks(np.arange(len(full_names),step=20))
                else:
                    if i!=(distances.shape[-1]-1):
                        axes[i,j].tick_params(top=False, bottom=False, left=True, right=False, labeltop=False, labelbottom=False, labelleft=True, labelright=False)
                    else:
                        axes[i,j].tick_params(top=False, bottom=True, left=True, right=False, labeltop=False, labelbottom=True, labelleft=True, labelright=False)
                        axes[i,j].set_xticks(np.arange(len(full_names),step=20))
                #if j>i:
                #    axes[i,j].set_box_aspect(1.0)
                #    axes[i,j].spines[:].set_visible(False) 
                #    continue        
                if i==j:
                    axes[i,j].set_box_aspect(1.0)
                    axes[i,j].spines[:].set_visible(False) 
                    continue
                    # This is probably still worth comparing, but it's not a 2D distance so should be treated differently
                
                #print("plotting with vcenter=1,vmin=0,vmax="+ str(max_abs_val0))
                im=axes[i,j].imshow(similarity_dict[ChannelName1][ChannelName2][:,:,0], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1.0,vmin=0.0,vmax=max_abs_val0))#, vmax = max_abs_val0, vmin = -max_abs_val0+2)
                axes[i,j].set_box_aspect(1.0)
                axes[i,j].spines[:].set_visible(False)
        
        #fig.suptitle('Similarity Scores Calculated With Average Closest Distances')
        fig.tight_layout()
        #fig.subplots_adjust(wspace=0.04,hspace=0.001)
        plt.savefig('similarity_avg_full_MSE_noself.svg')#,bbox_inches='tight')
        
        fig.subplots_adjust(wspace=0.04,hspace=0.001,left=0.1,right=0.99)#,left=0.01,right=0.015)
        ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val0]
        ticks.sort()
        cbar = fig.colorbar(im,ax=axes.ravel().tolist(),aspect=40, ticks = ticks)
        cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size=Title_size)
        cbar.ax.tick_params(labelsize=Label_size)
        #cbar.formatter.set_powerlimits((0,0))
        plt.savefig('similarity_avg_full_MSE_colorbar_noself.png')#,bbox_inches='tight')
        
        fig, axes = plt.subplots(9,9, sharex = True, sharey = True, figsize=(20,20))
        
        for i in range(distances.shape[-1]):
            ChannelName1 = dataset.data_dict["DataNames"][i]

            for j in range(distances.shape[-1]):
                ChannelName2 = dataset.data_dict["DataNames"][j]
                plot_name1 = ChannelName1.replace(" ","\n")
                plot_name2 = ChannelName2.replace(" ","\n")
                if i == 0:
                    axes[i,j].set_title(plot_name2,size=Label_size)#,rotation=-30, ha="right", rotation_mode="anchor")
                if j == 0:
                    axes[i,j].set_ylabel(plot_name1,size=Label_size)#,rotation=-30, ha="right", rotation_mode="anchor")
                    
                if j!=0:
                    if i!=(distances.shape[-1]-1):
                        axes[i,j].tick_params(top=False, bottom=False, left=False, right=False, labeltop=False, labelbottom=False, labelleft=False, labelright=False)
                    else:
                        axes[i,j].tick_params(top=False, bottom=True, left=False, right=False, labeltop=False, labelbottom=True, labelleft=False, labelright=False)
                        axes[i,j].set_xticks(np.arange(len(full_names),step=20))
                else:
                    if i!=(distances.shape[-1]-1):
                        axes[i,j].tick_params(top=False, bottom=False, left=True, right=False, labeltop=False, labelbottom=False, labelleft=True, labelright=False)
                    else:
                        axes[i,j].tick_params(top=False, bottom=True, left=True, right=False, labeltop=False, labelbottom=True, labelleft=True, labelright=False)
                        axes[i,j].set_xticks(np.arange(len(full_names),step=20))
                        
                if i==j:
                    axes[i,j].set_box_aspect(1.0)
                    axes[i,j].spines[:].set_visible(False) 
                    continue
                    # This is probably still worth comparing, but it's not a 2D distance so should be treated differently
     
                im=axes[i,j].imshow(similarity_dict[ChannelName1][ChannelName2][:,:,1], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1.0,vmin=0.0,vmax=max_abs_val1))#, vmax = max_abs_val1, vmin = -max_abs_val1+2)
                axes[i,j].set_box_aspect(1.0)
                axes[i,j].spines[:].set_visible(False)
                
                
        
        #fig.suptitle('Similarity Scores Calculated With Maximum Closest Distances')
        fig.tight_layout()
        plt.savefig('similarity_max_full_MSE_noself.svg')#,bbox_inches='tight')
        
        fig.subplots_adjust(wspace=0.04,hspace=0.001,left=0.1,right=0.99)
        ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val1]
        ticks.sort()
        cbar = fig.colorbar(im,ax=axes.ravel().tolist(),aspect=40, ticks = ticks)
        cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size=Title_size)
        cbar.ax.tick_params(labelsize=Label_size)
        #cbar.formatter.set_powerlimits((0,0))
        plt.savefig('similarity_max_full_MSE_clorbar_noself.png')#,bbox_inches='tight')
        #breakpoint()
        
        print(max_abs_val0)
        
        fig, axes = plt.subplots(9,9, sharex = True, sharey = True, figsize=(20,20))
        
        for i in range(distances.shape[-1]):
            ChannelName1 = dataset.data_dict["DataNames"][i]

            for j in range(distances.shape[-1]):
                ChannelName2 = dataset.data_dict["DataNames"][j]
                plot_name1 = ChannelName1.replace(" ","\n")
                plot_name2 = ChannelName2.replace(" ","\n")
                if i == 0:
                    axes[i,j].set_title(plot_name2,size=Label_size)#,rotation=-30, ha="right", rotation_mode="anchor")
                if j == 0:
                    axes[i,j].set_ylabel(plot_name1,size=Label_size)#,rotation=-30, ha="right", rotation_mode="anchor")
                    
                if j!=0:
                    if i!=(distances.shape[-1]-1):
                        axes[i,j].tick_params(top=False, bottom=False, left=False, right=False, labeltop=False, labelbottom=False, labelleft=False, labelright=False)
                    else:
                        axes[i,j].tick_params(top=False, bottom=True, left=False, right=False, labeltop=False, labelbottom=True, labelleft=False, labelright=False)
                        axes[i,j].set_xticks(np.arange(len(full_names),step=20))
                else:
                    if i!=(distances.shape[-1]-1):
                        axes[i,j].tick_params(top=False, bottom=False, left=True, right=False, labeltop=False, labelbottom=False, labelleft=True, labelright=False)
                    else:
                        axes[i,j].tick_params(top=False, bottom=True, left=True, right=False, labeltop=False, labelbottom=True, labelleft=True, labelright=False)
                        axes[i,j].set_xticks(np.arange(len(full_names),step=20))
                        
                if i==j:
                    axes[i,j].set_box_aspect(1.0)
                    axes[i,j].spines[:].set_visible(False) 
                    continue
                    # This is probably still worth comparing, but it's not a 2D distance so should be treated differently
     
                im=axes[i,j].imshow(similarity_dict[ChannelName1][ChannelName2][:,:,2], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1.0,vmin=0.0,vmax=max_abs_val2))#, vmax = max_abs_val1, vmin = -max_abs_val1+2)
                axes[i,j].set_box_aspect(1.0)
                axes[i,j].spines[:].set_visible(False)
                
        #fig.suptitle('Similarity Scores Calculated With Average Maximum Distances')
        fig.tight_layout()
        plt.savefig('similarity_avgmax_full_MSE_noself.svg')#,bbox_inches='tight')
        
        fig.subplots_adjust(wspace=0.04,hspace=0.001,left=0.1,right=0.99)
        ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val2]
        ticks.sort()
        cbar = fig.colorbar(im,ax=axes.ravel().tolist(),aspect=40, ticks = ticks)
        cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size=Title_size)
        cbar.ax.tick_params(labelsize=Label_size)
        #cbar.formatter.set_powerlimits((0,0))
        plt.savefig('similarity_avgmax_full_MSE_clorbar_noself.png')#,bbox_inches='tight')
        #breakpoint()
        
        
        for i in range(distances.shape[-1]):
            ChannelName1 = dataset.data_dict["DataNames"][i]
            
            fig, axes = plt.subplots(figsize=(12,12))
            axes.set_box_aspect(1.0)
            axes.set_title(ChannelName1,size=Title_size)

            axes.set_ylabel("Case A", size = Title_size)
            axes.set_xlabel("Case B", size = Title_size)
            
            im = axes.imshow(similarity_dict[ChannelName1][ChannelName1][:,:,0], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1,vmin=0,vmax=max_abs_val0))#, vmax = max_abs_val0, vmin = 0)
            
            ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val0]
            ticks.sort()
            
            cbar = axes.figure.colorbar(im, ax=axes, ticks = ticks)
            cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size = Title_size)
            cbar.ax.tick_params(labelsize=Label_size)
            # yticks = cbar.ax.get_yticks()
            # yticks = np.insert(yticks, 1, 1)
            # cbar.ax.set_yticks(yticks)
            
            axes.spines[:].set_visible(False)
            
            axes.set_xticks(np.arange(len(full_names),step=5))#, labels = full_names)
            axes.set_yticks(np.arange(len(full_names),step=5))#, labels = full_names)
            axes.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)    
            
            plt.setp(axes.get_xticklabels())#, rotation=-45, ha="right", rotation_mode="anchor")
            plt.setp(axes.get_yticklabels())#, rotation=30, ha="right", rotation_mode="anchor")
            
            #breakpoint()
            
            axes.grid(which="minor", color="w", linestyle="-", linewidth=10)
            axes.tick_params(which="minor", bottom=False, left=False)      
        
            fig.suptitle('Similarity Scores Calculated With Average Closest Distances')
            fig.tight_layout()
            plt.savefig(ChannelName1.replace(" ","")+'_MSE_similarity_avg.png')
            
        for i in range(distances.shape[-1]):
            ChannelName1 = dataset.data_dict["DataNames"][i]
            
            fig, axes = plt.subplots(figsize=(12,12))
            axes.set_box_aspect(1.0)
            axes.set_title(ChannelName1,size=Title_size)

            axes.set_ylabel("Case A", size = Title_size)
            axes.set_xlabel("Case B", size = Title_size)
            
            im = axes.imshow(similarity_dict[ChannelName1][ChannelName1][:,:,1], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1,vmin=0,vmax=max_abs_val1))#, vmax = max_abs_val0, vmin = 0)
            
            ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val1]
            ticks.sort()
            
            cbar = axes.figure.colorbar(im, ax=axes, ticks = ticks)
            cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size = Title_size)
            cbar.ax.tick_params(labelsize=Label_size)
            # yticks = cbar.ax.get_yticks()
            # yticks = np.insert(yticks, 1, 1)
            # cbar.ax.set_yticks(yticks)
            
            axes.spines[:].set_visible(False)
            
            axes.set_xticks(np.arange(len(full_names),step=5))#, labels = full_names)
            axes.set_yticks(np.arange(len(full_names),step=5))#, labels = full_names)
            axes.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)    
            
            plt.setp(axes.get_xticklabels())#, rotation=-45, ha="right", rotation_mode="anchor")
            plt.setp(axes.get_yticklabels())#, rotation=30, ha="right", rotation_mode="anchor")
            
            #breakpoint()
            
            axes.grid(which="minor", color="w", linestyle="-", linewidth=10)
            axes.tick_params(which="minor", bottom=False, left=False)      
        
            fig.suptitle('Similarity Scores Calculated With Maximum Closest Distances')
            fig.tight_layout()
            plt.savefig(ChannelName1.replace(" ","")+'_MSE_similarity_max.png')
            
        for i in range(distances.shape[-1]):
            ChannelName1 = dataset.data_dict["DataNames"][i]
            
            fig, axes = plt.subplots(figsize=(12,12))
            axes.set_box_aspect(1.0)
            axes.set_title(ChannelName1,size=Title_size)

            axes.set_ylabel("Case A", size = Title_size)
            axes.set_xlabel("Case B", size = Title_size)
            
            im = axes.imshow(similarity_dict[ChannelName1][ChannelName1][:,:,2], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1,vmin=0,vmax=max_abs_val2))#, vmax = max_abs_val0, vmin = 0)
            
            ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val2]
            ticks.sort()
            
            cbar = axes.figure.colorbar(im, ax=axes, ticks = ticks)
            cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size = Title_size)
            cbar.ax.tick_params(labelsize=Label_size)
            # yticks = cbar.ax.get_yticks()
            # yticks = np.insert(yticks, 1, 1)
            # cbar.ax.set_yticks(yticks)
            
            axes.spines[:].set_visible(False)
            
            axes.set_xticks(np.arange(len(full_names),step=5))#, labels = full_names)
            axes.set_yticks(np.arange(len(full_names),step=5))#, labels = full_names)
            axes.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)    
            
            plt.setp(axes.get_xticklabels())#, rotation=-45, ha="right", rotation_mode="anchor")
            plt.setp(axes.get_yticklabels())#, rotation=30, ha="right", rotation_mode="anchor")
            
            #breakpoint()
            
            axes.grid(which="minor", color="w", linestyle="-", linewidth=10)
            axes.tick_params(which="minor", bottom=False, left=False)      
        
            fig.suptitle('Similarity Scores Calculated With Average Maximum Distances')
            fig.tight_layout()
            plt.savefig(ChannelName1.replace(" ","")+'_MSE_similarity_avgmax.png')
        
        if not load_distances:
            """ Calculate all-channel similarity """
            print("Calculating 9-channel similarity")
            paramset1_no = 0
            for paramset1 in sampled_data:
                sample1 = cp.asarray(sampled_data[paramset1]) # move all data to GPU
                self_distance1_array = squareform(pdist(sample1).get())
                self_distancemax1 = cp.amax(self_distance1_array, axis = 0)#, where = self_distance1_array>0, initial = 0.0)
                self_distancemin1 = cp.asarray(np.min(self_distance1_array, axis = 0, where = self_distance1_array>0, initial = 1000))
                
                self_distance1_avg = cp.average(self_distancemin1).get()
                self_distance1_max = cp.amax(self_distancemin1).get()
                self_distance1_avgmax = cp.average(self_distancemax1).get()
                
                paramset2_no = 0
                for paramset2 in sampled_data:
                    # Calculate simularity of paramset 1 data to paramset 2 data for the channel1xchannel2 plot
                    if paramset1_no == paramset2_no: # This is always going to be self similar
                        similarity_dict["full"][paramset1_no, paramset2_no,0] = 0
                        similarity_dict["full"][paramset1_no, paramset2_no,1] = 0
                        similarity_dict["full"][paramset1_no, paramset2_no,2] = 0
                        continue
                        
                    if paramset2_no > paramset1_no: # Only calculate if in the lower triangular, to avoid calculating the distances for every pair twice
                        continue
                        
                    sample2 = cp.asarray(sampled_data[paramset2]) # move all data to GPU
                    self_distance2_array = squareform(pdist(sample2).get())
                    self_distancemax2 = cp.amax(self_distance2_array, axis = 0)#, where = self_distance2_array>0, initial = 0.0)
                    self_distancemin2 = cp.asarray(np.min(self_distance2_array, axis = 0, where = self_distance2_array>0, initial = 1000))
                    self_distance2_avg = cp.average(self_distancemin2).get()
                    self_distance2_max = cp.amax(self_distancemin2).get()
                    self_distance2_avgmax = cp.average(self_distancemax2).get()
                    
                    cross_distance = cdist(sample1,sample2)
                    
                    cross_distancemin1 = cp.amin(cross_distance, axis = 0)
                    cross_distancemax1 = cp.amax(cross_distance, axis = 0)
                    cross_distancemin2 = cp.amin(cross_distance, axis = 1)
                    cross_distancemax2 = cp.amax(cross_distance, axis = 1)
                    
                    cross_distance = 0
                    
                    cross_distance1_avg = cp.average(cross_distancemin1).get()
                    cross_distance1_max = cp.amax(cross_distancemin1).get()
                    cross_distance1_avgmax = cp.average(cross_distancemax1).get()
                    
                    cross_distance2_avg = cp.average(cross_distancemin2).get()
                    cross_distance2_max = cp.amax(cross_distancemin2).get()
                    cross_distance2_avgmax = cp.average(cross_distancemax2).get()
                    
                    similarity1_avg = cross_distance1_avg/self_distance1_avg
                    similarity2_avg = cross_distance2_avg/self_distance2_avg
                    
                    similarity1_max = cross_distance1_max/self_distance1_max
                    similarity2_max = cross_distance2_max/self_distance2_max
                    
                    similarity1_avgmax = cross_distance1_avgmax/self_distance1_avgmax
                    similarity2_avgmax = cross_distance2_avgmax/self_distance2_avgmax
                    
                    # Lower triangular is axis 0 WRT axis 1
                    # Upper triangular is axis 1 WRT axis 0
                    similarity_dict["full"][paramset1_no, paramset2_no,0] = similarity1_avg
                    similarity_dict["full"][paramset2_no, paramset1_no,0] = similarity2_avg
                    
                    similarity_dict["full"][paramset1_no, paramset2_no, 1] = similarity1_max
                    similarity_dict["full"][paramset2_no, paramset1_no, 1] = similarity2_max
                    
                    similarity_dict["full"][paramset1_no, paramset2_no, 2] = similarity1_avgmax
                    similarity_dict["full"][paramset2_no, paramset1_no, 2] = similarity2_avgmax
                    
                    
                    paramset2_no += 1
                    
                paramset1_no += 1
                
            if save_distances:
                with open('similarity_dict.pkl', 'wb') as f:
                    pickle.dump(similarity_dict,f)
                    
        elif load_distances:
            print("Data already loaded")
        
        max_abs_val_full0 = np.max(similarity_dict["full"][:,:,0])
        max_abs_val_full1 = np.max(similarity_dict["full"][:,:,1])
        max_abs_val_full2 = np.max(similarity_dict["full"][:,:,2])

        
        fig, axes = plt.subplots(figsize=(12,12))
        axes.set_box_aspect(1.0)
        
        axes.set_ylabel("Case A", size = Title_size)
        axes.set_xlabel("Case B", size = Title_size)
        
        im = axes.imshow(similarity_dict["full"][:,:,0], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1,vmin=0,vmax=max_abs_val_full0))#, vmax = max_abs_val0, vmin = 0)
        ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val_full0]
        ticks.sort()
        cbar = axes.figure.colorbar(im, ax=axes, ticks = ticks)
        cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size = Title_size)
        cbar.ax.tick_params(labelsize=Label_size)
        # yticks = cbar.ax.get_yticks()
        # yticks = np.insert(yticks, 1, 1)
        # cbar.ax.set_yticks(yticks)
        
        axes.spines[:].set_visible(False)
        
        axes.set_xticks(np.arange(len(full_names),step=5))
        axes.set_yticks(np.arange(len(full_names),step=5))
        axes.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)    
        
        plt.setp(axes.get_xticklabels())#, rotation=-45, ha="right", rotation_mode="anchor")
        plt.setp(axes.get_yticklabels())#, rotation=30, ha="right", rotation_mode="anchor")
        
        #breakpoint()
        
        axes.grid(which="minor", color="w", linestyle="-", linewidth=10)
        axes.tick_params(which="minor", bottom=False, left=False)      

        fig.suptitle('Similarity Scores of 9-Channel Data Calculated With Average Closest Distances')
        fig.tight_layout()
        plt.savefig('full_MSE_similarity_avg.png')
        
        
        
        
        
        fig, axes = plt.subplots(figsize=(12,12))
        axes.set_box_aspect(1.0)
        
        axes.set_ylabel("Case A", size = Title_size)
        axes.set_xlabel("Case B", size = Title_size)
        
        im = axes.imshow(similarity_dict["full"][:,:,1], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1,vmin=0,vmax=max_abs_val_full1))#, vmax = max_abs_val0, vmin = 0)
        ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val_full1]
        ticks.sort()
        cbar = axes.figure.colorbar(im, ax=axes, ticks = ticks)
        cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size = Title_size)
        cbar.ax.tick_params(labelsize=Label_size)
        # yticks = cbar.ax.get_yticks()
        # yticks = np.insert(yticks, 1, 1)
        # cbar.ax.set_yticks(yticks)
        
        axes.spines[:].set_visible(False)
        
        axes.set_xticks(np.arange(len(full_names),step=5))
        axes.set_yticks(np.arange(len(full_names),step=5))
        axes.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)    
        
        plt.setp(axes.get_xticklabels())#, rotation=-45, ha="right", rotation_mode="anchor")
        plt.setp(axes.get_yticklabels())#, rotation=30, ha="right", rotation_mode="anchor")
        
        #breakpoint()
        
        axes.grid(which="minor", color="w", linestyle="-", linewidth=10)
        axes.tick_params(which="minor", bottom=False, left=False)      

        fig.suptitle('Similarity Scores of 9-Channel Data Calculated With Maximum Closest Distances')
        fig.tight_layout()
        plt.savefig('full_MSE_similarity_max.png')
        
        
        fig, axes = plt.subplots(figsize=(12,12))
        axes.set_box_aspect(1.0)
        
        axes.set_ylabel("Case A", size = Title_size)
        axes.set_xlabel("Case B", size = Title_size)
        
        im = axes.imshow(similarity_dict["full"][:,:,2], interpolation=None, cmap = my_colormap, norm = colors.TwoSlopeNorm(vcenter=1,vmin=0,vmax=max_abs_val_full2))#, vmax = max_abs_val0, vmin = 0)
        ticks = [0,0.5,1,500,1000,2000,4000,6000, max_abs_val_full2]
        ticks.sort()
        cbar = axes.figure.colorbar(im, ax=axes, ticks = ticks)
        cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom", size = Title_size)
        cbar.ax.tick_params(labelsize=Label_size)
        # yticks = cbar.ax.get_yticks()
        # yticks = np.insert(yticks, 1, 1)
        # cbar.ax.set_yticks(yticks)
        
        axes.spines[:].set_visible(False)
        
        axes.set_xticks(np.arange(len(full_names),step=5))
        axes.set_yticks(np.arange(len(full_names),step=5))
        axes.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)    
        
        plt.setp(axes.get_xticklabels())#, rotation=-45, ha="right", rotation_mode="anchor")
        plt.setp(axes.get_yticklabels())#, rotation=30, ha="right", rotation_mode="anchor")
        
        #breakpoint()
        
        axes.grid(which="minor", color="w", linestyle="-", linewidth=10)
        axes.tick_params(which="minor", bottom=False, left=False)      

        fig.suptitle('Similarity Scores of 9-Channel Data Calculated With Average Maximum Distances')
        fig.tight_layout()
        plt.savefig('full_MSE_similarity_avgmax.png')
    
    
    
    
    plt.show()
    
    
    
        
            

    

