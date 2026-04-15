# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:37:33 2022

@author: mason

Calculates, but does not plot, the jacobians of the model's predictions with respect to the input cubes. Tracks the max and min at each voxel.
Multiplies by average variance.

Based on Convnext v1
"""

import numpy as np
import pandas as pd
import pickle
rng=np.random.default_rng()

import matplotlib.pyplot as plt

import time
import datetime
from itertools import zip_longest
import meshio
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.func import vmap, jacrev, functional_call
from torch import nn

from tensordict.tensordict import TensorDict

from operator import itemgetter

import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from timm.models.layers import trunc_normal_, DropPath

from typing import Iterator, Sequence
        
Testing = False
ensemble_size = 'largest'
ensemble_treatment = 'strict'
ensemble_comparison = False # Note: this should do nothing at present
Training = "All"   # True "All" 
# Set training to True when only using part of your available data

Model_name = "PrancingPony2_ADnlgP90b256"
#Model_name = "CosmicKiwi_ADnlgP25b256"
#Model_name = "AmpleArtichoke_MDnlgP90b256"
#Model_name = "PurplePanda_MDnlgP90b256"
#Model_name = "PoingantPenguin_MDnlgP90b256"
#Model_name = "WatchfulWombat_MDnlgP90b256"
#Model_name = "EnergeticEmu_MDnlgP25b256"
#Model_name = "BrilliantBalloon_MDnlgP90b256"
#Model_name = "FancyFinches_MDnlgP90b256"

# Lets you use a scratch directory for the microstructure data if running in a different directory.
# Currently assumes scratch home directory is /scratch/
# Implemented for use with memmaps
use_scratch = True
scratch_dir = 'mydir'
metadata_file = "data_points_metadata.csv"

# Disable the pre-check to see if the raw data files exist:
# Only set to true if the data dict has already been populated (and the raw files are gone)
DISABLE_RAW_FILE_CHECK = True 

device = "cuda:7" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def get_data_filenames(Testing, ensemble_size, ensemble_treatment, ensemble_comparison, Training, scratch_path, metadata_file):
    MS_folders = [1,5,10,15,20,25,50,75,100,200]

    # load information about data
    data_metadata = pd.read_csv(metadata_file, keep_default_na = False, dtype={'ModelLayersThermal': str, 'LargestEnsemble': str})

    # Separate out the data we want to train on
    if Training == "All":
        training_metadata = data_metadata
    else:
        training_metadata = data_metadata.loc[data_metadata.loc[:,'TrainingData']==Training,:]
    
    comparison_files = []    
    comparison_files_MS = []

    training_files_MS = []
    training_files_thermal = []
    
    if ensemble_size == 'largest' and ensemble_comparison:
        print("Warning: Ensemble size set to largest and largest ensemble comparison enabled. This will result in duplicate data loading. Turning off comparison.")
        ensemble_comparison = False
    
    for index, row in training_metadata.iterrows():
        training_layers = [int(ele) for ele in row['ModelLayersMS'].split(',')]
        #print(len(training_layers))
        
        # Get the largest ensemble size for each process parameter set (and layer if applicable)
        # So that we can chack requested ensemble size against it
        largest_ensemble = [int(ele) for ele in row['LargestEnsemble'].split(',')]
        
        # If we're tring to train on the largest ensemble for each process parameter set
        # then use the largest ensemble
        if ensemble_size == 'largest':
            ensemble_size_params = largest_ensemble
        # If we aren't, use the set ensemble size. We'll check for cases that don't exist when we go to load the data
        else:
            ensemble_size_params = [ensemble_size]

        # Check for mismatch in training layers specified. Might be worth adding more checks
        # But for now we just assume thermal layers = MS layers
        if len(training_layers) > 1 and row['ModelLayersThermal'] == 'None':
            print('Warning: Skipping '+row['MS_name']+', multiple MS layers indicated but "None" listed for thermal')
        
        # If we pass the first check, deal with the single layer thermal layers = None condition
        elif len(training_layers) == 1 and row['ModelLayersThermal'] == 'None':
            print('Single layer: '+row['MS_name'])
            
            # Check for mismatch in size of largest ensemble list vs number of layers
            # If there is more than one listed, for this single layer print, skip
            if len(ensemble_size_params) > 1:
                print("Warning: Largest ensemble list contains more than one element. Skipping.")
                continue
            # Check directly if ensemble comparison is enabled
            elif ensemble_comparison and len(largest_ensemble)>1:
                print("Warning: Largest ensemble list contains more than one element. Skipping.")
                continue
            
            # Set the local ensemble size for a single layer
            ensemble_size_local = ensemble_size_params[0]
            
            # If the value of the requested ensemble size is larger than the largest ensemble then skip or use largest
            if ensemble_size_local > largest_ensemble[0]:
                if ensemble_treatment.lower() == 'strict':
                    print("Requested ensemble size too large. Skipping.")
                    continue
                elif ensemble_treatment.lower() == 'loose':
                    print("Requested ensemble size too large. Using next largest available.")
                    ensemble_size_local = largest_ensemble[0]
                else:
                    print("Warning: 'ensemble_treatment' command not recognized. Using strict")
                    print("Requested ensemble size too large. Skipping.")
                    continue
            
            # Check if there is a folder for this ensemble size, if so prepend it
            if ensemble_size_local in MS_folders:
                folder = "n"+str(ensemble_size_local)+"/"
            else: 
                folder = ""
            
            layer = training_layers[0]
            
            # Add filenames to list
            training_files_MS.append(scratch_path+'MS_data/'+folder+row['MS_name']+'_hist'+str(int(layer))+'_n'+str(int(ensemble_size_local))+'.npy')
            training_files_thermal.append('thermal_data/'+row['Thermal_name']+'.xmf')
            # If comparing to largest (ensemble_comparison=True) add largest ensemble to list
            if ensemble_comparison:
                if largest_ensemble[0] in MS_folders:
                    folder = "n"+str(largest_ensemble[0])+"/"
                else: 
                    folder = ""
                comparison_files_MS.append(scratch_path+'MS_data/'+folder+row['MS_name']+'_hist'+str(int(layer))+'_n'+str(int(largest_ensemble[0]))+'.npy')
        
        # Know that thermal layers not equal to None
        # Check if thermal layers match MS layers, if not, skip
        elif not row['ModelLayersThermal'] == 'None':
            #print('Checking: '+row['MS_name'])
            #breakpoint()
            #print('type: ' + str(type(row['ModelLayersThermal']))+' value: ' +  str(row['ModelLayersThermal']))
            thermal_layers = [int(ele) for ele in row['ModelLayersThermal'].split(',')]
            
            if thermal_layers != training_layers:
                print('Warning: Skipping '+row['MS_name']+', MS layers mismatch with thermal layers')
                continue
            
            # Multi-layer case
            if len(training_layers) > 1:
                print('Multiple layers: '+row['MS_name'])
               
                # Check for mismatch in size of largest ensemble list vs number of layers
                # If there is a mismatch, skip
                if ensemble_size == "largest" and len(ensemble_size_params) != len(training_layers):
                    print("Warning: Largest ensemble list not equal to number of layers. Skipping.")
                    continue
                       
                # Index for keeping track of largest ensemble size
                layer_index = 0
                for layer in training_layers:
                    
                    # If using the largest ensemble size, set local ensemble size to appropriate size for layer
                    if ensemble_size == "largest":
                        ensemble_size_local = ensemble_size_params[layer_index]
                    # Otherwise set it to the base ensemble size
                    else: 
                        ensemble_size_local = ensemble_size_params[0]
                    
                        # Check if the requested ensemble size is too large
                        if ensemble_size_local > largest_ensemble[layer_index]:
                            if ensemble_treatment == 'strict':
                                print("Requested ensemble size too large. Skipping.")
                                continue
                            elif ensemble_treatment == 'loose':
                                print("Requested ensemble size too large. Using next largest available.")
                                ensemble_size_local = largest_ensemble[0]
                            else:
                                print("Warning: 'ensemble_treatment' command not recognized. Using strict")
                                print("Requested ensemble size too large. Skipping.")
                                continue
                    
                    # Check if there is a folder for this ensemble size, if so prepend it
                    if ensemble_size_local in MS_folders:
                        folder = "n"+str(ensemble_size_local)+"/"
                    else: 
                        folder = ""
                    
                    # Add filenames to list
                    training_files_MS.append(scratch_path+'MS_data/'+folder+row['MS_name']+'_hist'+str(int(layer))+'_n'+str(int(ensemble_size_local))+'.npy')
                    training_files_thermal.append('thermal_data/'+row['Thermal_name']+'_L'+str(int(layer))+'.xmf')
                    # If comparing to largest (ensemble_comparison=True) add largest ensemble to list
                    if ensemble_comparison:
                        if largest_ensemble[layer_index] in MS_folders:
                            folder = "n"+str(largest_ensemble[layer_index])+"/"
                        else: 
                            folder = ""
                        comparison_files_MS.append(scratch_path+'MS_data/'+folder+row['MS_name']+'_hist'+str(int(layer))+'_n'+str(int(largest_ensemble[layer_index]))+'.npy')
                    
                    layer_index += 1
            
            # Single layer case
            elif len(training_layers) == 1:
                print('Single layer: '+row['MS_name'])
                
                # Check for mismatch in size of largest ensemble list vs number of layers
                # If there is more than one listed, for this single layer print, skip
                if len(ensemble_size_params) > 1:
                    print("Warning: Largest ensemble list contains more than one element. Skipping.")
                    continue
                
                # Set the local ensemble size for a single layer
                ensemble_size_local = ensemble_size_params[0]
                
                # If the value of the requested ensemble size is larger than the largest ensemble then skip or use largest
                if ensemble_size_local > largest_ensemble[0]:
                    if ensemble_treatment == 'strict':
                        print("Requested ensemble size too large. Skipping.")
                        continue
                    elif ensemble_treatment == 'loose':
                        print("Requested ensemble size too large. Using next largest available.")
                        ensemble_size_local = largest_ensemble[0]
                    else:
                        print("Warning: 'ensemble_treatment' command not recognized. Using strict")
                        print("Requested ensemble size too large. Skipping.")
                        continue
                
                # Check if there is a folder for this ensemble size, if so prepend it
                if ensemble_size_local in MS_folders:
                    folder = "n"+str(ensemble_size_local)+"/"
                else: 
                    folder = ""
                
                layer = training_layers[0]
            
                # Add filenames to list
                training_files_MS.append(scratch_path+'MS_data/'+folder+row['MS_name']+'_hist'+str(int(layer))+'_n'+str(int(ensemble_size_local))+'.npy')
                training_files_thermal.append('thermal_data/'+row['Thermal_name']+'_L'+str(int(layer))+'.xmf')
                # If comparing to largest (ensemble_comparison=True) add largest ensemble to list
                if ensemble_comparison:
                    if largest_ensemble[0] in MS_folders:
                        folder = "n"+str(largest_ensemble[0])+"/"
                    else: 
                        folder = ""
                    comparison_files_MS.append(scratch_path+'MS_data/'+folder+row['MS_name']+'_hist'+str(int(layer))+'_n'+str(int(largest_ensemble[0]))+'.npy')
                
            else:
                print('Warning: Skipping '+row['MS_name']+"Something went wrong, number of layers less than one")
        else:
            print('Warning: ModelLayersThermal somehow equal to None')
        
        if Testing == True and index>2:
                    break
                    
                    
    i = 0            
    while i < (len(training_files_MS)):
        if not DISABLE_RAW_FILE_CHECK:
            if os.path.isfile(training_files_MS[i]) == False:
                print("Warning: " + training_files_MS[i] + " not found, removing from list")
                del training_files_MS[i]
                del training_files_thermal[i]
                del comparison_files_MS[i]
                
                i = i-1
                if i >= len(training_files_MS):
                    break
                continue
            if os.path.isfile(training_files_thermal[i]) == False:
                print("Warning: " + training_files_thermal[i] + " not found, removing from list")
                del training_files_MS[i]
                del training_files_thermal[i]
                del comparison_files_MS[i]
                
                i = i-1
                if i >= len(training_files_MS):
                    break
                    
            if ensemble_comparison:    
                if os.path.isfile(comparison_files_MS[i]) == False:
                    print("Warning: " + comparison_files_MS[i] + " not found, removing from list")
                    del training_files_MS[i]
                    del training_files_thermal[i]
                    del comparison_files_MS[i]
                    
                    i = i-1
                    if i >= len(training_files_MS):
                        break
            
        i+=1
        
            
    microstructure_files = training_files_MS
    thermal_files = training_files_thermal
    comparison_files = comparison_files_MS

    
    files = zip_longest(thermal_files, microstructure_files, comparison_files)
    
    return files

def PreloadData(files, path: str, ensemble_comparison = False, input_width = 25):
    '''
    Checks for a pre-saved TensorDict and loads it if it exists, otherwise creates it
    
    Path is the location to check for/save the TensorDict
    '''
    
    # Check if TensorDict already exists
    try:
        data_dict = TensorDict.load_memmap(path)
        #breakpoint()
        TensorDict_exists = True
    except:
        data_dict = {}
        data_dict["names"] = []
        TensorDict_exists = False

    
    changes_made = False
    
    training_names = []
    training_sizes = []
    if ensemble_comparison:
        comparison_sizes = []
    else:
        comparison_sizes = None
    
    for thermal_file, microstructure_file, comparison_file in files:
    
        paramset_name = thermal_file.split(".npy")[0].split("/")[1]
        training_names.append(paramset_name)
        
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
            data_dict["DataNames"] = param_names
            
            # Add the appropriate padding to the thermal data and save to data_dict 
            data_dict[paramset_name]["arrayT"] = np.pad(thermal_data, ((x_min_pad, x_max_pad), (y_min_pad, y_max_pad), (z_min_pad, half_width), (0,0)))
            data_dict[paramset_name]["pad"] = np.array([x_min_pad, y_min_pad, z_min_pad])
            
            # Create list of melted indices
            melted_ind = np.array(np.where(point_data["Melt Count"] > 0)).T
            #rng.shuffle(melted_ind) 
            data_dict[paramset_name]["indices"] = np.int32(melted_ind)
        
        
        # Get the ensemble size string from the microstructure file name
        ensemble_size = microstructure_file.split('.npy')[0].split('_n')[1]
        training_sizes.append(ensemble_size)
        
        # Check if this ensemble size is already in the TensorDict, if so skip
        if ensemble_size in data_dict[paramset_name]["arrayM"].keys():
            print(microstructure_file+" pre-populated, skipping microstructure file ")
        # Otherwise load and process
        else:
            changes_made = True
            # Load data files
            print("Loading ", microstructure_file)
            microstructure_data = np.load(microstructure_file)
            
            # Compare dimensions if it's a new paramset, otherwise assume check has passed with a different identically sized MS file
            # Note that this breaks the 1-too-tall MS fix, but this should be avoided anyways
            if new_paramset:
                # check z dimension compatibility of data, remove top layer from microstructure data if applicable
                if microstructure_data.shape[2] > zdim:
                    zdiff = microstructure_data.shape[2] - zdim
                    if zdiff == 1:
                        print('Cutting off top layer of microstructure data: ' + microstructure_file)
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
        
            # Process the microstructure data
            num_runs = np.sum(microstructure_data[0,0,0,:])
            fractional_microdata = microstructure_data/num_runs
            fractional_microdata[np.isnan(fractional_microdata)]=0
            fractional_microdata = np.float32(fractional_microdata)
        
            # Save the microstructure data  to data_dict
            data_dict[paramset_name]["arrayM"][ensemble_size] = fractional_microdata

        if ensemble_comparison:
            comparison_ensemble_size = comparison_file.split('.npy')[0].split('_n')[1]
            comparison_sizes.append(comparison_ensemble_size)
            if comparison_ensemble_size in data_dict[paramset_name]["arrayM"].keys():
                print(comparison_file+" pre-populated, skipping microstructure file ")
            else:
                changes_made = True
                print("Loading ", comparison_file)
                data_dict[paramset_name]["LEname"] = comparison_ensemble_size
                if comparison_file is not None:
                    comparison_data = np.load(comparison_file)
                else:
                    raise Exception("ensemble comparison enables, but no comparison file given")

                # Process the comparison microstructure data
                num_runs_comparison = np.sum(comparison_data[0,0,0,:])
                fractional_microdata_comparison = comparison_data/num_runs_comparison
                fractional_microdata_comparison[np.isnan(fractional_microdata_comparison)]=0
                fractional_microdata_comparison = np.float32(fractional_microdata_comparison)
                
                # Save the microstructure data to the dictionary
                data_dict[paramset_name]["arrayM"][comparison_ensemble_size] = fractional_microdata_comparison            
        
    # If there wasn't already a tensordict, convert the dict to one
    if not TensorDict_exists:
        tensor_dataDict = TensorDict.from_dict(data_dict)
        tensor_dataDict.set_non_tensor("names",data_dict["names"])
        tensor_dataDict.set_non_tensor("DataNames",data_dict["DataNames"])
        # Save the tensordisct as a memmap
        tensor_dataDict.memmap(path, num_threads=32)
        
    # If it alredy existed, and changes were made, save changes inplace
    elif changes_made:
        data_dict.memmap_(path, num_threads=32)
        tensor_dataDict = data_dict
        
    # If it already existed and nothing was changed, don't do anything
    else:
        tensor_dataDict = data_dict
        
    # If it already existed and nothing was changed, don't do anything
    
    print('Done pre-loading data')
    return (tensor_dataDict, training_names, training_sizes, comparison_sizes)

class InfSurrogateDataset(Dataset):
    
    def __init__(self, files, path, input_width = 25, ensemble_comparison = False):     
        self.half_width = int(np.floor((input_width-1)/2))

        self.inf_indices = []
        
        self.data_dict, inf_names, comparison_sizes, _ = PreloadData(files, path, ensemble_comparison, input_width)
        
        self.comp_ensembles = comparison_sizes
        
        self.ensemble_comparison = ensemble_comparison
        
        self.name_index = np.int32(0)
        self.n_voxels = np.int32(0)
                
        for inf_name in inf_names:
            
            # Get the number of data points for this file
            num_melted = len(self.data_dict[inf_name]["indices"])
            #print("num melted:"+str(num_melted))

            self.n_voxels += num_melted       
            
            for i in range(num_melted):
                self.inf_indices.append([self.name_index, i])
             
            self.name_index += np.int32(1)
            
        print("num voxels:"+str(self.n_voxels)+"\n")
        
        self.inf_indices = np.array(self.inf_indices, dtype=np.int32)
        rng.shuffle(self.inf_indices)
        num_samples = int(self.n_voxels*.1)
        self.sample_indices = self.inf_indices[:num_samples,:]
                   
    def __getitem__(self, indexes, transform=False):
        #print(indexes)
        #breakpoint()
        paramset_ind, index = indexes
        paramset = self.data_dict["names"][paramset_ind]

        ensemble_size = self.comp_ensembles[paramset_ind]

        indices = self.data_dict[paramset]["indices"][index]
        thermal_range_upper = indices + self.half_width + self.data_dict[paramset]["pad"] + 1
        thermal_range_lower = indices - self.half_width + self.data_dict[paramset]["pad"]
        thermal_sample = self.data_dict[paramset]["arrayT"][thermal_range_lower[0]:thermal_range_upper[0], 
                                           thermal_range_lower[1]:thermal_range_upper[1], 
                                           thermal_range_lower[2]:thermal_range_upper[2], 
                                           :].detach().clone()
        
        #thermal_sample = thermal_sample.permute(3,0,1,2)
        
        rotate_p = torch.randint(4,(1,))
        flip_p = torch.randint(2,(2,))
        
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

            
        if flip_p[0] == 1:
            thermal_sample = np.flip(thermal_sample, axis=0)
            thermal_sample[:,:,:,3] = -thermal_sample[:,:,:,3]
        if flip_p[1] == 1:
            thermal_sample = np.flip(thermal_sample, axis=1)
            thermal_sample[:,:,:,4] = -thermal_sample[:,:,:,4]
            
        thermal_sample = torch.as_tensor(thermal_sample.copy()).permute(3,0,1,2)
        
        return thermal_sample
        
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

class Inference:
    """
    Run inference over all the thermal inputs, then sort outputs into respective arrays.
    Janky implementation because I didn't want to redo how all the data is structured.
    """
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.inf_sampler = InfSampler(self.dataset.inf_indices)
        self.inf_loader = DataLoader(self.dataset, batch_size = 4096, shuffle=False, sampler=self.inf_sampler, num_workers=32, pin_memory=True)
        self.outputs_list = []
        self.net = net.to(device)
        
        self.results = {}
        
    def _run_inference(self):
        self.net.eval()

        with torch.no_grad():
            for i, x in enumerate(tqdm(self.inf_loader)):
                
                x = x.to(device,non_blocking=True)
                outputs = self.net(x)
                outputs[outputs<1e-5] = 0.0
                self.outputs_list.append(outputs.cpu().numpy())

                
        self.outputs_list = np.vstack(self.outputs_list)
        
    def _assemble_results(self):
        paramset_ind = self.dataset.inf_indices[:,0]
        all_indices = self.dataset.inf_indices[:,1]
        
        for name_index in range(self.dataset.name_index):
            name = self.dataset.data_dict["names"][name_index]
            ensemble_size = self.dataset.comp_ensembles[name_index]
            
            self.results[name] = np.zeros_like(self.dataset.data_dict[name]["arrayM"][ensemble_size])
            self.results[name][:,:,:,0] = -1
            
            rows = np.where(paramset_ind==name_index)
            #breakpoint()
            indeces = self.dataset.data_dict[name]["indices"][all_indices[rows]]
            
            self.results[name][indeces[:,0],indeces[:,1],indeces[:,2],:] = self.outputs_list[rows]
    
    def inference(self):
        self._run_inference()
        self._assemble_results()

class Importance:
    """
    Run inference over all the thermal inputs, then sort outputs into respective arrays.
    Calculate Jacobian statistics and input statistics.
    """
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.inf_sampler = InfSampler(self.dataset.sample_indices)
        self.inf_loader = DataLoader(self.dataset, batch_size = 512, shuffle=False, sampler=self.inf_sampler, num_workers=16, pin_memory=True)
        self.net = net.to(device)
        
        dimension = 25
        dimensions = [9,dimension,dimension,dimension]
        
        self.max_vals_jac = torch.zeros(dimensions, device = device)#[234,376,34,50])#
        self.min_vals_jac = torch.ones(dimensions, device = device)*1000#[234,376,34,50])#
        self.mean_abs_vals_jac = torch.zeros(dimensions, device = device)
        self.mean_jac = torch.zeros(dimensions, device = device)
        self.M2_jac = torch.zeros(dimensions, device = device)
        self.variance_jac = 0
        
        self.max_vals_in = torch.zeros(dimensions, device = device)#[234,376,34,50])#
        self.min_vals_in = torch.ones(dimensions, device = device)*1000#[234,376,34,50])#
        self.mean_abs_vals_in = torch.zeros(dimensions, device = device)
        self.mean_in = torch.zeros(dimensions, device = device)
        self.M2_in = torch.zeros(dimensions, device = device)
        self.variance_in = 0
        
        self.count_jac = 0
        self.count_in = 0
        
        self.results = {}
    
    def _functional_net(self, inputs):
        return functional_call(self.net, dict(self.net.named_parameters()), inputs.unsqueeze(dim=0))
        
        
    def _compute_jacobians(self):
        self.net.eval()

        with torch.no_grad():
            for i, x in enumerate(tqdm(self.inf_loader)):
                
                x = x.to(device,non_blocking=True)
                self.count_jac += x.shape[0]*51
                # Calculate Jacobian
                outputs = vmap(jacrev(self._functional_net, argnums = 0), in_dims = 0)(x) # APPEARS TO WORK!!!!
                
                # Calculate Jacobian statistics
                deltas_jac = outputs - self.mean_jac.unsqueeze(0).unsqueeze(0)
                self.mean_jac += torch.sum(deltas_jac, dim = (0,1))/self.count_jac
                delta2_jac = outputs - self.mean_jac.unsqueeze(0).unsqueeze(0)
                self.M2_jac += torch.sum(deltas_jac*delta2_jac, dim = (0,1))/self.count_jac
                
                max_outputs_jac = torch.max(torch.max(outputs, dim=0).values, dim=0).values
                min_outputs_jac = torch.min(torch.min(outputs, dim=0).values, dim=0).values

                self.max_vals_jac = torch.max(max_outputs_jac,self.max_vals_jac)
                self.min_vals_jac = torch.min(min_outputs_jac,self.min_vals_jac)
        
                self.mean_abs_vals_jac += torch.sum(torch.abs(outputs),dim = (0,1))
                
                # Caclulate input statistics
                self.count_in += x.shape[0]
                deltas_in = x - self.mean_in.unsqueeze(0)
                self.mean_in += torch.sum(deltas_in, dim = (0))/self.count_in
                delta2_in = x - self.mean_in.unsqueeze(0)
                self.M2_in += torch.sum(deltas_in*delta2_in, dim = (0))/self.count_in
                
                # max_inputs_in = torch.max(x, dim=0).values
                # min_inputs_in = torch.min(x, dim=0).values

                self.max_vals_in = torch.max(torch.max(x, dim=0).values,self.max_vals_in)
                self.min_vals_in = torch.min(torch.min(x, dim=0).values,self.min_vals_in)
        
                self.mean_abs_vals_in += torch.sum(torch.abs(x),dim = 0)
                
                #breakpoint()

                
        
    def _finalize(self):
        #self.mean = self.mean/self.count
        self.mean_abs_vals_jac = self.mean_abs_vals_jac/self.count_jac
        self.variance_jac = self.M2_jac/self.count_jac
        
        self.mean_abs_vals_in = self.mean_abs_vals_in/self.count_in
        self.variance_in = self.M2_in/self.count_in
    
    def inference(self):
        self._compute_jacobians()
        self._finalize()

           

            
if __name__ == '__main__':    
    if use_scratch:
        scratch_path = '/scratch/'+scratch_dir+'/'
    else:
        scratch_path = ''
        
    files = get_data_filenames(Testing, ensemble_size, ensemble_treatment, ensemble_comparison, Training, scratch_path, metadata_file)
    dataset = InfSurrogateDataset(files, path= '/scratch/'+scratch_dir+'/data_tensordict', input_width = 25, ensemble_comparison = ensemble_comparison)
    print("finished loading")
    
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
    net = ConvNeXt(optim_config=config)
    net.load_state_dict(torch.load(Model_name+'_final.pt'))
    print('net initialized: '+Model_name)
    
    Imp = Importance(dataset, net)
    Imp.inference()
    
    np.save('max_jacobians.npy', Imp.max_vals_jac.cpu().detach().numpy())
    np.save('min_jacobians.npy', Imp.max_vals_jac.cpu().detach().numpy())

    np.save('Welford_variance_jacobian.npy', Imp.variance_jac.cpu().detach().numpy())
    np.save('Welford_deviation_jacobian.npy', np.sqrt(Imp.variance_jac.cpu().detach().numpy()))
    np.save('Welford_mean_jacobian.npy', Imp.mean_jac.cpu().detach().numpy())
    
    np.save('mean_abs_jacobian.npy', Imp.mean_abs_vals_jac.cpu().detach().numpy())
    
    
    np.save('max_inputs.npy', Imp.max_vals_in.cpu().detach().numpy())
    np.save('min_inputs.npy', Imp.max_vals_in.cpu().detach().numpy())

    np.save('Welford_variance_inputs.npy', Imp.variance_in.cpu().detach().numpy())
    np.save('Welford_deviation_inputs.npy', np.sqrt(Imp.variance_in.cpu().detach().numpy()))
    np.save('Welford_mean_inputs.npy', Imp.mean_in.cpu().detach().numpy())
    
    np.save('mean_abs_inputs.npy', Imp.mean_abs_vals_in.cpu().detach().numpy())
    

