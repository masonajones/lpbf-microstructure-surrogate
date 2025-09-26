# -*- coding: utf-8 -*-
"""
WORK IN PROGRESS
This is the microstructure surrogate model hyperparameter optimization script, with multi-GPU support and pytorch tensordict based memmaps to minimize system RAM usage.
WORK IN PROGRESS (code structure changed with DDP, need to re-wrap in optim func)

To only use subset of GPUs on device (including single) run from command line using:
    CUDA_AVAILABLE_DEVICES=<Device IDs> python microstructure_surrogate_25_DDP.py
Replace <device IDs> with comma separated list of CUDA IDs
To see available devices use command on command line:
    nvidia-smi
Consider memory usage of available devices:
    0% usage does not mean avaialbe if memory usage is high
    Memory footprint of training is high
    Some seemingly idle processes also degrade performance

Needs Microstructure data stored locally, use:
    rsync -av --exclude *.tar.xz /gpfs/user/MS_data/ .
from scratch directory /scratch/mydir
Recommend using scratch drive if available to avoid networking overhead.

There was some speedup from compiling the model, but this has been disabled due to issues with DDP.
Compiling could still be useful for inference.

There are some remnant bits of code from experimenting with Automatic Mixed Precision (AMP)
But I have not been able to train the model successfully using AMP

File created on Tue May 10 09:37:33 2022
Last updated on Sun Oct 20 2024

@author: Mason Jones
    
"""

import os
#import pickle
import time
import datetime
from typing import Iterator, Sequence

from operator import itemgetter
from itertools import zip_longest
from tqdm import tqdm

import meshio
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from tensordict.tensordict import TensorDict

from timm.models.layers import trunc_normal_, DropPath

# Configure torch back end to ensure better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Initialize numpy random functions for sampling
rng = np.random.default_rng()

### BEGIN TRAINING SETTINGS ###
# Model will be saved using this name
Model_name = "testing_parallel"

# Provide name of data_points_metadata.csv for microstructure and thermal data:
metadata_filename = "data_points_metadata_singleexcludedL60.csv"

# Percentage of data to be set aside for training (the rest will be used for testing)
training_data_split = 0.50
# Number of epochs. Learning rate scheduler tied to this value, minimum LR at # epochs.
epochs = 4
        
# Options for telling the model which data to use
# Set testing to True to manually set specific data files to use inside get_data_filenames function
# Leave set to False to use data_points_metadata.csv for auto selection of data
Testing = False
# Tell the model what MS ensemble size to train on. Options: 'largest' or integer value of ensemble size, currently: 1,5,10,15,20,25,50,75,100,200.
ensemble_size = 'largest'
# Tell the model what to do for ensembles that do not have the given integer size. Options: 'strict' or 'loose'
# If set to 'loose' model will fall back to largest ensemble size available if requested size is larger
ensemble_treatment = 'strict'
# Tell the model if it should run tests using data from the largest ensembles
ensemble_comparison = False 
# Tell the model if it should only use the "training" portion of the available dataset, as defined in data_points_metadata.csv, or all of it
# Set training to true when only using part of your available data, else use "All"
# This is a training script, it will always train the model
Training = True  # Options: True "All" 

# Lets you use a scratch directory for the microstructure data if running in a different directory.
# Currently assumes scratch base directory is /scratch/ followed by <scratch_dir> given here
# Implemented for use with memmaps. HIGHLY SUGGEST USING SCRATCH IF ALTERNATIVE IS NETWORK ATTACHED.
use_scratch = True
scratch_dir = 'mydir' # 'masjone' #'mydir'

### END TRAINING SETTINGS ###

# OLD NAMES:
#Model_name = "CosmicKiwi_ADnlgP25b256"
#Model_name = "AmpleArtichoke_MDnlgP90b256"
#Model_name = "PurplePanda_MDnlgP90b256"
#Model_name = "PoingantPenguin_MDnlgP90b256"
#Model_name = "ArbitraryAlbatross_MDnlgP90b256"
#Model_name = "FancyFinches_MDnlgP90b256"
#Model_name = "EnergeticEmu_MDnlgP25b256"
#Model_name = "WatchfulWombat_MDnlgP25b256"


def get_data_filenames(Testing, ensemble_size, ensemble_treatment, ensemble_comparison, Training, scratch_path, metadata_filename):
    r"""
    Automatically generates a list of file names for PreloadData to use by reading metadata from metadata csv
    
    Some of the checks in here are a bit convoluted and might miss problems.
    """
    
    MS_folders = [1,5,10,15,20,25,50,75,100,200]

    # Load data metadata
    data_metadata = pd.read_csv(metadata_filename, keep_default_na = False, dtype={'ModelLayersThermal': str, 'LargestEnsemble': str})

    # Separate out the data we want to train on
    # Uglier than necessary because using boolean and string
    if Training == True:
        training_metadata = data_metadata.loc[data_metadata.loc[:,'TrainingData']==Training,:]
    elif Training == False:
        raise Exception("Invalid value for 'Training': use 'All' or True")
    elif Training.lower() == "all":
        training_metadata = data_metadata
    else:
        raise Exception("Invalid value for 'Training': use 'All' or True")
    
    # Initialize lists used for keeping track of file names
    comparison_files = []    
    comparison_files_MS = []

    # If not testing the algorithm, set up the list of datafiles corresponding to the chosen settings and metadata_filename: "data_points_metadata.csv"
    if Testing == False:
        training_files_MS = []
        training_files_thermal = []
        
        if ensemble_size.lower() == 'largest' and ensemble_comparison:
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
            if ensemble_size.lower() == 'largest':
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
                # if not we assume the file exists loose
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

        i = 0            
        while i < (len(training_files_MS)):
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

    if Testing == True:
        microstructure_files = [
                            'L1P90V105_hist1.npy'
                            ]

        thermal_files = [
                        'L1P90V105.xmf'
                        ]

        microstructure_files = ["MS_data/old_data/" + microstructure_file for microstructure_file in microstructure_files]
        thermal_files = ["thermal_data/" + thermal_file for thermal_file in thermal_files]
        comparison_files = []
    
    files = zip_longest(thermal_files, microstructure_files, comparison_files)
    
    return files

def PreloadData(files, path: str, ensemble_comparison = False, input_width = 25):
    '''
    Checks for a pre-saved TensorDict and loads it if it exists, otherwise creates it.
    In both cases checks if data is populated, and attempts to add it if it is not
    
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
        
        ### NOW CHECK ON MICROSTRUCTURE DATA ###
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
    
    return (tensor_dataDict, training_names, training_sizes, comparison_sizes)

class SurrogateDataset(Dataset):
    r"""
    The Dataset used by the DataLoader for getting datapoints for training.
    First creates list of all datapoints and splits them into test and training sets.
    When loading a datapoint, the data can be augmented and sampled using transform=True when calling __getitem__
    """
    
    def __init__(self, files, path, training_data_split = 0.05, input_width = 25, ensemble_comparison = False):
        self.flipcount=0
        self.rotcount=0
        self.sample=0
        self.half_width = int(np.floor((input_width-1)/2))
        self.always_sample = None
        self.num_resamples = 1000
        
        self.n_samples = 0
        self.n_training = 0
        self.n_test = 0
        self.test_indices = []
        self.train_indices = []
        
        self.data_dict, training_names, training_sizes, comparison_sizes = PreloadData(files, path, ensemble_comparison, input_width)
        
        self.train_ensembles = training_sizes
        
        self.ensemble_comparison = ensemble_comparison
        
        if self.ensemble_comparison:
            self.test_ensembles = comparison_sizes
        else: 
            self.test_ensembles = training_sizes
        
        name_index = np.int32(0)
        
                
        for training_name in training_names:
            
            # Get the number of data points for this file
            num_melted = len(self.data_dict[training_name]["indices"])
            #print("num melted:"+str(num_melted))
            
            # Split the data
            self.n_samples += num_melted
            split_ind = int(np.floor(num_melted*training_data_split))          
            self.n_training += split_ind
            
            self.n_test += (num_melted - split_ind)
            
            for i in range(split_ind):
                self.train_indices.append([name_index, i, True])
                
            for i in range(split_ind, num_melted):
                self.test_indices.append([name_index, i, False])
                
                
                
            name_index += np.int32(1)
            
        print("num training:"+str(self.n_training))
        print("num test:"+str(self.n_test), "\n")
        
        self.test_indices = np.array(self.test_indices, dtype=np.int32)
        self.train_indices = np.array(self.train_indices, dtype=np.int32)
        
                   
    def __getitem__(self, indexes, transform=False):
        #print(indexes)
        #breakpoint()
        paramset_ind, index, transform = indexes
        paramset = self.data_dict["names"][paramset_ind]
        
        # if transform is active, get ensemble size for training
        if transform:
            ensemble_size = self.train_ensembles[paramset_ind]
        # otherwise get sensemble size for testing
        else:
            ensemble_size = self.test_ensembles[paramset_ind]
        
        indices = self.data_dict[paramset]["indices"][index]
        thermal_range_upper = indices + self.half_width + self.data_dict[paramset]["pad"] + 1
        thermal_range_lower = indices - self.half_width + self.data_dict[paramset]["pad"]
        thermal_sample = self.data_dict[paramset]["arrayT"][thermal_range_lower[0]:thermal_range_upper[0], 
                                           thermal_range_lower[1]:thermal_range_upper[1], 
                                           thermal_range_lower[2]:thermal_range_upper[2], 
                                           :].detach().clone()
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
                
            thermal_sample = torch.as_tensor(thermal_sample.copy()).permute(3,0,1,2)
            
        else:
            thermal_sample = thermal_sample.permute(3,0,1,2)
                    
        # If ensemble_comparison is enabled and transform is False for this data, assume testing and return ensemble comparison data
        if self.ensemble_comparison and not transform:
            probs = self.data_dict[paramset]["arrayM"][ensemble_size][i,j,k,:].detach().clone()
        # Otherwise return normal microstructure data
        else:
            probs = self.data_dict[paramset]["arrayM"][ensemble_size][i,j,k,:].detach().clone()
        
        if sample_p == 1:
            sample = np.random.choice(51,self.num_resamples,p=probs.detach().numpy())

            microstructure_sample = torch.from_numpy(np.bincount(sample,minlength=51))+torch.rand(51)*1E-40
            #microstructure_sample = torch.as_tensor(microstructure_sample/np.sum(microstructure_sample),dtype=torch.float32) + torch.rand(51)*1E-15
            microstructure_sample = microstructure_sample/torch.sum(microstructure_sample)
            self.sample+=1
        else:
            #microstructure_sample = torch.as_tensor(probs/np.sum(probs),dtype=torch.float32)+torch.rand(51)*1E-15
            microstructure_sample = probs/torch.sum(probs)+torch.rand(51)*1E-40
        
        #thermal_sample = torch.as_tensor(thermal_sample.copy()).permute(3,0,1,2)
        
                
        return thermal_sample, microstructure_sample
        
    def __len__(self):
        return self.n_samples        

        
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

def ddp_setup(rank: int, world_size: int):
    """
    Args: 
        rank: Identifier of each process
        world_size: Number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Lets us get the correct dataset indices from the indices returned by DistributedSampler.
    Necessary because DistributedSampler returns random indices of len(dataset) rather than dataset[indices]
    Inspired by SpeechBrain, which is under Apache 2.0 license.
    No longer accepts a sampler, but intstead accepts the indices returned by Dataset.
    https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/dataio/sampler.html#DistributedSamplerWrapper
    """
    
    def __init__(self, indices, *args, **kwargs):
        super().__init__(dataset = indices, *args, **kwargs)
        self.indices = indices
        
    def __iter__(self):
        DistributedSampler_indices = super().__iter__()
        return iter(itemgetter(*DistributedSampler_indices)(self.indices))
    
    def set_epoch(self, epoch):
        """Pass epoch through to DistributedSampler for shuffle"""
        super().set_epoch(epoch)
    

def test(net, testloader):
    r"""
    Primary function for testing the model accuracy and speed against the data provided by testloader.
    Calculation of model accuracy means that the runtime is not the same as the inference time.
    """    
    
    sum_loss = 0.0  
    count = 0
    net.eval()
    with torch.no_grad():
        runtime=0
        #total_time = 0
        for i, (x, y) in enumerate(tqdm(testloader)):
            #start = torch.cuda.Event(enable_timing=True)
            #end = torch.cuda.Event(enable_timing=True)
            
            
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # return x,y
            count += y.shape[0]
            with torch.amp.autocast(device_type='cuda', enabled = False):
                start_time = time.perf_counter()
                #start.record()
                outputs = net(x)
                #end.record()
                #total_time += start.elapsed_time(end)
                torch.cuda.synchronize()
                runtime = runtime + (time.perf_counter()-start_time)
                diff = torch.abs(outputs - y)
                diff_square = torch.square(diff)
                loss = torch.sum(diff_square, dim=1)/51
                sum_loss += torch.sum(loss).item()
                
            #runtime = runtime + (time.time()-start_time)
            
    print('runtime ' +str(runtime)+' count '+str(count))
    mean_loss = sum_loss/count 
    return mean_loss, runtime
    
def test_time(net, testloader):
    r"""
    OLD?
    Function for timing model inference without computing loss, for more accurate inference time
    """
    count = 0
    net.eval()
    
    with torch.no_grad():
        runtime=0
        start = time.time()
        for i, (x, y) in enumerate(tqdm(testloader)):
            end = torch.cuda.Event(enable_timing=True)
            
            
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # return x,y
            count += y.shape[0]
            with torch.amp.autocast(device_type='cuda', enabled = False):
                outputs = net(x)
                end.record()                
            
    
    runtime = runtime + (time.time()-start)
    print('runtime ' +str(runtime)+' count '+str(count))
    
    return runtime

def inference(net, infloader):
    r"""
    OLD
    Runs inference using "net" for samples provided by "infloader". 
    Currently only works for a single simulation domain, which must be manually entered into this function code.
    
    Args:
        net (PyTorch model): the model being used for inference.
        infloader (Pytorch Dataloader): the dataloader used to load thermal inputs needed for inference.
    """
    microstructure = np.zeros([350,250,22,50])#[234,376,34,50])#
    net.eval()
    with torch.no_grad():
        for i, (x, indices) in enumerate(tqdm(infloader)):
            x = x.to(device)
            outputs = net(x)
            outputs[outputs<1e-5] = 0.0
            microstructure[tuple(indices.T)] = outputs.cpu().numpy()
#            for ind in range(len(indices)):
#                i,j,k = indices[ind]
#                microstructure[i,j,k,:] = outputs[ind].cpu().numpy()
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

def train_net(net, optimizer, scheduler, train_loader):  
    train_logger = {
            "batch_epoch": [],
            "batch_loss": [],
            "learning_rate": [],
            #"epoch_loss": [],
            "epoch": [],
            }

    scaler = torch.cuda.amp.GradScaler(growth_interval=10,backoff_factor=.9375,growth_factor=1.125, enabled=False)
    min_scale=512
    scaler_reset = 0
    iters = len(train_loader)
    loss_Func = nn.MSELoss()
    epochs =  1
    j=0
    for epoch in range(epochs):
        # DDP shuffle
        train_loader.sampler.set_epoch(epoch)
        
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
                # print(outputs[1]-y[1])
                train_logger["batch_epoch"].append((epoch*j)+(i+1))
                train_logger["batch_loss"].append(running_loss/4000)
                train_logger["learning_rate"].append(scheduler.get_last_lr()[0])
                #train_logger["epoch_loss"].append((epoch*j)+(i+1))
                train_logger["epoch"].append(epoch+1)
                print("batch:", i+1, "batch_loss:", running_loss/4000)#, "scaler_resets:", scaler_reset)
                running_loss = 0.0

        j = i+1
        epoch_loss += running_loss
        epoch_time = datetime.timedelta(seconds=time.time()-epochstart)

        train_logger["batch_epoch"].append((epoch*j)+(i+1))
        train_logger["batch_loss"].append(running_loss/((i+1)%4000))
        train_logger["learning_rate"].append(scheduler.get_last_lr()[0])
        #train_logger["epoch_loss"].append((epoch*j)+(i+1))
        train_logger["epoch"].append(epoch+1)
        print("batch:", i+1, "batch_loss:", running_loss/((i+1)%4000))
        
        print('Epoch took: %s' %epoch_time)
        chkpt=net.module.state_dict()
        if gpu_id == 0:
            _save_checkpoint(epoch)
        
    train_logger_df = pd.DataFrame(train_logger)
    train_logger_df.to_csv("train_logger_25.csv")

class Trainer:
    def __init__(
        self,
        net,
        loss_func,
        train_loader,
        test_loader,
        optimizer, 
        # scaler,
        scheduler,
        scheduler_iters,
        gpu_id: int,
        world_size: int
        ) -> None:  
        self.gpu_id = gpu_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        # self.scaler = scaler
        self.scheduler = scheduler
        self.scheduler_iters = scheduler_iters
        self.net = net.to(gpu_id)
        self.net = DDP(net, device_ids = [gpu_id],find_unused_parameters=True)
        ### Used to compile the model but was having problems with DDP
        #self.net = torch.compile(self.net)
        self.loss_Func = loss_func
        self.world_size = world_size

        self.running_loss = 0.0
        
        if gpu_id == 0:
            self.head_process = True
        else:
            self.head_process = False
        
        self. j = 0 # I don't remember how this is supposed to work, but it's to help with indexing for logging
        
        self.train_logger = {
                "batch_epoch": [],
                "batch_loss": [],
                "learning_rate": [],
                #"epoch_loss": []
                "epoch": [],
                }
        self.test_logger = {
                "epoch": [],
                "test_loss": []
                }
    
    ### Used to compile the training but was having too many problems with DDP
    ### Theoretically this is better than compiling just the model in __init__ (also disabled currently)
    #@torch.compile          
    def _run_batch(self, i, x, y, epoch):
        #x, y = torch.as_tensor(x).permute(0,4,1,2,3), torch.as_tensor(y).permute(0,4,1,2,3)
        x, y = x.to(self.gpu_id, non_blocking=True), y.to(self.gpu_id, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        #with torch.amp.autocast(device_type='cuda', enabled=False):#, dtype=torch.float32),enabled=False
        outputs = self.net(x)
        loss = self.loss_Func(outputs, y)
        # self.scaler.scale(loss).backward()
        loss.backward()
        self.optimizer.step()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        self.scheduler.step(epoch + i / self.scheduler_iters)
        #if scaler._scale<min_scale:
         #   scaler_reset+=1
        #    scaler._scale = torch.tensor(min_scale).to(scaler._scale
        
        return loss.item()

    def _run_epoch(self, epoch):
        # DDP shuffle
        self.train_loader.sampler.set_epoch(epoch)
        
        epochstart = time.time()
        epoch_loss = 0.0
        
        #with tqdm(self.train_loader, disable = not self.head_process) as pbar:
        
        for i, (x, y) in enumerate(tqdm(self.train_loader, disable = not self.head_process)): 
            loss = self._run_batch(i, x, y, epoch)
            self.running_loss += loss
            
            if self.gpu_id == 0 and (i+1) % 4000 == 0:
                self.j = np.max([i+1, self.j])
                self.train_logger["batch_epoch"].append((epoch*self.j)+(i+1))
                self.train_logger["batch_loss"].append(self.running_loss/4000)
                self.train_logger["learning_rate"].append(self.scheduler.get_last_lr()[0])
                self.train_logger["epoch"].append(epoch+1)
                print("batch:", i+1, "batch_loss:", self.running_loss/4000)
                self.running_loss = 0.0
        
        if self.gpu_id == 0:
            epoch_time = datetime.timedelta(seconds=time.time()-epochstart)
            print('Epoch took: %s' %epoch_time)
            self.j = i+1
            epoch_loss += self.running_loss
            self.train_logger["batch_epoch"].append((epoch*self.j)+(i+1))
            self.train_logger["batch_loss"].append(self.running_loss/((i+1)%4000))
            self.train_logger["learning_rate"].append(self.scheduler.get_last_lr()[0])
            self.train_logger["epoch"].append(epoch+1)
            print("batch:", i+1, "batch_loss:", self.running_loss/((i+1)%4000))

            
           
    def _save_checkpoint(self, epoch, filename = None):
        checkpoint = self.net.module.state_dict()
        if filename is None:
            path = "epoch"+str(epoch)+"_checkpoint.pt"
            torch.save(checkpoint, path)
            print("Checkpoint Saved")
            
        else:
            torch.save(checkpoint, filename)
    
    def train(self, epochs: int):
        for epoch in range(epochs):
            self.net.train()
            self._run_epoch(epoch)
            
            # Only check test data on GPU 0, would rather use all GPUs but need all reduce. Probably exists, but haven't looked into it.
            if self.gpu_id == 0:
                self._save_checkpoint(epoch)
                
            test_loss = self.test()
            if self.gpu_id == 0:
                print('GPU0 loss: ' +str(test_loss))
            dist.barrier()
            dist.reduce(test_loss, 0)
            if self.gpu_id == 0:
                test_loss = test_loss.item()/self.world_size
                print('world size: '+ str(self.world_size) +' loss: ' +str(test_loss))
                self.test_logger["epoch"].append(epoch+1)
                self.test_logger["test_loss"].append(test_loss)     
                
    def test(self):
        sum_loss = 0.0  
        count = 0
        self.net.eval()
        with torch.no_grad():
            #runtime=0
            for i, (x, y) in enumerate(tqdm(self.test_loader, disable = not self.head_process)):
                start = time.time()
                #x, y = torch.as_tensor(x).permute(0,4,1,2,3), torch.as_tensor(y).permute(0,4,1,2,3)
                x, y = x.to(self.gpu_id, non_blocking=True), y.to(self.gpu_id, non_blocking=True)
                count += y.shape[0]
                with torch.amp.autocast(device_type='cuda', enabled = False):
                    outputs = self.net(x)
                    diff = torch.abs(outputs - y)
                    diff_square = torch.square(diff)
                    loss = torch.sum(diff_square, dim=1)/51
                    sum_loss += torch.sum(loss)
        mean_loss = sum_loss/count 
        return mean_loss
            
            
def main(rank: int, world_size: int, epochs: int, dataset, config, Model_name):
    r"""
    This is where the actual machine learning stuff happens.
    Note that this is run in parallel once per active GPU. For things that should only happen once, check for rank 0 first.
    """
        
    print('Starting rank: '+str(rank))
    # Initialize Distributed Data Parallel settings for current rank
    ddp_setup(rank, world_size)

    # Create samplers
    test_sampler = DistributedSamplerWrapper(dataset.test_indices, shuffle=False) # Wrap DistributedSampler so that we can return dataset specific indices
    train_sampler = DistributedSamplerWrapper(dataset.train_indices, shuffle=True) # Wrap DistributedSampler so that we can return dataset specific indices

    # Create dataloaders
    train_loader = DataLoader(dataset, batch_size = int(2*128/world_size), shuffle=False, sampler=train_sampler, num_workers=6, pin_memory=True)
    test_loader = DataLoader(dataset, batch_size = 1024, shuffle=False, sampler=test_sampler, num_workers=4, pin_memory=True)

    # Initialize model
    net = Model(optim_config=config)
    
    # Optimizer config
    optimizer = optim.AdamW(net.parameters(), lr = torch.tensor(0.001))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, epochs, eta_min =0.5E-5, T_mult = 2)
    scheduler_iters = len(train_loader) # Needed to update scheduler. Might be a better way to do this
    
    ### AMP stuff (Currently turned off)
    # scaler = torch.cuda.amp.GradScaler(growth_interval=10,backoff_factor=.9375,growth_factor=1.125, enabled=False)
    # min_scale=512
    # scaler_reset = 0

    # Use MSE for loss function
    loss_Func = nn.MSELoss()
    
    ### Train model
    # Initialize training funtion
    trainer = Trainer(net, loss_Func, train_loader, test_loader, optimizer, scheduler, scheduler_iters, rank, world_size)
    print('Start training rank '+str(rank))
    # Train
    trainer.train(epochs)
    
    # Save the training and testing log data from GPU 0
    if rank == 0:
        filename = Model_name+"_final.pt"
        trainer._save_checkpoint(0, filename = filename)
        print("Model Saved: " + filename)
        train_logger_df = pd.DataFrame(trainer.train_logger)
        train_logger_df.to_csv(Model_name+"_train_logger.csv")
        test_logger_df = pd.DataFrame(trainer.test_logger)
        test_logger_df.to_csv(Model_name+"_test_logger.csv")
        return
    destroy_process_group()
    
def objective(rank: int, world_size: int, epochs: int, dataset, config, Model_name):    
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
    

if __name__ == '__main__':    
    """
    Set everything up to be used for training the model.

    """
    # Check if the given filename already exists. If it does add a number and give warning
    if os.path.isfile(Model_name+"_final.pt") == True:
        print("Warning: Model with this name already exists")
        i = 1
        while os.path.isfile(Model_name+"_("+str(i)+")_final.pt") == True:
            i+=1
            # Used to check if a bunch of files with the same name exist just in case the name kept getting reused accidentally
            # if i>10:
                # print("Warning: too many files with same name, quitting")
                # quit()
                #break
        Model_name = Model_name + "_" + str(i)
        print("New model name: " + Model_name)
        
    else:
        print("Model name: " +Model_name)
    
    # Check if set to use the scratch directory of the node, and set the appropriate directory.
    # For convenience, doesn't change behavior. Recommend using scratch drive if available to avoid networking overhead.
    if use_scratch:
        scratch_path = '/scratch/'+scratch_dir+'/'
    else:
        scratch_path = ''
    
    # Automatically generate list of files if configured
    files = get_data_filenames(Testing, ensemble_size, ensemble_treatment, ensemble_comparison, Training, scratch_path, metadata_filename)
    # Load the data and create the dataset
    dataset = SurrogateDataset(files, path= '/scratch/'+scratch_dir+'/data_tensordict', training_data_split = training_data_split, input_width = 25, ensemble_comparison = ensemble_comparison)
    print("finished loading")
    
    # Initialize model hyperparams to best known config
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
    # config = best_so_far
    
    sampler = optuna.integration.BoTorchSampler(n_startup_trials=4)
    study = optuna.create_study(directions=["maximize", "minimize"], 
        study_name = "multilayer_tk3", 
        storage = "sqlite:///optimization3.db", 
        load_if_exists=True)
    
    world_size = torch.cuda.device_count()
    func = lambda trial: objective(rank, world_size, epochs, dataset, config, Model_name)
    #mp.spawn(main, args=(world_size, epochs, dataset, config, Model_name), nprocs=world_size)
    
    study.optimize(func, n_trials=10)
    plot = optuna.visualization.plot_pareto_front(study, targets=lambda t: ((t.values[0]/1000),t.values[1]), target_names=["Performance [points/second]","MSE Loss"])
    plot.update_yaxes(mirror=True,linewidth=2, gridwidth=2, range=[.0001,0.00015])
    plot.update_xaxes(mirror=True,linewidth=2, gridwidth=2)
    plot.update_layout(title=dict(text='Pareto Front Plot of Speed vs Accuracy', font=dict(size=40), automargin=True, yref='container', xanchor='center', x=0.5, yanchor = 'middle',y=0.95),plot_bgcolor='#eeeeee')
    plot.update_layout(xaxis=dict(titlefont=dict(size=35),tickfont=dict(size=25)),yaxis=dict(titlefont=dict(size=35),tickfont=dict(size=25)))
    plot.update_traces(marker=dict(size=12, line=dict(width = 2, color='DarkSlateGrey')))
    plot.update_traces(marker=dict(showscale=False))
    plot.update_layout(width=1000, height=800)
    plot.write_image("test_pareto.png")
    
    ### OLD CODE:
        
    # #train_sampler = SubsetRandomSampler(dataset.train_indices)
    # test_sampler = SubsetRandomSampler(dataset.test_indices)
    # train_sampler = DistributedSampler(dataset.train_indices, shuffle=True)
    # #test_sampler = DistributedSampler(dataset.test_indices)
    
    # print("samplers setup")
    
    # train_loader = DataLoader(dataset, batch_size = 128, shuffle=False, sampler=train_sampler, num_workers=8, pin_memory=True)
    # test_loader = DataLoader(dataset, batch_size = 1024, sampler=test_sampler, num_workers=8, pin_memory=True)
    # print("Dataloaders setup")
    
    # #net = Model(optim_config=config).to(device)
    # net = Model(optim_config=config)
    # net = DDP(net, device_ids=gpu_ids)
    # print("net setup")
    
    # optimizer = optim.AdamW(net.parameters(), lr = 0.001)
    # print("optimizer setup")
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4, eta_min =0.5E-5, T_mult = 2)
    # print("scheduler setup")
    
    # train_net(net, optimizer, scheduler, train_loader)  # Train the model
    # print("training finished")
    # train_loader = None
    # print("trainloader emptied")

    # loss = test(net, test_loader)  # let cudnn see test run before timing actual test run
    # print("test finished")
    # performance = len(dataset.test_indices)/best_time
    # print("loss [MSE]= ", loss)
    # print('performance [points/s] = ', performance)
    #torch.save(net.state_dict(), 'new_model_25.pt')

    

