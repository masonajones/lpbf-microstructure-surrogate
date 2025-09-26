# -*- coding: utf-8 -*-
"""
@author: Mason Jones

Converts 4D Numpy arrays into 3D XDMF mesh files with labeled data sets.

Input: np array (from npy file) of shape [n_x, n_y, n_z, n_datasets] or [n_x, n_y, n_z] if only 1 dataset
Output: xdmf file with shape [n_x, n_y, n_z] with n_datasets channels


"""

import meshio
import numpy as np

# Numpy filename, without extension
# Will be reused for saving XDMF file
filename = "numpy_file"

# spacing of data points
[x_spacing, y_spacing , z_spacing] = [5,5,5]

# Labels for data, only add more if n_datasets>1
data_labels = ["data",]

def Create_point_list(array_obj, x_spacing = 1, y_spacing = 1, z_spacing = 1):
	"""
	Turns the array based point data into lists for use with meshio
	Should only be used for initializing the mesh object, for later time steps us "Create_point_data"
	
	Parameters
	----------
	array_obj : A 3D or 4D numpy array

	Returns
	-------
	points : The coordinates of all the points
	point_data : A dictionary consisting of lists of the data at each point
	point_lookup : A 3D array with the list index corresponding to each point
	"""
	

	mesh_dims = array_obj.shape
	if len(mesh_dims) == 3:
		[nx,ny,nz] = mesh_dims
	else:
		[nx,ny,nz,_] = mesh_dims
	point_data = {}
	num_points = nx*ny*nz
	point_lookup = np.arange(num_points, dtype=int).reshape(nx,ny,nz)
	points_array = np.indices(array_obj.shape[:3])
	points_x = points_array[0,:,:,:].reshape(-1,1) * x_spacing
	points_y = points_array[1,:,:,:].reshape(-1,1) * y_spacing
	points_z = points_array[2,:,:,:].reshape(-1,1) * z_spacing
	points = np.hstack((points_x,points_y,points_z))	
	
	if len(mesh_dims) == 3:
		index = 0
		for label in data_labels:
			if index == 0:
				point_data[label] = array_obj[:,:,:].reshape(-1)
				index += 1
			else: print(label+" not used, not enough dimensions in data")
	else:
		index = 0
		for label in data_labels:
			point_data[label] = array_obj[:,:,:,index].reshape(-1)
			index += 1
	        
	return points, point_data, point_lookup

def DefineCell(indices,point_lookup):
	"""Creates an array of point indexes defining the cell. 
	Only supports hexahedral cells"""
	[i,j,k] = indices
	cell_points = np.array([
            point_lookup[i+0, j+0, k+0],
            point_lookup[i+1, j+0, k+0],
            point_lookup[i+1, j+1, k+0],
            point_lookup[i+0, j+1, k+0],
            point_lookup[i+0, j+0, k+1],
            point_lookup[i+1, j+0, k+1],
            point_lookup[i+1, j+1, k+1],
            point_lookup[i+0, j+1, k+1],
        ])
	return cell_points


def Create_cell_list(point_lookup):
	"""
	Returns a list of points defining each cell, formatted for the meshio writer

	Parameters
	----------
	point_lookup : A list of indexes corresponding to each point

	Returns
	-------
	cells : A list of points corresponding to each cell

	"""
	# May be faster to loop over points list
	mesh_dims = point_lookup.shape
	[nx,ny,nz] = mesh_dims
	cells_array = np.zeros(((nx-1)*(ny-1)*(nz-1),8))
	cell_index = 0
	for i in np.arange(nx-1):
		for j in np.arange(ny-1):
			for k in np.arange(nz-1):
				cell_points = DefineCell([i,j,k], point_lookup)
				cells_array[cell_index] = cell_points
				cell_index = cell_index+1
	cells = [("hexahedron", cells_array)]
	return cells

def ConvertToMesh(array_obj, x_spacing = 1, y_spacing = 1, z_spacing = 1):
	"""Converts the array used by the solver (variable name "mesh") to a meshio mesh"""
	[points, point_data, point_lookup] = Create_point_list(array_obj, x_spacing = x_spacing, y_spacing = y_spacing, z_spacing = z_spacing)
	cells = Create_cell_list(point_lookup)
	
	return points, point_data, cells

### FOR USE WITH TIME SERIES DATA: ###

def Create_point_data(array_obj):
	"""
	Turns the array based point data into lists for use with meshio
	Similar to "Create_point_list", except it only collects the data
	For use when the mesh data has already been initialized (i.e. when doing time series data)

	Parameters
	----------
	array_obj : TYPE
		DESCRIPTION.

	Returns
	-------
	point_data : TYPE
		DESCRIPTION.

	"""

	point_data = {}
	index = 0
	for label in data_labels:
		point_data[label] = array_obj[:,:,:,index].reshape(-1)
		index += 1

	return point_data


def ConvertDataToMesh(array_obj):
	point_data = Create_point_data(array_obj)
	return point_data

### END FOR USE WITH TIME SERIES DATA ###


data = np.load(filename+".npy")
# example data:
# data = np.ones([20,30,10,1],dtype=np.int8)

points, point_data, cells = ConvertToMesh(data, x_spacing = x_spacing, y_spacing = y_spacing, z_spacing = z_spacing)

meshio.write_points_cells(filename+".xdmf", points=points, cells=cells, point_data=point_data)
	
	