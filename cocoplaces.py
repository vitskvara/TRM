# EDIT THESE PATHS ACCORDING TO WHERE YOUR RAW DATA IS SAVED/TO BE SAVED
imsize = 64
cocopath = '/home/skvarvit/generativead-sgad/sgad_data/raw_datasets/coco'
placespath = '/home/skvarvit/generativead-sgad/sgad_data/raw_datasets/places'
outpath = '/home/skvarvit/generativead-sgad/sgad_data/raw_datasets/cocoplaces'

import numpy as np
import h5py, os, sys
import random

# if the coco.py and places.py files are not changed, this is the matching of the classes
# in the uniform files
# boat/beach = 0/0
# airplane/bamboo_forest = 1/1
# truck/canyon = 2/2
# dog/forest = 3/3
# zebra/ball_pit = 4/4
# horse/orchard = 5/5
# bird/rock_arch = 6/6
# train/shower = 7/7
# bus/ski_slope = 8/8
# motorcycle/wheat_field = 9/9

# infiles
cocotrainf = os.path.join(cocopath, f'processed_{imsize}/train.h5py')
cocovalf = os.path.join(cocopath, f'processed_{imsize}/validation.h5py')
placesf = os.path.join(placespath, f'processed_{imsize}/data.h5py')

# outfiles
if not os.path.exists(outpath):
    os.makedirs(outpath)
uniff = os.path.join(outpath, f'uniform_data_{imsize}.npy')
mashf = os.path.join(outpath, f'mashed_data_{imsize}.npy')
unifyf = os.path.join(outpath, f'uniform_labels_{imsize}.npy')
mashyf = os.path.join(outpath, f'mashed_labels_{imsize}.npy')

# load the data
cocotrain = h5py.File(cocotrainf, 'r')
cocoval = h5py.File(cocovalf, 'r')
places = h5py.File(placesf, 'r')

# setup
s_train = cocotrain['resized_images'].shape
n_train = s_train[0]
s_val = cocoval['resized_images'].shape
n_val = s_val[0]
places_uniform = places['resized_place'][:][:,0:n_train//10,:,:]
places_uniform = places_uniform.reshape((-1,s_train[1],s_train[2],s_train[3]))
places_mashed = places['resized_place'][:][:,n_train//10:,:,:]
places_mashed = places_mashed.reshape((-1,s_val[1],s_val[2],s_val[3]))

# first create and save the uniform data that have the factors fixed
train_mask = cocotrain['resized_mask'][:] 
X_uniform = train_mask*cocotrain['resized_images'][:] + (1-train_mask)*places_uniform
y_uniform = cocotrain['y'][:]
y_places = np.concatenate([places['y'][1000*c:1000*(c+1)-500] for c in range(10)])
y_uniform = np.concatenate((y_uniform.reshape(-1,1),y_places.reshape(-1,1)),1)
np.save(uniff, X_uniform)
np.save(unifyf, y_uniform)

# now  do the same for the mashed data - but mix the factors
inds = np.array(random.sample(range(n_val), n_val))
val_mask = cocoval['resized_mask'][:][inds]
X_mashed = val_mask*cocoval['resized_images'][:][inds] + (1-val_mask)*places_mashed
y_mashed = cocoval['y'][:][inds]
y_places = np.concatenate([places['y'][1000*c+500:1000*(c+1)] for c in range(10)])
y_mashed = np.concatenate((y_mashed.reshape(-1,1),y_places.reshape(-1,1)),1)
np.save(mashf, X_mashed)
np.save(mashyf, y_mashed)