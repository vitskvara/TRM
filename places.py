# EDIT THESE PATHS ACCORDING TO WHERE YOUR RAW DATA IS SAVED/TO BE SAVED
imsize = 64
n_samples = 1000
output_dir = '../sgad_data/raw_datasets/places'
places_dir = '../sgad_data/raw_datasets/places/data_256'

import os, subprocess
import numpy as np
import random
from PIL import Image
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')

import h5py

biased_places = ['b/beach',
                 'b/bamboo_forest',
                 'c/canyon',
                 'f/forest/broadleaf',
                 'b/ball_pit',
                 'o/orchard',
                 'r/rock_arch',
                 's/shower',
                 's/ski_slope',
                 'w/wheat_field'
                 ]
NUM_CLASSES = len(biased_places)

print('----- Bias places ------')
print(biased_places)

dataset_name = f'processed_{imsize}'
h5pyfname = os.path.join(output_dir, dataset_name)
print('---------------- FNAME --------------')
print(h5pyfname)
if not os.path.exists(h5pyfname):
    os.makedirs(h5pyfname)
######################################################################################
biased_place_fnames = {}
for i, target_place in enumerate(biased_places):
    L = [f'{target_place}/{filename}' for filename in os.listdir(os.path.join(places_dir, target_place)) if filename.endswith('.jpg')]
    random.shuffle(L)
    biased_place_fnames[i] = L


tr_i = n_samples * NUM_CLASSES
train_fname = os.path.join(h5pyfname,'data.h5py')
if os.path.exists(train_fname): subprocess.call(['rm', train_fname])
train_file = h5py.File(train_fname, mode='w')
train_file.create_dataset('resized_place', (NUM_CLASSES,n_samples,3,imsize,imsize), dtype=np.dtype('float32'))
train_file.create_dataset('y', (NUM_CLASSES*n_samples,), dtype='int32')

tr_s, val_s, te_s = 0, 0, 0

for c in range(NUM_CLASSES):

    tr_si = 0
    print('Class {} (train) : '.format(c), biased_places[c], end=' ')
    while tr_si < n_samples:
        place_path = biased_place_fnames[c][tr_si]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))
        # that's the one:
        resized_place = resize(place_img, (imsize, imsize))
        train_file['resized_place'][c, tr_si, ...] = np.transpose(resized_place, (2,0,1))
        train_file['y'][c*n_samples + tr_si] = c
        tr_si += 1
        if tr_si % 100 == 0: print('>'.format(c), end='')
        ########################################
    print('')

train_file.close()
