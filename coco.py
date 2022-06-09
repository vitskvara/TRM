# EDIT THESE PATHS ACCORDING TO WHERE YOUR RAW DATA IS SAVED/TO BE SAVED
# where the processed coco data will be saved
output_dir = '../sgad_data/raw_datasets/coco/processed'
# where the train images are saved
coco_path = '../sgad_data/raw_datasets/coco/train2017'
# location of the annotation file
anot_path = '../sgad_data/raw_datasets/coco/annotations/instances_train2017.json'

import os, sys, time, io, subprocess, requests
import numpy as np
import random

from PIL import Image
from pycocotools.coco import COCO
from skimage.transform import resize

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import h5py
################ Paths and other configs - Set these #################################
CLASSES = [
        'boat',
        'airplane',
        'truck',
        'dog',
        'zebra',
        'horse',
        'bird',
        'train',
        'bus',
        'motorcycle'
        ]
imsize = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

NUM_CLASSES = len(CLASSES)
ANOMALY = 0

h5pyfname = output_dir
print(h5pyfname,os.path.exists(h5pyfname))
if not os.path.exists(h5pyfname):
    os.makedirs(h5pyfname)

# we dont use this function below
def getClassName(cID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == cID:
            return cats[i]['name']
    return 'None'

# lets go
tr_i = 500*NUM_CLASSES
val_i = 500*NUM_CLASSES

train_fname = os.path.join(h5pyfname,f'train_{imsize}.h5py')
val_fname = os.path.join(h5pyfname,f'validation_{imsize}.h5py')

if os.path.exists(train_fname): subprocess.call(['rm', train_fname])
if os.path.exists(train_fname): subprocess.call(['rm', val_fname])

train_file = h5py.File(train_fname, mode='w')
val_file = h5py.File(val_fname, mode='w')

# create subsets
train_file.create_dataset('resized_images', (tr_i,3,imsize,imsize), dtype=np.dtype('float32'))
val_file.create_dataset('resized_images', (val_i,3,imsize,imsize), dtype=np.dtype('float32'))
train_file.create_dataset('resized_mask', (tr_i,3,imsize,imsize), dtype=np.dtype('float32'))
val_file.create_dataset('resized_mask', (val_i,3,imsize,imsize), dtype=np.dtype('float32'))

# g stands for 'group'
train_file.create_dataset('y', (tr_i,), dtype='int32')
val_file.create_dataset('y', (val_i,), dtype='int32')

# load coco
coco = COCO(anot_path)
cats = coco.loadCats(coco.getCatIds())

# now iterate over classes
tr_s, val_s = 0, 0
for c in range(NUM_CLASSES):
    catIds = coco.getCatIds(catNms=[CLASSES[c]])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    i = -1
    tr_si = 0
    print('Class {} (train) : #images = {}'.format(c, len(images)))
    while tr_si < tr_i//NUM_CLASSES:
        i += 1

        # get the image
        im = images[i]
        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']

        if max_ann < 10000: continue;
        img_path = os.path.join(coco_path, im['file_name'])
        I = np.asarray(Image.open(img_path))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (imsize, imsize), anti_aliasing=True)
        resized_image = resize(I, (imsize, imsize), anti_aliasing=True)
        train_file['resized_images'][tr_s, ...] = np.transpose(resized_image, (2,0,1))
        train_file['resized_mask'][tr_s, ...] = np.transpose(resized_mask, (2, 0, 1))
        train_file['y'][tr_s] = c

        tr_s += 1
        tr_si += 1
        if tr_si % 100 == 0:
            print('>'.format(c), end='')
            time.sleep(1)
    print(' ')

    val_si = 0
    while val_si < val_i//NUM_CLASSES:
        i += 1

        # get the image
        im = images[i]

        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']

        if max_ann < 10000: continue;

        img_path = os.path.join(coco_path, im['file_name'])
        I = np.asarray(Image.open(img_path))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (imsize, imsize), anti_aliasing=True)
        resized_image = resize(I, (imsize, imsize), anti_aliasing=True)

        # val_id:
        val_file['resized_images'][val_s, ...] = np.transpose(resized_image, (2,0,1))
        val_file['resized_mask'][val_s, ...] = np.transpose(resized_mask, (2, 0, 1))
        val_file['y'][val_s] = c

        val_s += 1
        val_si += 1
        if val_si % 100 == 0: print('>'.format(c), end='')
    print('')

train_file.close()
val_id_file.close()
