# Transfer Risk Minimization (TRM)

Forked from https://github.com/Newbeeer/TRM in order to prepare the SceneCOCO dataset.

Places dataset can be downloaded at:

http://data.csail.mit.edu/places/places365/train_256_places365standard.tar ; 

COCO dataset can be downloaded at:

http://images.cocodataset.org/zips/train2017.zip

COCO annotations can be downloaded from:

http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Extract only the needed places classes (in the dir where the downloaded .tar file is located):

```shell
tar -xvf train_256_places365standard.tar data_256/b/beach data_256/c/canyon data_256/b/building_facade data_256/s/staircase data_256/d/desert/sand data_256/c/crevasse data_256/b/bamboo_forest data_256/f/forest/broadleaf data_256/b/ball_pit data_256/k/kasbah data_256/l/lighthouse data_256/p/pagoda data_256/r/rock_arch data_256/o/oast_house data_256/o/orchard data_256/v/viaduct data_256/w/water_tower data_256/w/waterfall data_256/z/zen_garden
```

# Requirements

Install these Python libraries:

```
cython
torch
tqdm
requests
numpy
scikit-image
matplotlib
h5py
```

Also download https://github.com/cocodataset/cocoapi and install the Python api.

#### Preprocess the SceneCOCO dataset (you may want to first edit the files with the locations of the data downloaded above):

```shell
# preprocess COCO
python coco.py

# preprocess Places
python places.py

# generate SceceCOCO dataset
python cocoplaces.py
```

