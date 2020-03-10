import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage

path_image = "preprocessed_images/cat_"
path_hdf5 = "datasets/dataset_300cats.hdf5"
num_images = 300
images128 = []
images256 = []
images512 = []

for i in range(num_images):
    path128 = path_image + str(i + 1) + "_128.jpg"
    path256 = path_image + str(i + 1) + "_256.jpg"
    path512 = path_image + str(i + 1) + "_512.jpg"

    image = np.array(ndimage.imread(path128, flatten=False))
    images128.append(image)
    image = np.array(ndimage.imread(path256, flatten=False))
    images256.append(image)
    image = np.array(ndimage.imread(path512, flatten=False))
    images512.append(image)

f = h5py.File(path_hdf5, "w")
train = f.create_group("train")
# dev = f.create_group("dev")
# test = f.create_group("test")
train.attrs["desc"] = "Training set"
# dev.attrs["desc"] = "Development set"
# test.attrs["desc"] = "Test set"

train.create_dataset("images128", dtype="int32", data=images128)
train.create_dataset("images256", dtype="int32", data=images256)
train.create_dataset("images512", dtype="int32", data=images512)
