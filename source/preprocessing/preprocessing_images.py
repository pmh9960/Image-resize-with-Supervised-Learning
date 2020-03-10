import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
import os
import h5py


def preprocess_images(folder_name, file_name=None):
    if file_name == None:
        file_list = os.listdir(folder_name)
        image_list = [
            file
            for file in file_list
            if file.endswith(".jpg")
            or file.endswith(".jpeg")
            or file.endswith(".bmp")
            or file.endswith(".png")
        ]
        # print(image_list)
    else:
        file_list = os.listdir(folder_name)
        image_list = [file for file in file_list if file.startswith(file_name)]

    num_pxs = (128, 256, 512)

    for image_name in image_list:
        for num_px in num_pxs:
            fname = folder_name + "/" + image_name

            # pip install scipy==1.1.0 (need)
            image = Image.open(fname)
            image = image.convert("RGB")  # OSError: cannot write mode LA as JPEG

            margin = int(abs(image.size[0] - image.size[1]) / 2)
            crop_image = (
                image.crop((margin, 0, image.size[0] - margin, image.size[1]))
                if image.size[0] > image.size[1]
                else image.crop((0, margin, image.size[0], image.size[1] - margin))
            )

            reshaped_image = crop_image.resize((num_px, num_px))

            if image_name[-5] == ".":
                sliced_name = image_name[:-5]
            else:
                sliced_name = image_name[:-4]
            save_path = (
                folder_name
                + "/preprocessed_images/"
                + sliced_name
                + "_"
                + str(num_px)
                + ".jpg"
            )
            reshaped_image.save(save_path)


def preprocessed_images_to_h5py(folder_name, file_names, index=0):
    path_image = folder_name + "/preprocessed_images/"
    path_hdf5 = "datasets/" + folder_name + str(index) + ".hdf5"

    images128 = []
    images256 = []
    images512 = []
    for file_name in file_names:
        path128 = path_image + file_name + "_128.jpg"
        path256 = path_image + file_name + "_256.jpg"
        path512 = path_image + file_name + "_512.jpg"
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
    f.close()
