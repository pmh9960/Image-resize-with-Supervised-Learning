import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_setting import load_dataset, slicing_images, flatting_images
from keras.models import model_from_json
from keras import optimizers
import keras
from keras.utils.vis_utils import plot_model
from preprocessing import preprocessing_images


def predict_my_image(model, filename, num_slice):

    index = 0
    preprocessing_images.preprocess_images("my_images", filename)
    preprocessing_images.preprocessed_images_to_h5py(
        "my_images", [filename], index=index
    )

    test128_orig, test256_orig, test512_orig = load_dataset("my_images" + str(index))
    test128, test256 = slicing_images(num_slice, test128_orig, test256_orig)

    test128_flatten, _ = flatting_images(test128, test256)

    testX = test128_flatten / 255.0

    predict_testY = model.predict(testX)
    for i in range(num_slice):
        for j in range(num_slice):
            if j == 0:
                result = predict_testY[i * num_slice + j].reshape(
                    (int(256 / num_slice), int(256 / num_slice), 3)
                )
            else:
                result = np.append(
                    result,
                    predict_testY[i * num_slice + j].reshape(
                        (int(256 / num_slice), int(256 / num_slice), 3)
                    ),
                    axis=1,
                )
        if i == 0:
            ans = result
        else:
            ans = np.append(ans, result, axis=0)

    plt.imshow(ans)
    plt.savefig("models/" + folder_name + "/" + filename + "_predict.png")
    plt.imshow(test128_orig.reshape(128, 128, 3))
    plt.savefig("models/" + folder_name + "/" + filename + "_128_orig.png")
    plt.imshow(test256_orig.reshape(256, 256, 3))
    plt.savefig("models/" + folder_name + "/" + filename + "_256_orig.png")


def load_model(folder_name):
    json_file = open("models/" + folder_name + "/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/" + folder_name + "/weight.h5")

    return loaded_model


def last_model():
    file_list = os.listdir("models")
    return file_list[-1]


# folder_name = "20200310_1711"
folder_name = last_model()
loaded_model = load_model(folder_name)
num_file = open("models/" + folder_name + "/num_slice.txt", "r")
num_slice = int(num_file.read())
predict_my_image(loaded_model, "juwon", num_slice)
