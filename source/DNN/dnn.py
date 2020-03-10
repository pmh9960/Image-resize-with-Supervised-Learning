from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras import optimizers, initializers
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from data_setting import load_dataset, slicing_images, flatting_images
import time, os
from keras.utils.vis_utils import plot_model

current_time = time.time()
current_time_str = time.strftime("%Y%m%d_%H%M", time.localtime(current_time))

(
    train_set_images128_orig,
    train_set_images256_orig,
    train_set_images512_orig,
) = load_dataset("dataset_300cats")
# print(train_set_images128_orig.shape)
# m_train = train_set_images128_orig.shape[0]

num_slice = 4
train_set_images128, train_set_images256 = slicing_images(
    num_slice, train_set_images128_orig, train_set_images256_orig
)

train_set_images128_flatten, train_set_images256_flatten = flatting_images(
    train_set_images128, train_set_images256
)

train_set_X = train_set_images128_flatten / 255.0
train_set_Y = train_set_images256_flatten / 255.0

# print(train_set_X.shape, train_set_Y.shape)

model = Sequential()

model.add(
    Dense(
        6144,
        input_dim=3072,
        activation="tanh",
        kernel_initializer=initializers.he_normal(),
    )
)
model.add(
    Dense(12288, activation="sigmoid", kernel_initializer=initializers.he_normal(),)
)


model.compile(
    loss="mean_squared_error",
    optimizer=optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, decay=0.0),
    metrics=["accuracy"],
)

model.fit(train_set_X, train_set_Y, epochs=500, batch_size=300)

os.mkdir("models/" + current_time_str)
# Save model with json format
model_json = model.to_json()
with open("models/" + current_time_str + "/model" + ".json", "w") as json_file:
    json_file.write(model_json)

# Save weight with h5 format
model.save_weights("models/" + current_time_str + "/weight.h5")

plot_model(model, to_file="models/" + current_time_str + "/model.png", show_shapes=True)

