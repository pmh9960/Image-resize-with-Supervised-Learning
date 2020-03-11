from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras import optimizers, initializers
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from data_setting import load_dataset, slicing_images, flatting_images, save_history
import time, os
from keras.utils.vis_utils import plot_model

current_time = time.time()
current_time_str = time.strftime("%Y%m%d_%H%M", time.localtime(current_time))

print("Loading Data... ")
start = time.time()
(
    train_set_images128_orig,
    train_set_images256_orig,
    train_set_images512_orig,
) = load_dataset("dataset_300cats")
# print(train_set_images128_orig.shape)
# m_train = train_set_images128_orig.shape[0]
print(round(time.time() - start, 2), "s")

print("Resize Data...")
start = time.time()
num_slice = 8
train_set_images128, train_set_images256 = slicing_images(
    num_slice, train_set_images128_orig, train_set_images256_orig
)

train_set_images128_flatten, train_set_images256_flatten = flatting_images(
    train_set_images128, train_set_images256
)

train_set_X = train_set_images128_flatten / 255.0
train_set_Y = train_set_images256_flatten / 255.0
print(round(time.time() - start, 2), "s")

# print(train_set_X.shape, train_set_Y.shape)
print("Making model...")
start = time.time()
model = Sequential()
layers_dim = []
input_dim = int((128 / num_slice) ** 2 * 3)  # 768
layers_dim.append(input_dim)
layers_dim.append(5000)
output_dim = int((256 / num_slice) ** 2 * 3)  # 3072
layers_dim.append(output_dim)

# First hidden layer
model.add(
    Dense(
        layers_dim[1],
        input_dim=layers_dim[0],
        activation="relu",
        kernel_initializer=initializers.he_normal(),
    )
)
# Hidden layers
for i in range(len(layers_dim) - 3):
    model.add(
        Dense(
            layers_dim[i + 2],
            activation="relu",
            kernel_initializer=initializers.he_normal(),
        )
    )
# Output layer
model.add(
    Dense(
        layers_dim[len(layers_dim) - 1],
        activation="sigmoid",
        kernel_initializer=initializers.he_normal(),
    )
)
model.compile(
    loss="mean_squared_error", optimizer=optimizers.Adam(), metrics=["accuracy"],
)
print(round(time.time() - start, 2), "s")

history = model.fit(
    train_set_X,
    train_set_Y,
    validation_split=0.25,
    epochs=300,
    batch_size=100,
    shuffle=True,
)


print("Saving model...")
start = time.time()
os.mkdir("models/" + current_time_str)
# Save model with json format
model_json = model.to_json()
with open("models/" + current_time_str + "/model" + ".json", "w") as json_file:
    json_file.write(model_json)

# Save weight with h5 format
model.save_weights("models/" + current_time_str + "/weight.h5")

with open(
    "models/" + current_time_str + "/num_slice" + ".txt", "w", encoding="utf-8",
) as num_file:
    num_file.write(str(num_slice))
save_history(history, current_time_str)


plot_model(model, to_file="models/" + current_time_str + "/model.png", show_shapes=True)
print(round(time.time() - start, 2), "s")
