import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_setting import load_dataset, slicing_images, flatting_images
from keras.models import load_model

# # Model visualization
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot, pydot
# import keras
# import pydot as pyd
# keras.utils.vis_utils.pydot = pyd

from keras.utils import plot_model


def test_my_image(model, filename, num_slice):
    test128_orig, test256_orig, test512_orig = load_dataset(filename)
    test128, test256 = slicing_images(num_slice, test128_orig, test256_orig)

    test128_flatten, test256_flatten = flatting_images(test128, test256)

    testX = test128_flatten / 255.0
    testY = test256_flatten / 255.0
    predict_testY = model.predict(testX)
    for i in range(4):
        for j in range(4):
            if j == 0:
                result = predict_testY[i * 4 + j].reshape((64, 64, 3))
            else:
                result = np.append(
                    result, predict_testY[i * 4 + j].reshape((64, 64, 3)), axis=1
                )
        if i == 0:
            ans = result
        else:
            ans = np.append(ans, result, axis=0)

    return ans


# def visualize_model(model):
#     """Model visualization"""
#     return SVG(model_to_dot(model).create(prog="dot", format="svg"))


def train_history(model):
    """Training history visualization"""
    # Plot training & validation accuracy values
    plt.plot(model.history["acc"])
    plt.plot(model.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()

    # Plot training & validation loss values
    plt.plot(model.history["loss"])
    plt.plot(model.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()


model_name = "20200310_0028"
model = load_model("models/" + model_name + ".h5")
# visualize_model(model)
# plot_model(model, to_file="model.png")
# train_history(model)
test_result = test_my_image(model, "cat_1", 4)
plt.imshow(test_result)
plt.show()
