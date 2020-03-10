import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_setting import load_dataset, slicing_images, flatting_images
from keras.models import model_from_json
from keras import optimizers
import keras
from keras.utils.vis_utils import plot_model


def predict_my_image(model, filename, num_slice):
    test128_orig, test256_orig, test512_orig = load_dataset(filename)
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
    plt.savefig("models/" + folder_name + "/cat1_predict.png")
    plt.imshow(test256_orig.reshape(256, 256, 3))
    plt.savefig("models/" + folder_name + "/cat1_orig.png")


def load_model(folder_name):
    json_file = open("models/" + folder_name + "/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("models/" + folder_name + "/weight.h5")

    return loaded_model


folder_name = "20200310_1317"
loaded_model = load_model(folder_name)


# # loaded_model.compile(
# #     loss="mean_squared_error",
# #     optimizer=optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, decay=0.0),
# #     metrics=["accuracy"],
# # )

predict_my_image(loaded_model, "cat_1", 32)
