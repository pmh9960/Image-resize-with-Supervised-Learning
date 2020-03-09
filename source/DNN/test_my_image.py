import numpy as np
import tensorflow as tf
from data_setting import load_dataset, slicing_images, flatting_images


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
