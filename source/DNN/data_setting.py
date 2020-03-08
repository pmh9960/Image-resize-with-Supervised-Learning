import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_dataset():
    dataset = h5py.File("datasets/dataset_300cats.hdf5", "r")
    train_set_images128 = np.array(dataset["train/images128"][:])
    train_set_images256 = np.array(dataset["train/images256"][:])
    train_set_images512 = np.array(dataset["train/images512"][:])

    # dev_set_images128 = np.array(dataset["dev/images128"][:])
    # dev_set_images256 = np.array(dataset["dev/images256"][:])
    # dev_set_images512 = np.array(dataset["dev/images512"][:])

    # test_set_images128 = np.array(dataset["test/images128"][:])
    # test_set_images256 = np.array(dataset["test/images256"][:])
    # test_set_images512 = np.array(dataset["test/images512"][:])

    # dataset = (
    #     train_set_images128,
    #     train_set_images256,
    #     train_set_images512,
    #     dev_set_images128,
    #     dev_set_images256,
    #     dev_set_images512,
    #     test_set_images128,
    #     test_set_images256,
    #     test_set_images512,
    # )
    # return dataset

    return train_set_images128, train_set_images256, train_set_images512


# # Example of a picture
# train_set_images128, train_set_images256, train_set_images512 = load_dataset()
# index = 0
# plt.imshow(train_set_images512[index])
# plt.show()
