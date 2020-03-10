import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_dataset(filename):
    dataset = h5py.File("datasets/" + filename + ".hdf5", "r")
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


def flatting_images(train_set_X, train_set_Y):
    # print(train_set_X.shape, train_set_Y.shape)
    num_piece = train_set_X.shape[0]
    m_train = train_set_X.shape[1]
    num_px_X = train_set_X.shape[2]
    num_px_Y = train_set_Y.shape[2]
    # for num in num_piece:
    train_set_X_flatten = train_set_X[0].reshape((m_train, -1))
    train_set_Y_flatten = train_set_Y[0].reshape((m_train, -1))
    for i in range(num_piece - 1):
        train_set_X_flatten = np.r_[
            train_set_X_flatten, train_set_X[i + 1].reshape((m_train, -1))
        ]
        train_set_Y_flatten = np.r_[
            train_set_Y_flatten, train_set_Y[i + 1].reshape((m_train, -1))
        ]

    return train_set_X_flatten, train_set_Y_flatten


def slicing_images(num_slice, train_set_X_orig, train_set_Y_orig):
    lengthX = int(train_set_X_orig.shape[1] / num_slice)
    lengthY = int(train_set_Y_orig.shape[1] / num_slice)

    train_set_X = []
    for i in range(num_slice):
        for j in range(num_slice):
            piece = train_set_X_orig[
                :, lengthX * i : lengthX * (i + 1), lengthX * j : lengthX * (j + 1), :,
            ]
            train_set_X.append(piece)

    train_set_Y = []
    for i in range(num_slice):
        for j in range(num_slice):
            piece = train_set_Y_orig[
                :, lengthY * i : lengthY * (i + 1), lengthY * j : lengthY * (j + 1), :,
            ]
            train_set_Y.append(piece)

    train_set_X = np.array(train_set_X)
    train_set_Y = np.array(train_set_Y)

    return train_set_X, train_set_Y


def save_history(history, current_time_str):
    # Plot training & validation accuracy values
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.savefig("models/" + current_time_str + "/history_accuracy.png")
    plt.clf()
    
    # Plot training & validation loss values
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.savefig("models/" + current_time_str + "/history_loss.png")

