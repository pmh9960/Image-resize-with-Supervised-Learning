from keras.utils.io_utils import HDF5Matrix

n_epoch = 1000
batch_size = 32
split_pos = 800

x_data = HDF5Matrix(filename, "image")  # 위에서 생성한 HDF5 파일의 image 경로의 데이터를 가져오게 된다.
x_train = HDF5Matrix(
    filename, "image", end=split_pos
)  # HDF5 파일의 데이터 중 일부만 가져오는 것도 가능하다.
x_test = HDF5Matrix(filename, "image", start=split_pos)
y_train = HDF5Matrix(filename, "label", end=split_pos)
y_test = HDF5Matrix(filename, "label", start=split_pos)

# 이미 .compile() 이 된 모델이라고 가정하자
model.fit(
    x_train,
    y_train,
    epochs=n_epoch,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
)  # numpy array 를 쓰듯이 사용하면 된다.
