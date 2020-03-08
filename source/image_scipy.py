import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


def crop_center(image):
    width = image.shape[1]
    height = image.shape[0]
    center_x = int(width / 2)
    center_y = int(height / 2)
    length = int(min(width, height) / 2)

    return image[
        center_y - length : center_y + length, center_x - length : center_x + length
    ]


num_px = 64

my_image = "cat_2.jpg"
fname = "sample_images/" + my_image

# pip install scipy==1.1.0 (need)
image = np.array(ndimage.imread(fname, flatten=False))
image = image / 255.0

crop_image = crop_center(image)

my_image = scipy.misc.imresize(
    crop_image, size=(num_px, num_px)
)  # scipy.misc (pip install pillow)

plt.imshow(my_image)
plt.show()

