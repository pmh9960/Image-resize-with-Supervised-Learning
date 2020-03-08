import numpy as np
import scipy
from PIL import Image
from scipy import ndimage


num_pxs = (128, 256, 512)

for num_px in num_pxs:
    image_name = "cat_1.jpg"
    fname = "sample_images/" + image_name

    # pip install scipy==1.1.0 (need)
    image = Image.open(fname)

    margin = int(abs(image.size[0] - image.size[1]) / 2)
    crop_image = (
        image.crop((margin, 0, image.size[0] - margin, image.size[1]))
        if image.size[0] > image.size[1]
        else image.crop((0, margin, image.size[0], image.size[1] - margin))
    )

    reshaped_image = crop_image.resize((num_px, num_px))

    save_path = "preprocessed_images/" + image_name + "_" + str(num_px) + ".jpg"
    reshaped_image.save(save_path)

