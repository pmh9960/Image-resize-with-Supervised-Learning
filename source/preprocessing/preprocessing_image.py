import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
import os

file_list = os.listdir("sample_images")
file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
# print(file_list_jpg)

num_pxs = (128, 256, 512)

for image_name in file_list_jpg:
    for num_px in num_pxs:
        fname = "sample_images/" + image_name

        # pip install scipy==1.1.0 (need)
        image = Image.open(fname)
        image = image.convert("RGB")  # OSError: cannot write mode LA as JPEG

        margin = int(abs(image.size[0] - image.size[1]) / 2)
        crop_image = (
            image.crop((margin, 0, image.size[0] - margin, image.size[1]))
            if image.size[0] > image.size[1]
            else image.crop((0, margin, image.size[0], image.size[1] - margin))
        )

        reshaped_image = crop_image.resize((num_px, num_px))

        sliced_name = image_name[:-4]
        save_path = "preprocessed_images/" + sliced_name + "_" + str(num_px) + ".jpg"
        reshaped_image.save(save_path)

