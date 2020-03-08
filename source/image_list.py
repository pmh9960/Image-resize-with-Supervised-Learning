import os

file_list = os.listdir("sample_images")
print(file_list)

file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
print(file_list_jpg)
