import requests
import os
from bs4 import BeautifulSoup

file_list = os.listdir("cat_html")
file_list_html = [file for file in file_list if file.endswith(".html")]


def extract_image_src(result):
    a = result.find("a")
    img = a.find("img")
    src = img.get("src")
    return src


def extract_pixabay():
    images = []
    for html_name in file_list_html:
        path = "cat_html/" + html_name
        html = open(path, "r", encoding="UTF8")
        soup = BeautifulSoup(html, "html.parser")
        results = soup.find_all("div", {"class": "item"})
        for result in results:
            image_src = extract_image_src(result)
            images.append(image_src)

    return images

