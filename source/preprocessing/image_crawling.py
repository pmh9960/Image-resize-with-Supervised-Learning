import requests
import urllib.request
from pixabay import extract_pixabay

# max_indeed_page = indeed.extract_indeed_pages()
# indeed_jobs = indeed.extract_indeed_jobs(max_indeed_page)
# save_to_file(indeed_jobs)

opener = urllib.request.build_opener()
opener.addheaders = [
    (
        "User-Agent",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36",
    )
]
urllib.request.install_opener(opener)

number = 1
sources = extract_pixabay()
length = len(sources)
for img_url in sources:
    urllib.request.urlretrieve(img_url, "sample_images/cat_" + str(number) + ".jpg")
    number = number + 1
    if number % 10 == 0:
        print(float(number / length) * 100, "%")
print(f"Number of images : {number - 1}")
