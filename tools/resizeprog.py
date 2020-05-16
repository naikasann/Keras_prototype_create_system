import os
import cv2
from tqdm import tqdm

# set folder
path = ""
# set imagepath.
size = (000, 000)

# Resize all images.
"""
filedir = os.listdir(path)
for labels in filedir:
    labels_path = os.path.join(path,labels)
    images = os.listdir(labels_path)
    for imagepath in images:
        imagepath = os.path.join(labels_path, imagepath)

        img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
        new_img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(imagepath, new_img)
"""

# [image_path]
"""
filedir = os.listdir(path)
print("File count :", len(filedir))
for imagepath in tqdm(filedir):
    imagepath = os.path.join(path, imagepath)

    img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
    new_img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
    cv2.imwrite(imagepath, new_img)
"""