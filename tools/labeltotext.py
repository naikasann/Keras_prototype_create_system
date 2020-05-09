import os
import cv2
from tqdm import tqdm

path = ""
textfile = ".txt"

f = open(textfile, "w")

### [画像パス] [ラベルつけ]を行う。
"""filedir = os.listdir(path)
for cnt, labels in enumerate(filedir):
    labels_path = os.path.join(path,labels)
    images = os.listdir(labels_path)
    for imagepath in images:
        imagepath = os.path.join(labels_path, imagepath)

        string = imagepath + " " + str(cnt) + "\n"
        f.writelines(string)"""

# 指定したフォルダーの内の画像をすべてテキストに書き込む
"""
filedir = os.listdir(path)
print("File count :", len(filedir))
for imagepath in tqdm(filedir):
    imagepath = os.path.join(path, imagepath)

    string = imagepath + "\n"
    f.writelines(string)
"""