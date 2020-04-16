import os
import cv2

path = "./val_dataset"
textfile = "val_dataset.txt"

f = open(textfile, "w")

filedir = os.listdir(path)
for cnt, labels in enumerate(filedir):
    labels_path = os.path.join(path,labels)
    images = os.listdir(labels_path)
    for imagepath in images:
        imagepath = os.path.join(labels_path, imagepath)

        string = imagepath + " " + str(cnt) + "\n"
        f.writelines(string)