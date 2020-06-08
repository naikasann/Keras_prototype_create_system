import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

import numpy as np
import cv2
import sys
import os
import h5py

model_path = "test_10.h5"
inputshape = (227, 227)
category_list = ["goo", "choki", "paa"]
color_list = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

windowsize = (1280, 720)
cameara_id = 1

count = 0
max_count = 10
fps = 0

# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(cameara_id)
if not capture.isOpened:
    print("Camera not found! exit program")
    sys.exit()

model = load_model(model_path)
model.summary()

# for FPS measure
tm = cv2.TickMeter()
tm.start()

while(True):
    ret, frame = capture.read()
    os.system("cls")
    # resize the window(show movie size)
    frame = cv2.resize(frame, windowsize)

    predictframae = cv2.resize(frame, inputshape)
    predictframae = cv2.cvtColor(predictframae, cv2.COLOR_BGR2RGB)
    predictframae = np.asarray([predictframae], dtype=np.float32)
    predictframae /= 255.0
    prd = model.predict(predictframae, batch_size=1)

    # measure fps.
    if count == max_count:
        tm.stop()
        fps = max_count / tm.getTimeSec()
        tm.reset()
        tm.start()
        count = 0
    count += 1

    cv2.putText(frame, 'FPS: {:.2f}'.format(fps),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

    for c, (category, p) in enumerate(zip(category_list, prd[0])):
        cv2.putText(frame, "{} : {}".format(category, p),
                    (10 , 60 + (c * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_list[c], thickness=2)

    # terminal write information
    print("======= movie info =======")
    print("FPS : ", fps)
    print(prd)
    print("==========================")

    cv2.imshow("KerasFramework_predicttest",frame)
    # end key setting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()