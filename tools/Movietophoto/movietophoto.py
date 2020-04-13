import cv2
import os

def save_all_frames(video_path, dir_path, savename, ext="jpg"):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("cant open.")
        return
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, savename)

    frames = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    
    while True:
        ret, frame = cap.read()
        if ret:
            print(frame)
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(frames), ext), frame)
            n += 1
        else:
            return

save_all_frames('./paa.mp4', './paa', 'paa')