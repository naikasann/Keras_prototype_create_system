import cv2
import os

def save_all_frames(video_path, dir_path, savename, ext="jpg"):
    # Read the video.
    cap = cv2.VideoCapture(video_path)

    # if not open.(not file exsist)
    if not cap.isOpened():
        print("cant open.")
        return None
    # makefolder
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, savename)

    # take movie frame.
    frames = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    
    while True:
        # Get Frame.
        ret, frame = cap.read()
        if ret:
            # Create and save the image.
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(frames), ext), frame)
            n += 1
        else:
            return None

# Specify the video file to be read.
moviepath = '.mp4'
# Name of the folder to save.
savefolder = './'
# File name to save
savefile = ""

save_all_frames(moviepath, savefolder, savefile)