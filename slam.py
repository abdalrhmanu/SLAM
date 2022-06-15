import imp
import os
import cv2
import numpy as np

from display import Display
from extractor import Extractor

W = 1920//2
H = 1080//2
F = 270 #740
K = np.array(([F,0,W//2], [0,F,H//2], [0, 0, 1]))

DIR_VIDEOS = os.path.join('.\\', 'videos')
        
disp = Display(W, H)
fe = Extractor(K)

def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches, pose = fe.extract(img)

    if pose is None:
        return


    print("%d matches" % (len(matches)))
    print(pose)

    for pt1, pt2 in matches:
        u1,v1 = fe.denormalize(pt1)
        u2,v2 = fe.denormalize(pt2)

        cv2.circle(img, (u1,v1), color=(0, 255,0), radius=3)
        cv2.line(img, (u1,v1), (u2,v2), color=(255, 0,0))
    

    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture(os.path.join(DIR_VIDEOS, 'test_countryroad.mp4'))

    while cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            process_frame(frame)

            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        else:
            break

    # Release everything if job is finished
    # cap.release()
    # cv2.destroyAllWindows()
