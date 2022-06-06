import os
import cv2
from display import Display

W = 1920//2
H = 1080//2

DIR_VIDEOS = os.path.join('.\\', 'videos')

display = Display(W, H)

def process_frame(img):
    img = cv2.resize(img, (W, H))
    display.paint(img)

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
