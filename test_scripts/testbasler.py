from utils.datasets import BaslerCamera
import cv2

#img_size = (2000, 2440)
img_size = None
half=False
dataset = BaslerCamera(0, img_size=img_size, half=half)
for path, im0s, vid_cap in dataset:
    im0s = im0s[0]
    print(im0s.shape)
    cv2.namedWindow('title', cv2.WINDOW_NORMAL)
    cv2.imshow('title', im0s)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
