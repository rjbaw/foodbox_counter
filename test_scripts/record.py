import os
import time
import numpy as np
from utils.datasets import *
try:
    from utils.basler import *
except:
    print('import failed')

def get_sources(source):
    keywords = ['rtsp', 'http', '.txt', 'v4l2src', 'basler', 'geni']
    try:
        int(source)
        webcam = True
    except Exception as e:
        webcam = np.any([True if (key in source) else False for key in keywords])
    if webcam:
        torch.backends.cudnn.benchmark = True  
        if (source.startswith('bas')):
            dataset = BaslerCamera(source, img_size=None)
            #dataset = BaslerCameraThread(source, img_size=None)
        elif (source.startswith('geni')):
            dataset = genicameras(source, img_size=None)
        else:
            dataset = LiveFeed(source)
    else:
        dataset = MediaFiles(source)
    return dataset

def next_video(save_name, save_folder):
    save_path = os.path.join(save_folder, str(save_name) + '.mp4')
    while os.path.exists(save_path):
        save_name += 1
        save_path = os.path.join(save_folder, str(save_name) + '.mp4')
    return save_name, save_path

def record_video():
    source = 'basler'
    #source = 'sample60.mp4'
    save_folder = 'save_video'
    time_interval = 60*10
    crop = False
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    dataset = get_sources(source)
    t0 = time.time()
    save_name = 1
    save_name, save_path = next_video(save_name, save_folder)
    vid_path = None
    for path, im0s, vid_cap in dataset:
        h, w = im0s[0].shape[:2]
        r = 1080/h
        #h = int(1080/h * 1000)
        #w = int(1920/w * 1000)
        #w = int(round(r*w, -2))
        #im0 = cv2.resize(im0s[0],(w,h),
        #                 interpolation=cv2.INTER_AREA)
        im0 = im0s[0]
        #cv2.imshow("", im0)
        if vid_path != save_path:  
            vid_path = save_path
            #fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(save_path,
                                         cv2.VideoWriter_fourcc(*'MPEG'),
                                         30, (w, h))
        if (time.time()-t0) > time_interval:
            save_name, save_path = next_video(save_name, save_folder)
            vid_writer.release()  
            t0 = time.time()
        vid_writer.write(im0)

if __name__ ==  '__main__':
    record_video()
