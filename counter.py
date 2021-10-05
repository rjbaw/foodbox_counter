try:
    from utils.basler import *
except:
    print("Failed to import Basler")
try:
    from utils.genicameras import *
except:
    print("Failed to import Harvesters")
import numpy as np
import time
import cv2
import dlib
import torch
import copy
import torchvision
import argparse
from models import *  
from utils.datasets import *
from utils.utils import *
from utils.xml_generator import PascalVocWriter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.spatial import distance as dist
from collections import OrderedDict
from dataclasses import dataclass
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from efficientdet.backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from efficientdet.utils_extra.utils import preprocess, invert_affine, postprocess, preprocess_video
from pyModbusTCP.client import ModbusClient

@dataclass
class object_attr:
    trackers : any
    conf : float
    clsconf : float
    coordinate : tuple
    centroid : tuple
    disappeared : int
    occur : int
    cls_type : int
    counted : bool

class Detector:
    def __init__(self, opt):
        self.out = opt.output
        self.half = opt.half
        self.view_img = opt.view_img
        self.cfg = opt.cfg
        self.conf_thres = opt.conf_thres
        self.count_thres = opt.count_thres
        self.nms_thres = opt.conf_thres
        self.tracker_type = opt.tracker
        self.address = opt.address
        self.slave_id = opt.slave_id
        self.fourcc = opt.fourcc
        self.model_type = opt.model_type
        self.save_img = opt.save
        self.generate_labels = opt.save_labels
        self.device = torch_utils.select_device(opt.device)
        self.classes = load_classes(parse_data_cfg(opt.data)['names'])
        self.clscolors = self.get_colors()
        self.vid_path = None
        self.vid_writer = None
        self.tack = []
        self.fps = 0
        self.totalframes = 0
        self.counter = 0
        self.t0 = time.time()
        # user defined
        self.crop = False
        self.skip_frames = 200 #50 40 200 1
        self.maxdistance = 500
        self.maxdisappear = 3
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.client_start_id = opt.client_shift

    def initialize_folder(self, dirs):
        if os.path.exists(dirs):
            shutil.rmtree(dirs)  
        os.makedirs(dirs)  

    def get_sources(self, source):
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

    def load_model(self, weights):
        model = Darknet(self.cfg, self.img_size)
        if weights.endswith('.pt'): 
            model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:  
            _ = load_darknet_weights(model, weights)
        model.to(self.device).eval()
        self.half = self.half and self.device.type != 'cpu'  
        if self.half:
            model.half()
        return model

    def load_effmodel(self, weights):
        anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        compound_coef = int(weights.split('-')[-1][1])
        model = EfficientDetBackbone(compound_coef=compound_coef,
                                     num_classes=len(self.classes),
                                     ratios=anchor_ratios,
                                     scales=anchor_scales)
        model.load_state_dict(
            torch.load(weights, map_location=self.device)
        )
        #model.requires_grad_(False)
        model.to(self.device).eval()
        self.half = self.half and self.device.type != 'cpu'  
        if self.half:
            model = model.half()
        return model

    def get_colors(self):
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        return colors

    def yolov3_transform(self, im0s):
        if type(im0s) == list:
            img = [letterbox(x, new_shape=self.img_size, interp=cv2.INTER_LINEAR)[0] for x in im0s]
            img = np.stack(img, 0)
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  
            img /= 255.0  
        else:
            img = letterbox(im0s, new_shape=self.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  
            img /= 255.0 
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = torch.from_numpy(img).to(self.device)
        return img

    def write_values(self, slave_address, values):
        c = ModbusClient(host=self.address,
                         port=502,
                         unit_id=self.slave_id,
                         auto_open=True,
                         auto_close=True)
        ret = c.write_single_register(slave_address, values)
        return ret

    def detect(self, path, im0s, vid_cap, save_txt=False):
        def eval_pred(i, det):
            rects = []
            s = '%g: ' % i
            s += '%gx%g ' % img.shape[2:]
            if det != None and len(det):
                det[:,:4] = scale_coords(img.shape[2:], det[:, :4], im0s[i].shape).round()
                rects = det.cpu().numpy()
            self.process_results(rects, im0s[i], path[i], self.tack[i], vid_cap)
            self.tack[i].update(rects, im0s[i])
            print(i, ' ', self.tack[i].count)
            print('%sDone. (%.3fs)' % (s, time.time() - t))
        t = time.time()
        img = self.yolov3_transform(im0s)
        pred = self.model(img)[0]
        if self.half:
            pred = pred.float()
        pred = non_max_suppression(pred, self.conf_thres, self.nms_thres)
        with ThreadPoolExecutor() as executor:
            all( executor.map(eval_pred, range(len(pred)), pred) )

    def efficientdet_transform(self, im0):
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        input_size = input_sizes[int(self.weights.split('-')[-1][1])]
        ori_imgs, framed_imgs, framed_metas = preprocess_video(im0, max_size=input_size)
        if self.device.type != 'cpu':
            img = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            img = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        if self.half:
            img = img.to(torch.float16).permute(0, 3, 1, 2)
        else:
            img = img.to(torch.float32).permute(0, 3, 1, 2)
        return img, framed_metas

    def detecteff(self, im0s, model, save_txt=False):
        g_rects = []
        img, framed_metas = self.efficientdet_transform(im0s)
        features, regression, classification, anchors = model(img)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        pred = postprocess(img, anchors, regression,
                           classification, regressBoxes, clipBoxes,
                           self.conf_thres, self.nms_thres)
        pred = invert_affine(framed_metas, pred)
        for i, det in enumerate(pred):
            s = '%g: ' % i
            s += '%gx%g ' % img.shape[2:]
            if len(det['rois']) != 0:
                for c in np.unique(det['class_ids']):
                    n = (det['class_ids'] == c).sum()
                    s += '%g %ss, ' % (n, self.classes[int(c)])
                for j in range(len(det['rois'])):
                    (x, y, xmax, ymax) = det['rois'][j].astype(np.int)
                    obj = self.classes[det['class_ids'][j]]
                    score = float(det['scores'][j])
                    if save_txt:
                        with open(self.save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (score, obj, score))
                    if self.save_img or self.view_img: 
                        label = '%s %.2f' % (obj, score)
                        plot_one_box((x,y,xmax,ymax), self.im0, label=label,
                                     color=self.clscolors[det['class_ids'][j]])
                    if self.crop:
                        crop_img = self.im0[y:ymax, x:xmax]
                    g_rects.append([x, y, xmax, ymax, score, int(det['class_ids'][j])])
        return g_rects, s

    def put_status_bar(self, im0, tack):
        height, width = im0.shape[:2]
        im0 = cv2.rectangle(im0, (0, 0),
                            (width, int(height/20.0)),
                            (255, 255, 255), -1)
        im0 = cv2.putText(im0, 'Elapsed Time: %.1f s' % ((time.time() - self.t0)),
                          (width - 170, 20),
                          self.font, 0.4, (255, 0, 0), 1)
        im0 = cv2.putText(im0, 'FPS: %.1f' % (self.fps),
                          (width - 112, 40),
                          self.font, 0.4, (255, 0, 0), 1)
        im0 = cv2.putText(im0, str(tack.count),
                          (10, 20), self.font, 0.6, (255, 0, 0), 1)
        return im0

    def save_crop_img(self, crop_imgs):
        if len(crop_imgs) > 1:
            for crop_img in crop_imgs:
                i = 1
                image_dir = os.path.join(self.out, str(i) + '.jpg')
                while os.path.exists(image_dir):
                    i += 1
                    image_dir = os.path.join(self.out, str(i) + '.jpg')
                try:
                    cv2.imwrite(image_dir, crop_img)
                except:
                    pass

    def save_video(self, im0, vid_cap, save_path):
        if self.vid_path != save_path:  
            self.tack[0].reset
            self.vid_path = save_path
            if isinstance(self.vid_writer, cv2.VideoWriter):
                self.vid_writer.release()  
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.vid_writer = cv2.VideoWriter(save_path,
                                              cv2.VideoWriter_fourcc(*self.fourcc), # *'MPEG'
                                              30, (w, h))
        self.vid_writer.write(im0)

    def save_to_out(self, im0, save_path, vid_cap, crop_img=[]):
        if self.dataset.mode == 'images':
            cv2.imwrite(save_path, im0)
        else:
            self.save_video(im0, vid_cap, save_path)
        if self.crop:
            self.save_crop_img(crop_img)

    def process_results(self, rects, im0, path, tack, vid_cap, tracker=False):
        def plot_bbox(rect, im0, save_txt=False):
            if save_txt:
                with open(save_path + '.txt', 'a') as file:
                    file.write(('%g ' * 6 + '\n') % (rect))
            if self.crop:
                crop_imgs.append(
                    im0[int(rect[1] - 200):int(rect[3] + 200),\
                        int(rect[0] - 200):int(rect[2] + 200)]
                )
            if self.save_img or self.view_img:  
                label = '%s %.2f' % (self.classes[int(rect[6])], rect[5])
                try: 
                    im0 = plot_one_box(rect[:4], im0,
                                       label=label, color=self.clscolors[int(rect[6])])
                except:
                    pass
        crop_imgs = []
        save_path = str(Path(self.out) / Path(path).name)
        im0_ori = copy.copy(im0)
        for rect in rects:
            plot_bbox(rect, im0)
        if self.dataset.mode != 'images' or self.view_img:
            im0 = self.put_status_bar(im0, tack)
        if self.view_img:
            h,w = im0.shape[:2]
            cv2.imshow(path, cv2.resize(im0,(int(1080/h*w),1080)))
        if self.save_img:
            self.save_to_out(im0, save_path, vid_cap, crop_imgs)
        if self.generate_labels:
            self.save_xml_labels(im0_ori, save_path, rects, im0.shape)
            if (len(rects) == 0) or (np.sum(rects[:,5]) == 0):
                return

    def save_xml_labels(self, im0, save_path, rects, shape):
        if (len(rects) == 0) or (np.sum(rects[:,5]) == 0):
            return
        height, width, depth = shape
        depth = 3
        imageshape = [height, width, depth]
        if self.dataset.mode != 'images':
            i = 1
            image_dir = os.path.join(self.out, str(i) + '.jpg')
            while os.path.exists(image_dir):
                i += 1
                image_dir = os.path.join(self.out, str(i) + '.jpg')
            try:
                cv2.imwrite(image_dir, im0)
            except:
                pass
            filename = str(i)
            xml_path = os.path.join(self.out, str(i) + '.xml')
        else:
            filename = os.path.basename(save_path)
            xml_path = os.path.join(os.path.dirname(save_path),
                                    filename.split('.')[0] + '.xml')
        writer = PascalVocWriter(self.out, filename, imageshape, localImgPath=xml_path)
        for rect in rects:
            label = self.classes[int(rect[-1])]
            difficult = 0
            writer.addBndBox(rect[0], rect[1], rect[2],
                             rect[3], label, difficult)
        writer.save(targetFile=xml_path)

    def main(self, save_txt = False):
        self.initialize_folder(self.out)
        self.dataset = self.get_sources(opt.source)
        if self.model_type == 'yolov3':
            self.img_size = opt.img_size  
            self.model = self.load_model(opt.weights)
        else:
            self.model = self.load_effmodel()
        for path, im0s, vid_cap in self.dataset:
            t = time.time()
            if len(self.tack) == 0:
                self.tack = [tracking_system() for _ in range(len(im0s))]
            if self.totalframes % self.skip_frames == 0:
                self.rundetect = True
            else:
                self.rundetect = np.any([self.tack[i].rundetect for i in range(len(self.tack))])
            if self.rundetect:
                self.rundetect = False
                if self.model_type == 'yolov3':
                    self.detect(path, im0s, vid_cap)
                else:
                    self.detecteff(im0s, self.model)
            else:
                def track_process(i, im0):
                    rects = self.tack[i].bbox(im0)
                    self.tack[i].update(rects, im0)
                    self.process_results(rects, im0, path[i], self.tack[i], vid_cap)
                with ThreadPoolExecutor() as executor:
                    all( executor.map(track_process, range(len(im0s)), im0s) )
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            if self.fps == 0:
                self.fps = 1/(time.time() - t)
            else:
                self.fps += (1 * 10 ** -1)*(1/(time.time() - t) - self.fps)
            self.totalframes +=1
        if save_txt or self.save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + self.out)
        print("elapsed time: {:.2f}".format(time.time() - self.t0))
        print("approx. FPS: {:.2F}".format(self.fps))

class tracking_system(Detector):
    def __init__(self):
        super().__init__(opt)
        self._count = {}
        self._rundetect = False
        self.objects = OrderedDict()
        self.objectID = 0
        if self.tracker_type.startswith('siam'):
            self.init_siam()

    def init_siam(self):
        cfg.merge_from_file(os.path.join('tracker', self.tracker_type, 'config.yaml'))
        cfg.CUDA = torch.cuda.is_available()
        self.model_tracker = ModelBuilder()
        self.model_tracker.load_state_dict(
            torch.load('tracker/' + str(self.tracker_type) + '/model.pth',
                        map_location=lambda storage, loc: storage.cpu()))
        self.model_tracker.eval().to(self.device)
        #siam_tracker = build_tracker(model_tracker)

    @property
    def count(self):
        return self._count

    @property
    def rundetect(self):
        result = self._rundetect
        self._rundetect = False
        return result

    @property
    def reset(self):
        self._count = {}

    def bbox(self, im0):
        def siam_bbox(objectID, im0, poly=False):
            outputs = self.objects[objectID].trackers.track(im0)
            cls = self.objects[objectID].cls_type
            if outputs['best_score'] < 0.7:
                self._rundetect = True
                #return 
            if poly:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                im0 = cv2.polylines(im0, [polygon.reshape((-1, 1, 2))],
                                    True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                #frame = cv2.addWeighted(im0, 0.77, mask, 0.23, -1)
                return mask
            else:
                bbox = list(map(int, outputs['bbox']))
                x, y, xmax, ymax = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
                return [x, y, xmax, ymax, 0, 0, cls]
        def dlib_bbox(objectID):
            score = self.objects[objectID].trackers.update(im0)
            cls = self.objects[objectID].cls_type
            if score < 10:
                self._rundetect = True
            pos = self.objects[objectID].trackers.get_position()
            x, y, xmax, ymax = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
            return [x, y, xmax, ymax, 0, 0, cls]
        def cv_bbox(objectID):
            ok, bbox = self.objects[objectID].trackers.update(im0)
            cls = self.objects[objectID].cls_type
            if not ok:
                self._rundetect = True
            x, y, xmax, ymax = bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]
            return [x, y, xmax, ymax, 0, 0, cls]
        if self.tracker_type.startswith("siam"):
            bbox_func = siam_bbox
            rects = []
            for objectID in self.objects.keys():
                out = bbox_func(objectID, im0) 
                if out is None:
                    continue
                else:
                    rects.append(out)
        else:
            if self.tracker_type == 'dlib':
                bbox_func = dlib_bbox
            elif self.tracker_type.startswith('cv'):
                bbox_func = cv_bbox
            rects = []
            for objectID in self.objects.keys():
                out = bbox_func(objectID) 
                if out is None:
                    continue
                else:
                    rects.append(out)
        return np.asarray(rects)

    def register(self, rects, im0):
        def assign_values(rect):
            x, y, xmax, ymax, conf, clsconf, cls = rect
            centroid = ( (x + xmax)/2.0, (y + ymax)/2.0 )
            coordinate = (x, y, xmax, ymax)
            if self.classes[int(cls)] not in self._count:
                self._count[self.classes[int(cls)]] = 0
            if conf > self.count_thres:
                self._count[self.classes[int(cls)]] += 1
                counted = True
                self.write_values(int(cls)+self.client_start_id,
                                  self._count[self.classes[int(cls)]])
            else:
                counted = False
            self.objects[self.objectID] = object_attr(
                trackers = self.track_objects(im0, boxes = coordinate),
                centroid = centroid,
                coordinate = coordinate,
                clsconf = clsconf,
                conf = conf,
                cls_type = cls,
                counted = counted,
                occur = 0, 
                disappeared = 0)
            self.objectID += 1
        with ThreadPoolExecutor() as executor:
            all(executor.map(assign_values, rects))

    def all_disappear(self):
        self._rundetect = True
        def miss_step(objectID):
            self.objects[objectID].disappeared += 1
            if self.objects[objectID].disappeared > self.maxdisappear:
                del self.objects[objectID]
        with ThreadPoolExecutor() as executor:
            all(executor.map(miss_step, list(self.objects.keys())))

    def sort_and_match(self, rects, im0):
        centroids = np.zeros((len(rects), 2), dtype=int)
        #centroids = torch.zeros(len(rects),2)
        centroids[:,0] = np.round((rects[:,0] + rects[:,2])/2.0)
        centroids[:,1] = np.round((rects[:,1] + rects[:,3])/2.0)
        usedRows = set()
        usedCols = set()
        registeredID = list(self.objects.keys())
        registered_centroids = np.vstack([obj.centroid for obj in self.objects.values()])
        euclid_distance = dist.cdist(registered_centroids, centroids)
        rows = euclid_distance.min(axis = 1).argsort()
        cols = euclid_distance.argmin(axis = 1)[rows]
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if euclid_distance[row, col] > self.maxdistance:
                continue
            objectID = registeredID[row]
            self.objects[objectID].centroid = centroids[col]
            self.objects[objectID].coordinate = rects[col,:4]
            if (rects[col,5] > 0.0):
                self.objects[objectID].cls_type = rects[col,6]
                self.objects[objectID].disappeared = 0
                self.objects[objectID].trackers = self.track_objects(im0, boxes = rects[col,:4])
                self.objects[objectID].conf = rects[col,4]
                self.objects[objectID].clsconf = rects[col,5]
                self.objects[objectID].occur += 1
            if (rects[col,5] > self.count_thres) and (self.objects[objectID].occur > 1):
                if (self.classes[int(rects[col,6])] not in self._count):
                    self._count[self.classes[int(rects[col,6])]] = 0
                if (self.objects[objectID].counted == False):
                    self._count[self.classes[int(rects[col,6])]] += 1
                    self.objects[objectID].counted = True
                    self.write_values(self.client_start_id+int(rects[col,6]),
                                      self._count[self.classes[int(rects[col,6])]])
            usedRows.add(row)
            usedCols.add(col)
        unusedRows = set(range(0, euclid_distance.shape[0])).difference(usedRows)
        unusedCols = set(range(0, euclid_distance.shape[1])).difference(usedCols)
        if np.sum(rects[:,5]) == 0:
            return
        for row in unusedRows:
            #self._rundetect = True
            objectID = registeredID[row]
            self.objects[objectID].disappeared += 1
            if self.objects[objectID].disappeared > self.maxdisappear:
                del self.objects[objectID]
        if len(unusedCols) != 0:
            #self._rundetect = True
            self.register(rects[list(unusedCols)], im0)

    def update(self, rects, im0):
        if len(rects) == 0:
            self.all_disappear()
            return 
        if len(self.objects) == 0:
            self.register(rects, im0)
        else:
            self.sort_and_match(rects, im0)

    def track_objects(self, im0, boxes):
        x, y, xmax, ymax = boxes
        if self.tracker_type.startswith('siam'):
            tracker = build_tracker(self.model_tracker)
            #boxes = (x, y, xmax-x, ymax-y)
            boxes = torch.as_tensor([x, y, xmax-x, ymax-y])
            tracker.init(im0, boxes)
        elif self.tracker_type == 'dlib':
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x, y, xmax, ymax)
            tracker.start_track(im0, rect)
        if self.tracker_type.startswith('cv'):
            if self.tracker_type == 'cvboosting':
                tracker = cv2.TrackerBoosting_create()
            elif self.tracker_type == 'cvmil':
                tracker = cv2.TrackerMIL_create()
            elif self.tracker_type == 'cvkcf':
                tracker = cv2.TrackerKCF_create()
            elif self.tracker_type == 'cvtld':
                tracker = cv2.TrackerTLD_create()
            elif self.tracker_type == 'cvmedianflow':
                tracker = cv2.TrackerMedianFlow_create()
            elif self.tracker_type == 'cvcsrt':
                tracker = cv2.TrackerCSRT_create()
            elif self.tracker_type == 'cvmosse':
                tracker = cv2.TrackerMOSSE_create()
            tracker.init(im0, (x, y, xmax-x, ymax-y))
        return tracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg',
                        help='cfg file path')
    parser.add_argument('--data', type=str, default='data/nongjok.data',
                        help='data file path')
    parser.add_argument('--weights', type=str, default='weights/nongjok.pt',
                        help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples',
                        help='image sources')
    parser.add_argument('--model-type', type=str, default='yolov3',
                        help='model type for inference')
    parser.add_argument('--output', type=str, default='output',
                        help='results folder')
    parser.add_argument('--img-size', type=int, default=416,
                        help='inference square length (pixels)') # (320, 192) or (416, 256) or (608, 352) for (height, width)
    parser.add_argument('--client-shift', type=int, default=0,
                        help='client shift')
    parser.add_argument('--slave-id', type=int, default=1,
                        help='modbus server slave id')
    parser.add_argument('--address', type=str, default='127.0.0.1',
                        help='modbus server TCP IP address')
    parser.add_argument('--conf-thres', type=float, default=0.3,
                        help='confidence threshold')
    parser.add_argument('--count-thres', type=float, default=0.8,
                        help='count confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='video output codec')
    parser.add_argument('--device', default='',
                        help='device id e.g. 0 or 0,1 or cpu')
    parser.add_argument('--half', action='store_true',
                        help='use half precision (FP16)')
    parser.add_argument('--view-img', action='store_false',
                        help='display results')
    parser.add_argument('--save', action='store_true',
                        help='save output')
    parser.add_argument('--save-labels', action='store_true',
                        help='save labels')
    parser.add_argument('--tracker', default='dlib', type=str,
                        choices=['siammask', 'dlib', 'siammask_e',
                                 'siamrpn_alex', 'siamrpn_alex_otb', 'siamrpn_mobilev2',
                                 'siamrpn_r50', 'siamrpn_r50_lt', 'siamrpn_r50_otb',
                                 'cvboosting', 'cvmil' , 'cvkcf',
                                 'cvcsrt', 'cvmedianflow', 'cvtld', 'cvmosse'],
                        help='siammask or centroid')
    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        Detector(opt).main()
