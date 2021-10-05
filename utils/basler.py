from pypylon import pylon
from pypylon import genicam
from threading import Thread
import cv2
import os
import numpy as np

class BaslerCameraThread:
    def __init__(self,
                 sources='basler',
                 img_size=None,
                 grabstrat='latest',
                 debug=False,
                 load=True):
        self.mode = 'images'
        self.grabstrat = grabstrat
        self.debug = debug
        buffer_type = 'custom_value'
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        try:
            self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            assert len(self.devices) > 0, "No Basler Camera found"
            self.imgs = [None] * len(self.devices)
            self.sources = ['Basler #{0}'.format(i) for (i,device) in enumerate(self.devices)]
            self.cap = pylon.InstantCameraArray(len(self.devices))
            for i, cam in enumerate(self.cap):
                cam.Attach(pylon.TlFactory.GetInstance().CreateDevice(self.devices[i]))
                print("Using device ", cam.GetDeviceInfo().GetModelName())
                cam.Open()
                if load:
                    nodeFile = "NodeMap" + str(i) + ".pfs"
                    if os.path.exists(nodeFile):
                        pylon.FeaturePersistence.Load(nodeFile, cam.GetNodeMap(), True)
                    else:
                        pylon.FeaturePersistence.Save(nodeFile, cam.GetNodeMap())
                if img_size is not None:
                    height, width = img_size
                    cam.Width.SetValue(width)
                    cam.Height.SetValue(height)
                if self.grabstrat == 'latest':
                    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
                elif self.grabstrat == 'buffer':
                    if buffer_type == 'latest':
                        cam.OutputQueueSize = 1
                    elif buffer_type == 'onebyone':
                        cam.OutputQueueSize = cam.MaxNumBuffer.Value
                    elif buffer_type == 'custom_value':
                        cam.OutputQueueSize = 2
                    cam.cap.StartGrabbing(pylon.GrabStrategy_LatestImages)
                elif self.grabstrat == 'onebyone':
                    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
                while type(None)==type(self.imgs[i]):
                    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_Return)
                    while not grabResult.IsValid():
                        grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_Return)
                    if grabResult.GrabSucceeded():
                        image = self.converter.Convert(grabResult)
                        im0 = image.GetArray()
                        self.imgs[i] = im0
                    else:
                        print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                    if self.debug:
                        if grabResult.GetNumberOfSkippedImages():
                            print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
                        cameraContextValue = grabResult.GetCameraContext()
                        print("Camera ", cameraContextValue, ": ",
                              self.cap[cameraContextValue].GetDeviceInfo().GetModelName())
                    grabResult.Release()
                thread = Thread(target=self.update, args=([i, cam]),daemon=True)
                thread.start()
        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())
    def update(self, index, cam):
#        if self.cap.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
#            self.cap.ExecuteSoftwareTrigger()
        while cam.IsGrabbing():
            #grabResult = cam.RetrieveResult(0, pylon.TimeoutHandling_Return)
            grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            while not grabResult.IsValid():
                #grabResult = cam.RetrieveResult(0, pylon.TimeoutHandling_Return)
                grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img0 = image.GetArray()
                self.imgs[index] = img0
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            if self.debug:
                if grabResult.GetNumberOfSkippedImages():
                    print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
                cameraContextValue = grabResult.GetCameraContext()
                print("Camera ", cameraContextValue, ": ",
                    self.cap[index].GetDeviceInfo().GetModelName())
            grabResult.Release()
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        im0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.StopGrabbing()
            self.cap.Close()
            cv2.destroyAllWindows()
            raise StopIteration
        return self.sources, im0, None
    def __len__(self):
        return 0

class BaslerCamera:
    def __init__(self,
                 sources='basler',
                 img_size=None,
                 grabstrat='latest',
                 debug=False,
                 thread=False,
                 load=True):
        self.mode = 'images'
        self.grabstrat = grabstrat
        self.debug = debug
        self.thread = thread
        buffer_type = 'custom_value'
        try:
            self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            assert len(self.devices) > 0, "No Basler Camera found"
            self.imgs = [None] * len(self.devices)
            self.sources = ['Basler #{0}'.format(i) for (i,device) in enumerate(self.devices)]
            self.cap = pylon.InstantCameraArray(len(self.devices))
            for i, cam in enumerate(self.cap):
                cam.Attach(pylon.TlFactory.GetInstance().CreateDevice(self.devices[i]))
                print("Using device ", cam.GetDeviceInfo().GetModelName())
                cam.Open()
                if load:
                    nodeFile = "NodeMap" + str(i) + ".pfs"
                    if os.path.exists(nodeFile):
                        pylon.FeaturePersistence.Load(nodeFile, cam.GetNodeMap(), True)
                    else:
                        pylon.FeaturePersistence.Save(nodeFile, cam.GetNodeMap())
                if img_size != None:
                    height, width = img_size
                    cam.Width.SetValue(width)
                    cam.Height.SetValue(height)
            if self.grabstrat == 'latest':
                self.cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
            elif self.grabstrat == 'buffer':
                if buffer_type == 'latest':
                    self.cap.OutputQueueSize = 1
                elif buffer_type == 'onebyone':
                    self.cap.OutputQueueSize = self.cap.MaxNumBuffer.Value
                elif buffer_type == 'custom_value':
                    self.cap.OutputQueueSize = 2
                self.cap.StartGrabbing(pylon.GrabStrategy_LatestImages)
            elif self.grabstrat == 'onebyone':
                self.cap.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())
        self.grab()
        if self.thread:
            thread = Thread(target=self.update, args=([]), daemon=True)
            thread.start()
    def grab(self):
        while np.any([type(None)==type(im) for im in self.imgs]):
            grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            while not grabResult.IsValid():
                grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                im0 = image.GetArray()
                cameraContextValue = grabResult.GetCameraContext()
                self.imgs[cameraContextValue] = im0
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            if self.debug:
                if grabResult.GetNumberOfSkippedImages():
                    print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
                print("Camera ", cameraContextValue, ": ",
                        self.cap[cameraContextValue].GetDeviceInfo().GetModelName())
            grabResult.Release()
    def update(self):
#        if self.cap.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
#            self.cap.ExecuteSoftwareTrigger()
        while self.cap.IsGrabbing():
            grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            #grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
            while not grabResult.IsValid():
                #grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
                grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                cameraContextValue = grabResult.GetCameraContext()
                im0 = image.GetArray()
                self.imgs[cameraContextValue] = im0
                if self.debug:
                    if grabResult.GetNumberOfSkippedImages():
                        print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
                    print("Camera ", cameraContextValue, ": ",
                            self.cap[cameraContextValue].GetDeviceInfo().GetModelName())
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        assert self.cap.IsGrabbing(), "Camera pipe broken"
        if cv2.waitKey(1) == ord('q'): 
            self.cap.StopGrabbing()
            self.cap.Close()
            cv2.destroyAllWindows()
            raise StopIteration
        if not self.thread:
            grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            #grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
            while not grabResult.IsValid():
                #grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
                grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                cameraContextValue = grabResult.GetCameraContext()
                im0 = image.GetArray()
                self.imgs[cameraContextValue] = im0
                if self.debug:
                    if grabResult.GetNumberOfSkippedImages():
                        print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
                    print("Camera ", cameraContextValue, ": ",
                            self.cap[cameraContextValue].GetDeviceInfo().GetModelName())
            else:
                print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
            grabResult.Release()
        return self.sources, self.imgs, None
    def __len__(self):
        return 0

class SingleBaslerCamera:
    def __init__(self, pipe=0, img_size=None, load=True):
#        self.img_size = img_size
        self.sources = ['Basler']
        self.mode = 'images'
        self.grabstrat = 'latest'
#        self.grabstrat = 'buffer'
        buffer_type = 'custom_value'
        try:
            self.devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            if pipe == 0:
                pipe = pylon.TlFactory.GetInstance().CreateFirstDevice()
            self.cap = pylon.InstantCamera(pipe)  
            self.cap.Open()
            print("Using device: ", self.cap.GetDeviceInfo().GetModelName())
            if load:
                nodeFile = "NodeMap.pfs"
                if os.path.exists(nodeFile):
                    pylon.FeaturePersistence.Load(nodeFile, self.cap.GetNodeMap(), True)
                else:
                    pylon.FeaturePersistence.Save(nodeFile, self.cap.GetNodeMap())
            if img_size is not None:
                height, width = img_size
                self.cap.Width.SetValue(width)
                self.cap.Height.SetValue(height)
#                self.cap.MaxNumBuffer = 15 # default 10
#                self.cap.StartGrabbingMax(100)
            if self.grabstrat == 'latest':
                self.cap.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
            elif self.grabstrat == 'buffer':
                if buffer_type == 'latest':
                    self.cap.OutputQueueSize = 1
                elif buffer_type == 'onebyone':
                    self.cap.OutputQueueSize = self.cap.MaxNumBuffer.Value
                elif buffer_type == 'custom_value':
                    self.cap.OutputQueueSize = 2
                self.cap.StartGrabbing(pylon.GrabStrategy_LatestImages)
            elif self.grabstrat == 'onebyone':
                self.cap.StartGrabbing(pylon.GrabStrategy_OneByOne)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        except genicam.GenericException as e:
            print("An exception occurred.")
            print(e.GetDescription())
    def __iter__(self):
        self.count = -1
        return self
    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.StopGrabbing()
            self.cap.Close()
            cv2.destroyAllWindows()
            raise StopIteration
#        if self.cap.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
#            self.cap.ExecuteSoftwareTrigger()
#        grabResult = self.cap.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
        while not grabResult.IsValid():
            grabResult = self.cap.RetrieveResult(0, pylon.TimeoutHandling_Return)
        if grabResult.GetNumberOfSkippedImages():
            print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
        if grabResult.GrabSucceeded():
            image = self.converter.Convert(grabResult)
            img0 = image.GetArray()
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
        return self.sources, img0, None
    def __len__(self):
        return 0
