import time
import numpy as np
import cv2
from harvesters.core import Harvester
from harvesters.util.pfnc import mono_location_formats,\
    rgb_formats,\
    bgr_formats,\
    rgba_formats,\
    bgra_formats,\
    bayer_location_formats
from threading import Thread
from dataclasses import dataclass
import copy

@dataclass
class bufferdata:
    data : np.ndarray
    data_format : str
    height : int
    width : int
    components_per_pixel : int
    fps : float

class genicameras: 
    def __init__(self, source='', img_size=None):
        sources = []
        for s in source.split(" ")[1:]:
            sources.append(s)
        self.sources = sources
        n = len(self.sources)
        self.imgs = [None] * n
        self.h = Harvester()
        self.h.add_file('/opt/baumer-gapi-sdk/lib/libbgapi2_usb.cti')
        self.h.add_file('/opt/pylon5/lib64/gentlproducer/gtl/ProducerU3V.cti')
        self.h.update()
        self.mode = 'images'
        print(self.h.device_info_list)
        imgacqs = self.select_camera(sources)
        for i, imgacq in enumerate(imgacqs):
            #imgacq = self.h.create_image_acquirer(vendor=s)
            #imgacq = self.h.create_image_acquirer(i)
            if type(img_size) != type(None):
                imgacq.remote_device.node_map.Height.value = img_size[0]
                imgacq.remote_device.node_map.Width.value = img_size[1]
            imgacq.num_filled_buffers_to_hold = 50
            imgacq.start_acquisition(run_in_background=False)
            assert imgacq.is_acquiring(), "not acquiring images"
            thread = Thread(target=self.update,
                            args=([i, imgacq]),
                            daemon=True)
            thread.start()
            print('%g/%g: %s... ' % (i + 1, n, sources[i]), end='')
            img = self.grab(i)
            h, w = img.shape[:2]
            print(' success (%gx%g at %.2f FPS).' % (w, h, imgacq.statistics.fps))
    def select_camera(self, sources):
        ias = []
        for (i, device) in enumerate(self.h.device_info_list):
            for select in sources:
                if select in str(device):
                    try:
                        ia = self.h.create_image_acquirer(i)
                    except Exception as e:
                        print(e)
                    ias.append(ia)
        return ias
    def process_buffer(self, i, ia):
        buff = ia.fetch_buffer()
        component = buff.payload.components[0]
        retbuffer = copy.deepcopy(
            bufferdata(data_format = component.data_format,
                       data = component.data,
                       width = component.width,
                       height = component.height,
                       components_per_pixel = component.num_components_per_pixel,
                       fps = ia.statistics.fps)
        )
        buff.queue()
        return retbuffer
    def grab(self, i):
        try:
            img = self.imgs[i].copy()
            return img
        except Exception as e:
            print("Cannot open source, retrying")
            time.sleep(1)
            return self.grab(i)
    def convert2dimage(self, i, retbuffer):
        data_format = retbuffer.data_format
        if data_format in mono_location_formats:
            content = retbuffer.data.reshape(retbuffer.height,
                                                retbuffer.width)
        else:
            content = retbuffer.data.reshape(retbuffer.height,
                                             retbuffer.width,
                                             int(retbuffer.components_per_pixel))
            if data_format in rgb_formats or \
               data_format in rgba_formats:
                #content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
                content = content[:, :, ::-1]
            elif data_format in bgr_formats or \
                 data_format in bgra_formats:
                pass
                #content = content[:, :, ::-1]
                #content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
            elif data_format in bayer_location_formats:
                content = cv2.demosaicing(content, cv2.COLOR_BayerRG2RGB)
        #content = self.subsample(content)
        self.imgs[i] = content.astype(np.uint8)
    def subsample(self, content):
#        cv2.cvtColor(content, cv2.COLOR_BGR2YCrCb)
#        SSV, SSH = 2,2
#        cv2.boxFilter(transcol[:,:,1],ddepth=-1, ksize(2,2))
#        cv2.boxFilter(transcol[:,:,1],ddepth=-1, ksize(2,2))
#        crsub = crf[::SSV, ::SSH]
#        crsub = crf[::SSV, ::SSH]
#        imsub = [ycrcb_img[:,:,0],crsub,cbsub]

        import sys
        print(sys.getsizeof(content))
        print(content.nbytes)

        from turbojpeg import TurboJPEG,\
            TJFLAG_PROGRESSIVE,\
            TJFLAG_FASTUPSAMPLE,\
            TJFLAG_FASTDCT
        jpeg = TurboJPEG()
        content = jpeg.encode(content, quality=30)
        #content = jpeg.encode(content)
        content = jpeg.decode(content, flags=TJFLAG_FASTUPSAMPLE|TJFLAG_FASTDCT)

        print(sys.getsizeof(content))
        print(content.nbytes)
        #content = jpeg.decode(content)

        #import os
        #i = 1
        #while (os.path.exists(os.path.join('output', str(i) + '.tiff'))):
        #    i += 1
        #cv2.imwrite(os.path.join('output', str(i) + '.tiff'), content)
        return content
    def update(self, i, imgacq):
        while imgacq.is_acquiring():
            try:
                retbuffer = self.process_buffer(i, imgacq)
            except Exception as e:
                continue
            self.convert2dimage(i, retbuffer)
    def __iter__(self):
        return self
    def __next__(self):
        img = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  
            self.h.reset()
            cv2.destroyAllWindows()
            raise StopIteration
            exit()
        return self.sources, img, None
    def __len__(self):
        return 0

class genicameras_pool: 
    def __init__(self, sources='Basler'):
        self.sources = sources
        self.imgs = None
        self.webcam = False
        self.h = Harvester()
        self.h.add_file('/opt/baumer-gapi-sdk/lib/libbgapi2_usb.cti')
        self.h.add_file('/opt/pylon5/lib64/gentlproducer/gtl/ProducerU3V.cti')
        self.h.update()
        self.mode = 'images'
        print(self.h.device_info_list)
        imgacq = self.select_camera(sources)
        imgacq.num_filled_buffers_to_hold = 50
        imgacq.start_acquisition(run_in_background=True)
        assert imgacq.is_acquiring(), "not acquiring images"
        thread = Thread(target=self.update,
                        args=(imgacq,),
                        daemon=True)
        thread.start()
        print('%s... ' % (sources), end='')
        img = self.grab()
        h, w, _ = img.shape
        print(' success (%gx%g at %.2f FPS).' % (w, h, imgacq.statistics.fps))
    def select_camera(self, sources):
        for (i, device) in enumerate(self.h.device_info_list):
            if sources in str(device):
                try:
                    ia = self.h.create_image_acquirer(i)
                    break
                except Exception as e:
                    print(e)
        return ia
    def process_buffer(self, ia):
        buff = ia.fetch_buffer()
        component = buff.payload.components[0]
        retbuffer = copy.deepcopy(
            bufferdata(data_format = component.data_format,
                       data = component.data,
                       width = component.width,
                       height = component.height,
                       components_per_pixel = component.num_components_per_pixel,
                       fps = ia.statistics.fps)
        )
        buff.queue()
        return retbuffer
    def grab(self):
        try:
            img = self.imgs.copy()
            return img
        except Exception as e:
            print("Cannot open source, retrying")
            time.sleep(1)
            return self.grab()
    def convert2dimage(self, retbuffer):
        data_format = retbuffer.data_format
        if data_format in mono_location_formats:
            content = retbuffer.data.reshape(retbuffer.height,
                                             retbuffer.width)
        else:
            content = retbuffer.data.reshape(retbuffer.height,
                                             retbuffer.width,
                                             int(retbuffer.components_per_pixel))
            if data_format in rgb_formats or \
               data_format in rgba_formats or \
               data_format in bgr_formats or \
               data_format in bgra_formats:
                content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
                if data_format in bgr_formats:
                    content = content[:, :, ::-1]
            if data_format in bayer_location_formats:
                content = cv2.demosaicing(content, cv2.COLOR_BayerRG2RGB)
        self.imgs = np.asarray(content)
    def update(self, imgacq):
        while imgacq.is_acquiring():
            try:
                retbuffer = self.process_buffer(imgacq)
            except Exception as e:
                continue
            self.convert2dimage(retbuffer)
    def __iter__(self):
        return self
    def __next__(self):
        img = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  
            self.h.reset()
            cv2.destroyAllWindows()
            raise StopIteration
            exit()
        return (self.sources, img, None)
    def __len__(self):
        return 0

if __name__ == '__main__':
    pool = False
    if pool:
        gen = genicameras_pool('Baumer')
        for data in gen:
            print(data)
            sources, img, _ = data
            cv2.imshow('Camera', img)
    else:
#        gen = genicameras(['Basler', 'Baumer'])
        gen = genicameras('genicameras: Basler Baumer')
        for sources, img, _ in gen:
            cv2.imshow(sources[0], img[0])
            cv2.imshow(sources[1], img[1])
