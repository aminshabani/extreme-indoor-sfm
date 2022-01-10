from detectron2.structures import BoxMode
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import numpy as np
import json
from . import tools
from .bbox import BBox
import os

class Panorama:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.img = None
        self.type = None
        self.layout = None
        self.obj_list = []
        self.camera_height = None
        self.camera_ceiling_height = None
        self.init_by_json()

    def init_by_json(self):
        json_path = '{}/{}.json'.format(self.path.replace('images', 'gt_labels'), self.name)
        if not os.path.exists(json_path):
            # print(f"couldn't find {json_path}")
            return
        jsdata = json.load(open(json_path))
        lp = jsdata['layoutPoints']
        lp = np.array([x['xyz'] for x in lp['points']])
        lp = np.round(lp, 5)
        lp = [(x[0], x[2]) for x in lp]
        self.layout = np.array(lp)

        lo = jsdata['layoutObj2ds']['obj2ds']
        lt = np.array([x['obj_type'] for x in lo])
        lo = np.array([x['points'] for x in lo])
        lo = np.round(lo, 5)
        self.obj_list = []
        for i in range(len(lo)):
            obj = BBox(lo[i], lt[i])
            self.obj_list.append(obj)

        self.camera_height = jsdata['cameraHeight']
        self.camera_ceiling_height = jsdata['cameraCeilingHeight']

    def get_panorama(self):
        if self.img is None:
            self.img = Image.open("{}/{}.png".format(self.path, self.name))
        return self.img

    def get_detectron_annotation(self, img_id=-1):
        dpi  = 300
        record = dict()
        record['file_name'] = "{}/{}.png".format(self.path, self.name)
        record['image_id'] = img_id
        record['height'] = 512
        record['width'] = 1024
        objs = []
        for i, obj in enumerate(self.obj_list):
            bbox = obj.bbox
            xmin = min(bbox[0, 0], bbox[1, 0])
            ymin = min(bbox[0, 1], bbox[1, 1])
            zmin = min(bbox[0, 2], bbox[1, 2])
            xmax = max(bbox[0, 0], bbox[1, 0])
            ymax = max(bbox[0, 1], bbox[1, 1])
            zmax = max(bbox[0, 2], bbox[1, 2])
            xy = []
            if zmax == zmin:
                ps = [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin],
                      [xmin, ymax, zmin]]
                x = np.append(np.linspace(xmin, xmax, num=dpi, endpoint=True),
                              np.linspace(xmax, xmin, num=dpi, endpoint=True),
                              axis=0)
                y = np.append(np.linspace(ymin, ymin, num=dpi, endpoint=True),
                              np.linspace(ymax, ymax, num=dpi, endpoint=True),
                              axis=0)
                for j in range(len(x)):
                    p = np.array(tools.xyz2coords([x[j], y[j], zmin])) * [1024, 512]
                    xy.extend([p[0], p[1]])
            else:
                ps = [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmax],
                      [xmin, ymax, zmin]]
                z = np.append(np.linspace(zmin, zmax, num=dpi, endpoint=True),
                              np.linspace(zmax, zmin, num=dpi, endpoint=True),
                              axis=0)
                y = np.append(np.linspace(ymin, ymin, num=dpi, endpoint=True),
                              np.linspace(ymax, ymax, num=dpi, endpoint=True),
                              axis=0)
                for j in range(len(z)):
                    p = np.array(tools.xyz2coords([xmin, y[j], z[j]])) * [1024, 512]
                    xy.extend([p[0], p[1]])

            test_oversized = []
            if(obj.type<0):
                obj.type = -obj.type
            for j in range(2, len(xy) - 2, 2):
                if xy[j] == xy[j + 2]:
                    continue
                if xy[j] < xy[j - 2] and xy[j] < xy[j + 2]:
                    test_oversized.append(j)
            if len(test_oversized) == 0:
                x = xy[0::2]
                y = xy[1::2]
                # poly = [(x + 0.5, y + 0.5) for x, y in zip(x, y)]
                # poly = [p for x in poly for p in x]
                poly = xy
                assert len(poly)%2 == 0, poly
                assert len(poly)>6 , poly
                tmp_obj = {
                        "bbox": [np.min(x), np.min(y), np.max(x), np.max(y)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": obj.type,
                        }
                objs.append(tmp_obj)
            else:
                x = xy[test_oversized[0] + 2:test_oversized[1] - 2:2]
                y = xy[test_oversized[0] + 3:test_oversized[1] - 1:2]
                # poly = [(x + 0.5, y + 0.5) for x, y in zip(x, y)]
                # poly = [p for x in poly for p in x]
                poly = xy[test_oversized[0] + 2:test_oversized[1] - 2]
                tmp_obj = {
                        "bbox": [np.min(x), np.min(y), np.max(x), np.max(y)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": obj.type,
                        }
                if len(poly)%2==0 and len(poly)>6:
                    objs.append(tmp_obj)
                del xy[test_oversized[0] + 2:test_oversized[1]]
                x = xy[0::2]
                y = xy[1::2]
                # poly = [(x + 0.5, y + 0.5) for x, y in zip(x, y)]
                # poly = [p for x in poly for p in x]
                poly = xy
                assert len(poly)%2 == 0, poly
                assert len(poly)>6 , poly
                tmp_obj = {
                        "bbox": [np.min(x), np.min(y), np.max(x), np.max(y)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": obj.type,
                        }
                objs.append(tmp_obj)
        record['annotations'] = objs
        return record

