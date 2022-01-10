from PIL import Image, ImageDraw
import numpy as np
import json
from . import tools
from .bbox import BBox
import os
import logging
import shapely
import scipy.stats as stats
logger = logging.getLogger('log')

# TYPES: [Balcony, Closet, Western style room, Japanese style room, Dining Room
#  Kitchen, Corridor, Washroom, Bathroom, Toilet]


class Panorama:
    def __init__(self, house_name, name, args, room_type=-1):
        self.house_name = house_name
        self.name = name
        self.img = None
        self.type = room_type
        self.layout = None
        self.obj_list = []
        self.camera_height = None
        self.camera_ceiling_height = None
        self.scale = None
        self.size = [512, 1024]

        if args.use_gt:
            self.init_by_gt_json(args)
        else:
            self.init_by_preds(args)

        if self.layout is not None:
            self.layout = np.round(self.layout, 3)

    def init_by_preds(self, args):
        # load type predictions
        house_dir = f"{args.data_dir}/{self.house_name}"
        types = np.load(f"{house_dir}/room_type_preds.npy", allow_pickle=True).item()
        self.type = types[str(self.name)]

        # load layout predictions
        with open(f"{house_dir}/layout_preds/aligned_{self.name}.json") as f:
            jdata = json.load(f)
        lh = 3.2
        jdata['z1'] = -jdata['z1']
        jdata['cameraHeight'] = jdata['z1']/(jdata['z1']+jdata['z0']) * lh
        jdata['layoutHeight'] = lh
        gPoints = []
        for i in range(1, len(jdata['uv']), 2):
            p = jdata['uv'][i]
            p[1] = 1-p[1]
            c = list(tools.coords2xyz(p, 1))
            c = list(tools.coords2xyz(p, jdata['cameraHeight']/c[1]))
            xyz = tuple([c[0], c[2]])
            gPoints.append(xyz)
        self.layout = np.array(gPoints)
        self.camera_height = jdata['cameraHeight']
        self.camera_ceiling_height = lh - self.camera_height

        # make layout clean
        N = len(self.layout)
        for i, p in enumerate(self.layout):
            if abs(p[0]-self.layout[(i+1) % N][0]) < abs(p[1]-self.layout[(i+1) % N][1]):
                tmp = (p[0]+self.layout[(i+1) % N][0]) / 2
                self.layout[i][0] = tmp
                self.layout[(i+1) % N][0] = tmp
            else:
                tmp = (p[1]+self.layout[(i+1) % N][1]) / 2
                self.layout[i][1] = tmp
                self.layout[(i+1) % N][1] = tmp

        self.layout = np.round(self.layout, 3)

        # load detections
        with open(f"{house_dir}/detection_preds/aligned_{self.name}.json") as f:
            jdata_detection = json.load(f)
        lt = np.array(jdata_detection['scores'])
        lo = jdata_detection['pred_boxes']
        polygon = shapely.geometry.LinearRing(self.layout)
        for i in range(len(lo)):
            xs = range(int(lo[i][0]), int(lo[i][2]))
            intersections = []
            for j, x in enumerate(xs):
                p = [x/1024.0, 0.5]
                c = list(tools.coords2xyz(p, 10))
                line = shapely.geometry.LineString(
                    [(0, 0), (c[0]*10, c[2]*10)])
                intersection = polygon.intersection(line)

                if intersection.type == 'Point':
                    intersections.append([intersection.x, intersection.y])
            intersections = np.array(intersections)
            if len(intersections) == 0:
                continue
            modes, counts = stats.mode(intersections, 0)
            modes = modes[0]
            counts = counts[0]

            p = [lo[i][0]/1024.0, lo[i][1]/512.0]
            # p[1] = 1-p[1]
            c1 = list(tools.coords2xyz(p, 1))
            p = [lo[i][2]/1024.0, lo[i][3]/512.0]
            p[1] = 1-p[1]
            c2 = list(tools.coords2xyz(p, 1))

            if(counts[0] > counts[1]):
                x = modes[0]
                indices = np.where(abs(intersections[:, 0] - x) < tools.eps)[0]
                intersections = intersections[indices, :]
                miny = np.min(intersections[:, 1])
                maxy = np.max(intersections[:, 1])

                tmp = modes[0]
                tmp = tmp/((c1[0] + c2[0])/2)
                p = [lo[i][0]/1024.0, lo[i][1]/512.0]
                c1 = list(tools.coords2xyz(p, tmp))
                c1[0] = x
                c1[2] = miny
                if c1[1] > self.camera_ceiling_height:
                    c1[1] = self.camera_ceiling_height
                elif c1[1] < -self.camera_height:
                    c1[1] = -self.camera_height

                p = [lo[i][2]/1024.0, lo[i][3]/512.0]
                c2 = list(tools.coords2xyz(p, tmp))
                c2[0] = x
                c2[2] = maxy
                if c2[1] > self.camera_ceiling_height:
                    c2[1] = self.camera_ceiling_height
                elif c2[1] < -self.camera_height:
                    c2[1] = -self.camera_height
            else:
                y = modes[1]
                indices = np.where(intersections[:, 1] == y)[0]
                intersections = intersections[indices, :]
                minx = np.min(intersections[:, 0])
                maxx = np.max(intersections[:, 0])

                tmp = modes[1]
                tmp = tmp/((c1[2] + c2[2])/2)
                p = [lo[i][0]/1024.0, lo[i][1]/512.0]
                c1 = list(tools.coords2xyz(p, tmp))
                c1[2] = modes[1]
                c1[0] = minx
                if c1[1] > self.camera_ceiling_height:
                    c1[1] = self.camera_ceiling_height
                elif c1[1] < -self.camera_height:
                    c1[1] = -self.camera_height

                p = [lo[i][2]/1024.0, lo[i][3]/512.0]
                c2 = list(tools.coords2xyz(p, tmp))
                c2[2] = modes[1]
                c2[0] = maxx
                if c2[1] > self.camera_ceiling_height:
                    c2[1] = self.camera_ceiling_height
                elif c2[1] < -self.camera_height:
                    c2[1] = -self.camera_height

            obj = BBox(np.array([c1, c2]), lt[i])
            if(np.argmax(lt[i]) != 6):  # remove background classes
                self.obj_list.append(obj)

        # merge boundary detections
        for i, obj1 in enumerate(self.obj_list):
            if obj1.direction != 1:
                continue
            for j, obj2 in enumerate(self.obj_list):
                if j <= i:
                    continue
                if obj2.direction != 1:
                    continue
                if obj1.bbox[0, 2] != obj2.bbox[0, 2]:
                    continue
                dist1 = min(abs(obj1.bbox[:, 0]))
                dist2 = min(abs(obj2.bbox[:, 0]))
                if abs(dist1-dist2) < 0.5:
                    tmp = np.append(obj1.bbox, obj2.bbox, 0)
                    new_bbox = np.array([np.min(tmp, 0), np.max(tmp, 0)])
                    if abs(obj1.bbox[0, 0]-obj1.bbox[1, 0]) > abs(obj2.bbox[0, 0]-obj2.bbox[1, 0]):
                        new_type = obj1.type
                    else:
                        new_type = obj2.type
                    self.obj_list.remove(obj1)
                    self.obj_list.remove(obj2)
                    self.obj_list.append(BBox(new_bbox, new_type))
                    return

    def init_by_gt_json(self, args):
        path = os.path.join(cfg['DIRS']['gt_layout_dir'], self.house_name,
                            'aligned_' + self.name + '.json')
        if not os.path.exists(path):
            logger.warning("couldn't find {}".format(path))
            return
        jsdata = json.load(open(path))
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
        if cfg['HOUSE'].getboolean('use_pred'):
            if self.camera_ceiling_height + self.camera_height != 3.2:
                scale = 3.2/(self.camera_ceiling_height + self.camera_height)
                self.camera_ceiling_height = np.round(self.camera_ceiling_height * scale, 3)
                self.camera_height = np.round(self.camera_height * scale, 3)
                self.layout = np.round(self.layout * scale, 3)
                for obj in self.obj_list:
                    obj.bbox = np.round(obj.bbox * scale, 3)
                self.scale = scale

    def get_type(self):
        if isinstance(self.type, np.ndarray):
            return np.argmax(self.type)
        return self.type

    def get_panorama(self):
        path = os.path.join(cfg['DIRS']['gt_image_dir'], self.house_name,
                            'aligned_' + self.name + '.png')
        img = Image.open(path)
        return img

    @DeprecationWarning
    def get_top_down_view_raymapping(self, apply_lv=False):
        self.map_to_tdv = tools.map_pano_to_tdv(self)
        img = np.asarray(self.get_panorama())
        result = np.zeros([1024, 1024, 4], dtype=int)
        result[self.map_to_tdv[0], self.map_to_tdv[1], :3] = img
        result[:, :, 3] = (np.sum(result, 2) > 0) * 255
        result = result[256:-256, 256:-256, :]
        result = Image.fromarray(result.astype(np.uint8))
        return result

    def get_poly(self, scale=1):
        poly = shapely.geometry.Polygon(self.layout * scale)
        return poly

    def get_one_hot_top_down_view(self, color_room=True, color_door=True):
        mask = np.zeros((512, 512, 16))
        res = Image.new('RGBA', (512, 512), 0)
        draw = ImageDraw.Draw(res, 'RGBA')
        layout = self.layout * 25.6 + 256
        layout = [(r[0], r[1]) for r in layout]
        if color_room and self.type is not None:
            # draw.polygon(layout, outline=(0, 0, 0, 255),
                         # fill=tuple(tools.rcolors[self.get_type()]))
            draw.polygon(layout, fill=tuple(tools.rcolors[self.get_type()]))
        else:
            # draw.polygon(layout, outline=(0, 0, 0, 255), fill=(0, 0, 0, 200))
            draw.polygon(layout, fill=(0, 0, 0, 200))
        mask[:, :, self.get_type()] = (np.sum(res, 2)>0)

        for i, x in enumerate(self.obj_list):
            res = Image.new('RGBA', (512, 512), 0)
            draw = ImageDraw.Draw(res)
            # break
            bbox = x.bbox
            bbox = bbox * 25.6 + 256
            if color_door:
                draw.line([bbox[0, 0], bbox[0, 2], bbox[1, 0], bbox[1, 2]],
                          width=3,
                          fill=tuple(tools.colors[x.get_type()]))
            else:
                draw.line([bbox[0, 0], bbox[0, 2], bbox[1, 0], bbox[1, 2]],
                          width=3,
                          fill=(255, 255, 255, 255))
            mask[:, :, 10+x.get_type()] = np.maximum(mask[:, :, 10+x.get_type()], np.sum(res, 2)>0)
        # draw.ellipse([255, 255, 257, 257], fill=(255, 255, 255, 255))
        return mask

    def get_top_down_view(self, args):
        res = Image.new('RGBA', (512, 512), 0)
        draw = ImageDraw.Draw(res, 'RGBA')
        layout = self.layout * 25.6 + 256
        layout = [(r[0], r[1]) for r in layout]
        if not args.vis_ignore_room_colors and self.type is not None:
            draw.polygon(layout, fill=tuple(tools.rcolors[self.get_type()]))
        else:
            draw.polygon(layout, fill=(0, 0, 0, 200))

        for i, x in enumerate(self.obj_list):
            # break
            bbox = x.bbox
            bbox = bbox * 25.6 + 256
            if not args.vis_ignore_door_colors:
                draw.line([bbox[0, 0], bbox[0, 2], bbox[1, 0], bbox[1, 2]],
                          width=3,
                          fill=tuple(tools.colors[x.get_type()]))
            else:
                draw.line([bbox[0, 0], bbox[0, 2], bbox[1, 0], bbox[1, 2]],
                          width=3,
                          fill=(255, 255, 255, 200))
        if not args.vis_ignore_centers:
            draw.ellipse([255, 255, 257, 257], fill=(255, 255, 255, 200))
        return res

    def get_pano_mask(self, door=None, overimage=False, dpi=100):
        door_img = None
        mask = Image.new('RGB', (1024, 512), 0)
        img = self.get_panorama()
        if overimage:
            mask = img.convert('RGB')
        draw = ImageDraw.Draw(mask, 'RGBA')
        # DRAW Layout
        for x in self.layout:
            x = [x[0], self.camera_height, x[1]]
            p1 = list(tools.xyz2coords(x))
            p1[0] = p1[0] * 1024
            p1[1] = 512 - p1[1] * 512
            draw.ellipse([p1[0] - 3, p1[1] - 3, p1[0] + 3, p1[1] + 3],
                         fill=(255, 255, 255, 200))
            x[1] = self.camera_ceiling_height
            p2 = list(tools.xyz2coords(x))
            p2[0] = p2[0] * 1024
            p2[1] = p2[1] * 512
            draw.ellipse([p2[0] - 3, p2[1] - 3, p2[0] + 3, p2[1] + 3],
                         fill=(255, 0, 0, 255))
            draw.line([p1[0], p1[1], p2[0], p2[1]],
                      width=3, fill=(255, 255, 255, 200))

        for i in range(len(self.layout)):
            # Draw Floor
            p1 = self.layout[i]
            p2 = self.layout[(i + 1) % len(self.layout)]

            x = np.linspace(p1[0], p2[0], num=dpi, endpoint=True)
            y = np.linspace(p1[1], p2[1], num=dpi, endpoint=True)
            xy = []
            for j in range(len(x)):
                p = np.array(tools.xyz2coords(
                    [x[j], self.camera_height, y[j]])) * [1024, 512]
                xy.extend([p[0], 512 - p[1]])

            for j in range(0, len(xy) - 2, 2):
                if (xy[j + 2] - xy[j] < 0):
                    draw.line(xy[:j + 2], width=3, fill=(255, 255, 255, 200))
                    draw.line(xy[j + 2:], width=3, fill=(255, 255, 255, 200))
                    break
            else:
                draw.line(xy, width=3, fill=(255, 255, 255, 200))

            # Draw Ceil
            xy = []
            for j in range(len(x)):
                p = np.array(tools.xyz2coords(
                    [x[j], self.camera_ceiling_height, y[j]])) * [1024, 512]
                xy.extend([p[0], p[1]])

            for j in range(0, len(xy) - 2, 2):
                if (xy[j + 2] - xy[j] < 0):
                    draw.line(xy[:j + 2], width=3, fill=(255, 255, 255, 200))
                    draw.line(xy[j + 2:], width=3, fill=(255, 255, 255, 200))
                    break
            else:
                draw.line(xy, width=3, fill=(255, 255, 255, 200))

        # DRAW OBJECTS
        center = 0
        for i, obj in enumerate(self.obj_list):
            if door is not None and door != i:
                continue
            bbox = obj.bbox
            xmin = min(bbox[0, 0], bbox[1, 0])
            ymin = min(bbox[0, 1], bbox[1, 1])
            zmin = min(bbox[0, 2], bbox[1, 2])
            xmax = max(bbox[0, 0], bbox[1, 0])
            ymax = max(bbox[0, 1], bbox[1, 1])
            zmax = max(bbox[0, 2], bbox[1, 2])

            if door is not None:
                xavg = (xmin+xmax)/2
                yavg = (ymin+ymax)/2
                zavg = (zmin+zmax)/2
                center_point = np.array(tools.xyz2coords(
                    [xavg, yavg, zavg])) * [1024, 512]
                center = 512 - center_point[0]

            xy = []
            if abs(zmax-zmin) < abs(xmax-xmin):
                # ps = [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin],
                #       [xmin, ymax, zmin]]
                x = np.append(np.linspace(xmin, xmax, num=dpi, endpoint=True),
                              np.linspace(xmax, xmin, num=dpi, endpoint=True),
                              axis=0)
                y = np.append(np.linspace(ymin, ymin, num=dpi, endpoint=True),
                              np.linspace(ymax, ymax, num=dpi, endpoint=True),
                              axis=0)
                for j in range(len(x)):
                    p = np.array(tools.xyz2coords(
                        [x[j], y[j], zmin])) * [1024, 512]
                    xy.extend([p[0], p[1]])
            else:
                # ps = [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmax],
                #       [xmin, ymax, zmin]]
                z = np.append(np.linspace(zmin, zmax, num=dpi, endpoint=True),
                              np.linspace(zmax, zmin, num=dpi, endpoint=True),
                              axis=0)
                y = np.append(np.linspace(ymin, ymin, num=dpi, endpoint=True),
                              np.linspace(ymax, ymax, num=dpi, endpoint=True),
                              axis=0)
                for j in range(len(z)):
                    p = np.array(tools.xyz2coords(
                        [xmin, y[j], z[j]])) * [1024, 512]
                    xy.extend([p[0], p[1]])

            test_oversized = []
            total_xy = xy.copy()
            for j in range(2, len(xy) - 2, 2):
                if (xy[j] == xy[j + 2]):
                    continue
                if (xy[j] < xy[j - 2] and xy[j] < xy[j + 2]):
                    test_oversized.append(j)
            if (len(test_oversized) == 0):
                draw.polygon(xy, fill=tuple(tools.colors[obj.get_type()]))
            else:
                assert(test_oversized[0]<test_oversized[1])
                draw.polygon(xy[test_oversized[0] + 2:test_oversized[1]],
                             fill=tuple(tools.colors[obj.get_type()]))
                del xy[test_oversized[0] + 2:test_oversized[1]]
                draw.polygon(xy, fill=tuple(tools.colors[obj.get_type()]))

            # obtain door_img
            if door is not None:
                xind = (
                    np.array(list(set([int(x) for x in total_xy[::2]]))) + int(center)) % 1024
                # xind = sorted(xind.tolist())
                xind = range(np.min(xind)-10, np.max(xind)+10)
                xind = (np.array(xind)-int(center)) % 1024
                yind = (np.array(list(set([int(y) for y in total_xy[1::2]]))))
                # door_img = np.array(img)[min(yind):max(yind), :, :]
                door_img = np.array(img)[:, :, :]
                door_img = door_img[:, xind, :]
                door_img = Image.fromarray(door_img)
                door_img = door_img.resize((128, 128))
        return img, mask, door_img, center


if __name__ == "__main__":
    import matplotlib.pyplot as plt
