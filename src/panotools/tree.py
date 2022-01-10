import numpy as np
import shapely
import shapely.wkt
from . import tools
from PIL import Image, ImageOps, ImageDraw
import copy

class Tree:
    def __init__(self):
        self.rooms = []
        self.pairs_list = []
        self.iou = -1
        self.position_memory = dict()
        self.polygons = []
        self.poly_types = []

    def __len__(self):
        return len(self.rooms)

    def __copy__(self):
        copy_tree = Tree()
        copy_tree.rooms = self.rooms.copy()
        copy_tree.pairs_list = self.pairs_list.copy()
        return copy_tree

    def add_pair(self, pair, house):
        if self.pairs_list == []:
            self.pairs_list = np.asarray([pair])
        else:
            self.pairs_list = np.append(self.pairs_list, [pair], 0)
        if len(self.pairs_list) == 1:
            self.rooms.append(pair[0][0])
            self.rooms.append(pair[1][0])
        else:
            if pair[0][0] not in self.rooms:
                assert pair[1][0] in self.rooms
                self.rooms.append(pair[0][0])
            elif pair[1][0] not in self.rooms:
                assert pair[0][0] in self.rooms
                self.rooms.append(pair[1][0])
            else:
                assert False

    def drop_last(self):
        self.pairs_list = self.pairs_list[:-1]
        if len(self.pairs_list) == 0:
            self.rooms = []
        else:
            self.rooms = self.rooms[:-1]

    def is_equal(self, other):
        if len(self) != len(other):
            return False
        other_list = other.pairs_list.reshape(
            [other.pairs_list.shape[0], -1]).tolist()
        for pair in self.pairs_list:
            if pair.reshape([-1]).tolist() not in other_list:
                return False
        return True

    def fix_positions(self, house, index=0, tmp_pos=None, polys=None, poly_types=None, offset_t=1):
        if index == 0:
            self.iou = -1
            self.position_memory = dict()
            self.polygons = []
            self.poly_types = []
        if index == len(self.pairs_list):
            iou = 0
            cnt = 0
            for i, p1 in enumerate(polys):
                if poly_types[i] in [4, 5]:
                    continue
                cnt += 1
                for j, p2 in enumerate(polys):
                    if j == i:
                        continue
                    # tmppolys = copy.deepcopy(polys)
                    # tmppolys.pop(i)
                    # union = shapely.ops.unary_union(tmppolys)
                    # union = shapely.wkt.loads(shapely.wkt.dumps(union, rounding_precision=2)) 
                    # print(union)
                    iou += p1.intersection(p2).area / p1.area
            if cnt != 0:
                iou = round(iou / cnt, 3)
            else:
                iou = 0
            if self.iou == -1 or iou < self.iou:
                self.iou = iou
                self.polygons = copy.deepcopy(polys)
                self.position_memory = copy.deepcopy(tmp_pos)
                self.poly_types = copy.deepcopy(poly_types)
            return

        pair = self.pairs_list[index].copy()
        if tmp_pos is not None and pair[0][0] not in tmp_pos:
            tmp = pair[0].copy()
            pair[0] = pair[1]
            pair[1] = tmp
        p1 = house.panos[pair[0][0]]
        d1 = p1.obj_list[pair[0][1]]
        p2 = house.panos[pair[1][0]]
        d2 = p2.obj_list[pair[1][1]]
        if index == 0:
            polys = [tools.update_location(p1.get_poly(), 0)]
            poly_types = [p1.get_type()]
            tmp_pos = {pair[0][0]: [0, 0, 0]}

        memo = tmp_pos[pair[0][0]]
        if (abs(d2.direction - d1.direction) == 0):
            rot = memo[2] + 180
        elif (abs(d2.direction - d1.direction) == 2):
            rot = memo[2]
        else:
            rot = ((d2.direction * 90) -
                   (d1.direction * 90) + memo[2]) % 360
        p = tools.update_location(d1.get_center(), memo[2])
        pp = tools.update_location(d2.get_center(), rot)
        offset = abs(d1.length() - d2.length())/2
        ddist = [p.x - pp.x, p.y - pp.y]
        poly_types.append(p2.get_type())
        if offset_t == 1:
            tmp_pos[pair[1][0]] = [ddist[0] + memo[0], ddist[1] + memo[1], rot]
            poly = tools.update_location(
                p2.get_poly(), rot, [ddist[0] + memo[0], ddist[1] + memo[1]])
            polys.append(poly)
            self.fix_positions(house, index+1, tmp_pos.copy(), polys.copy(), poly_types.copy())
            polys = polys[:-1]
        else:
            if (d1.direction) % 2 == 1:
                tmp_positions = np.linspace(ddist[0]-offset, ddist[0]+offset, offset_t)
                for x in tmp_positions:
                    tmp_ddist = [x, ddist[1]]
                    tmp_pos[pair[1][0]] = [tmp_ddist[0] + memo[0], tmp_ddist[1] + memo[1], rot]
                    poly = tools.update_location(
                        p2.get_poly(), rot, [tmp_ddist[0] + memo[0], tmp_ddist[1] + memo[1]])
                    polys.append(poly)
                    self.fix_positions(house, index+1, tmp_pos.copy(), polys.copy(), poly_types.copy(), offset_t)
                    polys = polys[:-1]
            else:
                tmp_positions = np.linspace(ddist[1]-offset, ddist[1]+offset, offset_t)
                for y in tmp_positions:
                    tmp_ddist = [ddist[0], y]
                    tmp_pos[pair[1][0]] = [tmp_ddist[0] + memo[0], tmp_ddist[1] + memo[1], rot]
                    poly = tools.update_location(
                        p2.get_poly(), rot, [tmp_ddist[0] + memo[0], tmp_ddist[1] + memo[1]])
                    polys.append(poly)
                    self.fix_positions(house, index+1, tmp_pos.copy(), polys.copy(), poly_types.copy(), offset_t)
                    polys = polys[:-1]

    def check_type_conditions(self, house):
        for pair in self.pairs_list:
            p1 = house.panos[pair[0][0]]
            d1 = p1.obj_list[pair[0][1]]
            p2 = house.panos[pair[1][0]]
            d2 = p2.obj_list[pair[1][1]]
            if d1.get_type() != d2.get_type():
                return False
        return True

    def get_types(self):
        return self.poly_types

    def get_masks(self, house, is_train):

        def draw_shape(poly, t, mask=None):
            image = Image.new('L', (256, 256), 0)
            if mask is None:
                mask = np.zeros((256, 256, 16))
            draw = ImageDraw.Draw(image)
            if poly.type == 'LineString':
                bbox = np.array(poly.xy) * 6.4 + 128
                bbox = bbox.transpose()
                draw.line([bbox[0, 0], bbox[0, 1], bbox[1, 0], bbox[1, 1]],
                          width=3, fill=255)
                image = ImageOps.flip(image)
                if isinstance(t, np.ndarray):
                    for i in range(len(t)-1):
                        mask[:, :, i+10] = np.maximum(image*t[i], mask[:, :, i+10])
                else:
                    mask[:, :, t+10] = np.maximum(image, mask[:, :, t+10])
            else:
                layout = np.array(poly.exterior.xy) * 6.4 + 128
                layout = [(r[0], r[1]) for r in layout.transpose()]
                draw.polygon(layout, outline=255, fill=255)
                image = ImageOps.flip(image)
                if isinstance(t, np.ndarray):
                    for i in range(len(t)):
                        mask[:, :, i] = np.maximum(image*t[i], mask[:, :, i])
                else:
                    mask[:, :, t] = np.maximum(image, mask[:, :, t])
            return mask

        masks = []
        types = []
        memory = self.position_memory

        # random augmentation
        if is_train:
            p = np.random.choice(list(memory.keys()), 1)[0]
            tmpx = memory[p][0]
            tmpy = memory[p][1]
        # translate by the camera mass center
        else:
            # tmpx = self.center_of_mass[0]
            # tmpy = self.center_of_mass[1]

            # tmpx, tmpy = 0, 0

            tmpx = np.mean([memory[m][0] for m in memory])
            tmpy = np.mean([memory[m][1] for m in memory])
        for m in memory:
            memory[m][0] = memory[m][0] - tmpx
            memory[m][1] = memory[m][1] - tmpy
        pairs = self.pairs_list
        painted = []
        for i, pair in enumerate(pairs):
            pair = pair.copy()
            if i != 0:
                if pair[0][0] not in painted:
                    tmp = pair[0].copy()
                    pair[0] = pair[1]
                    pair[1] = tmp
            p1 = house.panos[pair[0][0]]
            p2 = house.panos[pair[1][0]]
            memo = memory[pair[0][0]]
            if i == 0:
                poly = tools.update_location(
                    p1.get_poly(), 0, [memo[0], memo[1]])
                res = draw_shape(poly, p1.type)
                types.append(p1.get_type())
                for door in p1.obj_list:
                    p = tools.update_location(
                        door.get_line(), 0, [memo[0], memo[1]])
                    res = draw_shape(p, door.type, res)
                masks.append(res)
                painted.append(pair[0][0])

            memo = memory[pair[1][0]]
            poly = tools.update_location(
                p2.get_poly(), memo[2], [memo[0], memo[1]])
            res = draw_shape(poly, p2.type)
            types.append(p2.get_type())
            for door in p2.obj_list:
                p = tools.update_location(door.get_line(), memo[2], [
                    memo[0], memo[1]])
                res = draw_shape(p, door.type, res)
            masks.append(res)
            painted.append(pair[1][0])
        return masks 
