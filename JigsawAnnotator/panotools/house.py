import copy
from scipy.spatial.distance import cdist
import numpy as np
from PIL import Image, ImageDraw
import shapely.geometry as sg
import shapely.ops as so
from shapely.ops import transform, nearest_points
from shapely import affinity
import os
from .tree import Tree
from .panorama import Panorama
from . import tools
import json
import matplotlib.pyplot as plt

class House:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.labeld = False
        if not os.path.exists("{}/{}/labels.json".format(path, name)):
            return
        data = json.load(open("{}/{}/labels.json".format(path, name)))
        if len(data['flags'])!=0 and int(data['flags'][0])<4:
            return
        self.labeld = True
        self.pano_names = data['pano_names']
        self.rotations = data['rotations']
        self.room_types = data['room_types']
        self.positions = data['positions']
        self.scale = data['scales'][0]
        self.pano_scale = data['scales'][1]

        self.fp = "{}/{}/floorplan.jpg".format(path, name)
        self.panos = []
        for name in self.pano_names:
            self.panos.append(Panorama("{}/{}".format(self.path, self.name), 'aligned_'+name))

        self.positive_pairs = []
        self.negative_pairs = []
        self.check_connections()

    def get_fp_img(self, type="RGB"):
        img = Image.open(self.fp).convert(type)
        img = img.resize((int(img.size[0] * self.scale), int(img.size[1] * self.scale)))
        return img

    def dindex_to_panoindex(self, index):
        for i, pano in enumerate(self.panos):
            if (index < len(pano.doors)):
                return i, int(index)
            index -= len(pano.doors)

    def visualize_alignment(self):
        fp = self.get_fp_img()
        for i in self.positions:
            pos = self.positions[i]
            i = int(i)
            pano = self.panos[i].get_top_down_view()
            pano = pano.rotate(-90 * self.rotations[i])
            pano = pano.resize((int(pano.size[0] * self.pano_scale), int(pano.size[1] * self.pano_scale)))
            pano = pano.crop((-pos[0], -pos[1], fp.size[0] - pos[0], fp.size[1] - pos[1]))
            alpha = pano.split()[-1]
            fp = Image.composite(pano, fp, alpha)
        fp.show()

    def check_connections(self):
        objs = []
        for name in self.positions:
            pano = self.panos[int(name)]
            for j, obj in enumerate(pano.obj_list):
                dtype = obj.type
                bbox = obj.bbox * 25.6 + 256
                obj = sg.LineString([(bbox[0][0], bbox[0][2]), (bbox[1][0], bbox[1][2])])
                obj = affinity.rotate(obj, 90 * self.rotations[int(name)], (256,256))
                obj = affinity.translate(obj, self.positions[name][0], self.positions[name][1])
                objs.append([obj, int(name), j])
        dists = np.zeros([len(objs), len(objs)]) + 1e10
        for i in range(len(objs)):
            for j in range(len(objs)):
                if i==j:
                    continue
                tmp = nearest_points(objs[i][0].centroid, objs[j][0])
                d = tmp[1].distance(objs[i][0].centroid)
                dists[i,j] = d
        dists = np.round(dists,3)
        args = np.argmin(dists, 1)
        dists = np.min(dists,1)
        for i in range(len(objs)):
            for j in range(i+1, len(objs)):
                if args[i]==j and args[j]==i and dists[i]<10:
                    self.positive_pairs.append([objs[i][1:], objs[j][1:]])
                    # print(self.panos[objs[i][1]].obj_list[objs[i][2]].direction+self.rotations[objs[i][1]],
                        # self.panos[objs[j][1]].obj_list[objs[j][2]].direction+self.rotations[objs[j][1]])
                else:
                    self.negative_pairs.append([objs[i][1:], objs[j][1:]])

#####################################################################################
import glob
names = os.listdir("clean_data/")
count = 0
cnt = 0
lblds = 0
panos = np.zeros(13)
valid_panos = np.zeros(13)
for name in names[:]:
    house = House("clean_data", name)
    if house.labeld:
        panos[len(house.panos)]+=1
        valid_panos[len(house.positions)]+=1
        lblds+=1
        cnt += len(house.positive_pairs)
        count += len(house.negative_pairs)
        # house.visualize_alignment()
print(lblds, cnt, count)
print(panos)
print(valid_panos)
