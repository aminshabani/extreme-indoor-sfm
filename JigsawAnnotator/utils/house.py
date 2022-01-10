from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPixmap, QTransform
import qdarkstyle
import numpy as np
import os, glob, sys
from PIL import Image
import json
import utils.layout_loader as layout_loader


class House():
    def __init__(self):
        self.dir = None
        self.house_name = None
        self.flags = []
        self.pano_list = []
        self.current_pano = -1
        self.current_f_zoom = 1
        self.p_zoom = 1
        self.positions = dict()
        self.rotates = None
        self.types = None

    def reset(self, dir, house_name, ignore_rotations, ignore_centers):
        self.dir = dir
        self.house_name = house_name
        self.pano_list = os.listdir(os.path.join(dir, house_name))
        self.pano_list = [
            x[8:-4] for x in self.pano_list
            if ("aligned_" in x and ".png" in x)
        ]
        self.pano_list.sort()
        self.rotates = (np.zeros(len(self.pano_list))).tolist()
        self.types = (np.zeros(len(self.pano_list)) - 1).tolist()
        self.positions = dict()
        self.current_pano = -1
        self.current_f_zoom = 1
        self.flags = []

        if (os.path.exists(
                os.path.join(self.dir, self.house_name, "room_types.txt"))):
            self.load_room_types()

        if (os.path.exists(
                os.path.join(self.dir, self.house_name, "labels.json"))):
            self.load_data(ignore_rotations, ignore_centers)
        elif(os.path.exists(
                os.path.join('preds_data/old_alignments', self.house_name, "labels.json"))):
            self.load_old_data(ignore_rotations, ignore_centers)

    def load_room_types(self):
        file = open(os.path.join(self.dir, self.house_name, "room_types.txt"))
        for line in file:
            line = line.split()
            self.types[int(line[1])] = int(line[3])


    def load_old_data(self, ignore_rotations, ignore_centers):
        print("LOADING THE OLD DATAS!")
        try:
            file = open(os.path.join('preds_data/old_alignments', self.house_name, "labels.json"))
            data = json.load(file)
        except:
            print("EXCEPTION IN LOADING JSON FILE")
        if(not ignore_rotations):
            self.rotates = ((np.array(data['rotations'])-1)%4).tolist()
        self.current_f_zoom = data['scales'][0]/1.35
        self.p_zoom = 1
        tmp = data['positions']
        for key in tmp:
            if(not ignore_centers):
                self.positions[int(key)] = (np.array(tmp[key], dtype=float)/1.35).astype(int).tolist()
            else:
                if(tmp[key][0]!=0 or tmp[key][1]!=0):
                    self.positions[int(key)] = [np.random.randint(512),np.random.randint(512)]
        self.flags = data['flags']
        return


    def load_data(self, ignore_rotations, ignore_centers):
        try:
            file = open(os.path.join(self.dir, self.house_name, "labels.json"))
            data = json.load(file)
        except:
            print("EXCEPTION IN LOADING JSON FILE")
            return
        self.types = data['room_types']
        tmp = data['positions']

        if(not ignore_rotations):
            self.rotates = data['rotations']
        for key in tmp:
            if(not ignore_centers):
                self.positions[int(key)] = tmp[key]
            else:
                if(tmp[key][0]!=0 or tmp[key][1]!=0):
                    self.positions[int(key)] = [np.random.randint(512),np.random.randint(512)]

        self.flags = data['flags']
        self.current_f_zoom = data['scales'][0]
        self.p_zoom = data['scales'][1]

    def save_data(self):
        file = open(os.path.join(self.dir, self.house_name, "labels.json"),
                    'w')
        data = dict()
        data['annotator'] = "Amin"
        data['version'] = "1"
        data['name'] = self.house_name
        data['num_panos'] = len(self.pano_list)
        data['num_valid_panos'] = len(self.positions)
        data['pano_names'] = self.pano_list
        data['rotations'] = self.rotates
        data['room_types'] = self.types
        data['positions'] = self.positions
        data['flags'] = self.flags
        data['scales'] = [self.current_f_zoom, self.p_zoom]
        json.dump(data, file)

    def rotate(self):
        if (self.current_pano == -1):
            return
        self.rotates[self.current_pano] += 1
        self.rotates[self.current_pano] %= 4

    def zoom(self, is_zoom):
        if (qtw.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier):
            zoom_scale = 0.1
        else:
            zoom_scale = 0.01
        if (is_zoom):
            self.current_f_zoom += zoom_scale
        else:
            self.current_f_zoom -= zoom_scale

    def getFPPixmap(self):
        PM_floorplan = QPixmap("{}/{}/floorplan.jpg".format(
            self.dir, self.house_name))
        pixWidth = PM_floorplan.width()
        PM_floorplan = PM_floorplan.scaledToWidth(
            (pixWidth * self.current_f_zoom))
        return PM_floorplan

    @DeprecationWarning
    def getTDWPixmaps(self):
        PM_panos = []
        for i, name in enumerate(self.pano_list):
            #TODO Convert from numpy and change alpha, Then going for drag and drop, also what is the null pixmap
            pixmap = QPixmap("{}/{}/tdw_aligned_{}.png".format(
                self.dir, self.house_name, name))
            pixmap = pixmap.transformed(QTransform().scale(1, -1))
            pixmap = pixmap.transformed(
                QtGui.QTransform().rotate(90 * self.rotates[i]),
                QtCore.Qt.SmoothTransformation)
            PM_panos.append(pixmap)
        return PM_panos

    def getPanoPixmaps(self):
        PM_panos = []
        for i, name in enumerate(self.pano_list):
            pixmap = QPixmap("{}/{}/aligned_{}.png".format(
                self.dir, self.house_name, name))
            # pixmap = pixmap.transformed(QTransform().scale(1, -1))
            # pixmap = pixmap.transformed(QtGui.QTransform().rotate(90*self.rotates[i]), QtCore.Qt.SmoothTransformation)
            PM_panos.append(pixmap)
        return PM_panos

    def get_current_pano(self, view_room_colors, view_door_colors):
        if (self.current_pano == -1):
            return None, None, None
        name = self.pano_list[self.current_pano]
        vis = QPixmap("{}/{}/aligned_{}.png".format(self.dir, self.house_name,
                                                    name))
        layout = json.load(
            open("{}/{}/aligned_{}.json".format(self.dir, self.house_name,
                                                name)))

        tdw = layout_loader.get_tdv("{}/{}/aligned_{}".format(
            self.dir, self.house_name, name), self.types[self.current_pano], view_room_colors, view_door_colors)
        tdw = QtGui.QImage(np.array(tdw, dtype=np.uint8), tdw.size[0],
                           tdw.size[1], QtGui.QImage.Format_RGBA8888)
        tdw = QPixmap(tdw)
        # tdw = tdw.transformed(QTransform().scale(1, -1))
        tdw = tdw.transformed(
            QtGui.QTransform().rotate(90 * self.rotates[self.current_pano]),
            QtCore.Qt.SmoothTransformation)

        tdf = layout_loader.get_pano_mask("{}/{}/aligned_{}".format(
            self.dir, self.house_name, name),
                                          overimage=True)
        tdf = QtGui.QImage(np.array(tdf, dtype=np.uint8), tdf.size[0],
                           tdf.size[1], QtGui.QImage.Format_RGBA8888)
        tdf = QPixmap(tdf)
        # tdf = tdf.transformed(QTransform().scale(1, -1))
        # tdf = tdf.transformed(QtGui.QTransform().rotate(90*self.rotates[self.current_pano]), QtCore.Qt.SmoothTransformation)
        return vis, tdw, tdf

    def get_original_pano(self):
        name = self.pano_list[self.current_pano]
        pano = QPixmap("{}/{}/aligned_{}.png".format(self.dir, self.house_name,
                                                     name))
        return pano

    def add_pano(self):
        if (self.current_pano not in self.positions):
            self.positions[self.current_pano] = [0, 0]

    def remove_pano(self):
        if (self.current_pano in self.positions):
            del self.positions[self.current_pano]

    def get_added_panos(self, view_room_colors, view_door_colors):
        panos = []
        poses = []
        for key in self.positions:
            name = self.pano_list[key]
            # img = np.array(Image.open("{}/{}/tdf_aligned_{}.png".format(self.dir, self.house_name, name)))
            # img[:,:,3] = img[:,:,3]*0.85

            tdw = layout_loader.get_tdv("{}/{}/aligned_{}".format(
                self.dir, self.house_name, name), self.types[key], view_room_colors, view_door_colors)
            tdw = QtGui.QImage(np.array(tdw, dtype=np.uint8), tdw.size[0],
                               tdw.size[1], QtGui.QImage.Format_RGBA8888)
            tdw = QPixmap(tdw)
            # tdw = tdw.transformed(QTransform().scale(1, -1))
            tdw = tdw.transformed(
                QtGui.QTransform().rotate(90 * self.rotates[key]),
                QtCore.Qt.SmoothTransformation)

            #tdf = QPixmap("{}/{}/tdf_aligned_{}.png".format(self.dir, self.house_name, name))
            # qImg = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_RGBA8888)
            # tdf = QPixmap.fromImage(qImg)
            # tdf = tdf.transformed(QTransform().scale(1, -1))
            # tdf = tdf.transformed(QtGui.QTransform().rotate(90*self.rotates[key]), QtCore.Qt.SmoothTransformation)

            pixWidth = tdw.width()
            tdw = tdw.scaledToWidth((pixWidth * self.p_zoom))
            pos = self.positions[key]

            panos.append(tdw)
            poses.append(pos)
        return panos, poses

    def move_pano(self, d):
        if (self.current_pano not in self.positions):
            return
        scale = 1
        if (qtw.QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier):
            scale = 10
        pos = self.positions[self.current_pano]
        if (d == 'l'):
            self.positions[self.current_pano] = [pos[0] - scale, pos[1]]
        elif (d == 'r'):
            self.positions[self.current_pano] = [pos[0] + scale, pos[1]]
        elif (d == 'u'):
            self.positions[self.current_pano] = [pos[0], pos[1] - scale]
        elif (d == 'd'):
            self.positions[self.current_pano] = [pos[0], pos[1] + scale]

    def set_type(self, type):
        if (self.current_pano != -1):
            self.types[self.current_pano] = type
        return

    def get_type(self):
        if (self.current_pano != -1):
            return self.types[self.current_pano]
        return -1

    def update_flag(self, flag):
        if (self.dir is None):
            return
        if (flag > 4):
            return
        if (flag in self.flags):
            self.flags.remove(flag)
        else:
            self.flags.append(flag)
        return

    def get_flags(self):
        return self.flags


if __name__ == "__main__":
    house = House('dataset', '10A0iX')
