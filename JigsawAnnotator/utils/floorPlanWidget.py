from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QCoreApplication, Qt, QMimeData
from PyQt5.QtGui import QIcon, QPixmap, QTransform, QDrag, QCursor, QPainter
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

import qdarkstyle
import os, sys
import numpy as np


class FPExtendedQImage(qtw.QLabel):
    clicked = QtCore.pyqtSignal(str)

    def __init__(self, pixmap=None):
        super().__init__()
        if(pixmap is not None):
            self.pixmap = pixmap
            self.setPixmap(self.pixmap)
        self.setStyleSheet("QLabel { background-color: rgba(0,0,0,0%)}")


class FloorPlanWidget(qtw.QScrollArea):
    def __init__(self, MW, app, house):
        super().__init__(MW)
        self.house = house
        self.app = app
        self.setObjectName('fp_layout')

        self.widget = qtw.QWidget()
        self.vbox = qtw.QVBoxLayout()
        self.objs = []
        self.dir = None
        self.widget.setLayout(self.vbox)

        #Scroll Area Properties
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)
        self.updateUI()

    def updateUI(self):
        self.size = [
            self.app.MW.width() * 0.4, 25,
            self.app.MW.width() * 0.59,
            self.app.MW.height() - 60
        ]
        self.setGeometry(self.size[0], self.size[1], self.size[2],
                         self.size[3])

    def update(self):
        for x in self.objs:
            x.setParent(None)
            self.vbox.removeWidget(x)
        self.objs = []
        if self.app.view_fp:
            pixmap = self.house.getFPPixmap()
            fp = FPExtendedQImage(pixmap)
        else:
            fp = FPExtendedQImage()
        fp.setAlignment(Qt.AlignTop)
        fp.setObjectName('floorplan')
        self.objs.append(fp)
        self.vbox.addWidget(fp)

        tmp_obj, tmp_pos = self.house.get_added_panos(self.app.view_room_colors, self.app.view_door_colors)

        for i in range(len(tmp_pos)):
            pixmap = tmp_obj[i]
            obj = FPExtendedQImage(pixmap)
            obj.setParent(fp)
            obj.setAlignment(Qt.AlignTop)
            obj.setGeometry(tmp_pos[i][0], tmp_pos[i][1], 768, 768)
            obj.setObjectName('floorplan')
            self.objs.append(obj)
