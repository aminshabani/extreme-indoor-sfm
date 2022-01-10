from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPixmap, QTransform
import qdarkstyle
import os, sys
import numpy as np
from PanoAnnotator import PanoAnnotator


class RoomExtendedQImage(qtw.QLabel):
    clicked = QtCore.pyqtSignal()

    def __init__(self, pixmap, w):
        super().__init__()
        self.pixmap = pixmap
        self.pixWidth = self.pixmap.width()
        self.setPixmap(self.pixmap.scaledToWidth(int(w)))
        self.setStyleSheet(
            "QLabel { background-color: white} QLabel::hover {background-color : lightgray;}"
        )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.clicked.emit()
        else:
            self.clicked.emit()


class RoomViewWidget(qtw.QScrollArea):
    def __init__(self, MW, app, house):
        super().__init__(MW)
        self.house = house
        self.app = app
        self.setObjectName('image_layout')
        self.initUI()
        self.updateUI()

    def updateUI(self):
        self.size = [
            self.app.MW.width() * 0.245, 25,
            self.app.MW.width() * 0.15,
            self.app.MW.height() - 60
        ]
        self.setGeometry(self.size[0], self.size[1], self.size[2],
                         self.size[3])

    def initUI(self):
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
        return

    def update(self):
        for x in self.objs:
            x.setParent(None)
            self.vbox.removeWidget(x)
        self.objs = []
        if (self.house.current_pano == -1):
            return
        if (not os.path.exists("{}/{}/aligned_{}.json".format(
                self.house.dir, self.house.house_name,
                self.house.current_pano))):
            if (self.house.types[self.house.current_pano] != -1):
                self.app.panoAnnotator.run("{}/{}/aligned_{}.png".format(
                    self.house.dir, self.house.house_name,
                    self.house.current_pano))

        vis, tdw, tdf = self.house.get_current_pano(self.app.view_room_colors, self.app.view_door_colors)
        obj = RoomExtendedQImage(vis, self.app.MW.width() * 0.15)
        obj.clicked.connect(self.app.openPano)
        obj.setAlignment(Qt.AlignCenter)
        obj.setObjectName('vis_img')
        self.objs.append(obj)
        self.vbox.addWidget(obj)

        obj = RoomExtendedQImage(tdw, self.app.MW.width() * 0.15)
        obj.setAlignment(Qt.AlignCenter)
        obj.setObjectName('tdw_img')
        self.objs.append(obj)
        self.vbox.addWidget(obj)

        obj = RoomExtendedQImage(tdf, self.app.MW.width() * 0.15)
        obj.setAlignment(Qt.AlignCenter)
        obj.setObjectName('tdf_img')
        self.objs.append(obj)
        self.vbox.addWidget(obj)
