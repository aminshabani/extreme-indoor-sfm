from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPixmap, QTransform
import qdarkstyle
import os, sys


class ExtendedQImage(qtw.QLabel):
    clicked = QtCore.pyqtSignal(str)

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
            self.clicked.emit(self.objectName())
        else:
            self.clicked.emit(self.objectName())


class ImageListWidget(qtw.QScrollArea):
    def __init__(self, MW, app, house):
        super().__init__(MW)
        self.house = house
        self.app = app
        self.setObjectName('image_layout')
        self.initUI()
        self.updateUI()

    def updateUI(self):
        self.size = [
            self.app.MW.width() * 0.094, 25,
            self.app.MW.width() * 0.15,
            self.app.MW.height() - 60
        ]
        self.setGeometry(self.size[0], self.size[1], self.size[2],
                         self.size[3])

    def initUI(self):
        self.widget = qtw.QWidget()
        self.vbox = qtw.QVBoxLayout()
        self.objs = []
        self.widget.setLayout(self.vbox)

        #Scroll Area Properties
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)
        return

    def update(self):
        pixMaps = self.house.getPanoPixmaps()

        for x in self.objs:
            x.setParent(None)
            self.vbox.removeWidget(x)

        self.objs = []
        for i, x in enumerate(pixMaps):
            obj = ExtendedQImage(x, self.app.MW.width() * 0.15)
            obj.setAlignment(Qt.AlignCenter)
            obj.setObjectName('pano:{}'.format(i))
            obj.clicked.connect(self.app.update_image)
            self.objs.append(obj)
            self.vbox.addWidget(obj)
