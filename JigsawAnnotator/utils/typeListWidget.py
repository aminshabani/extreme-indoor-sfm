from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPixmap
import qdarkstyle
import os, sys

# room_types:
# 0) balcony 1) closet 2) western-style-room 3) japanese-style-room
# 4) Dining room 5) Kitchen 6) corridor 7) washroom 8) bathroom 9)toilet


class TypeListWidget():
    def __init__(self, MW, app, house):
        self.house = house
        self.app = app
        self.list = qtw.QListView(MW)
        self.list.setWindowTitle('Types List')
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        #self.list.clicked.connect(self.app.update_folder)
        self.model = QtGui.QStandardItemModel(self.list)
        self.list.setModel(self.model)

        self.list.setGeometry(5, 510, 150, 300)
        self.list.setObjectName('sca_flags')
        self.initUI()
        self.updateUI()

    def updateUI(self):
        self.size = [
            self.app.MW.width() * 0.002, 25 + self.app.MW.height() * 0.67,
            self.app.MW.width() * 0.09,
            self.app.MW.height() * 0.25
        ]
        self.list.setGeometry(self.size[0], self.size[1], self.size[2],
                              self.size[3])

    def initUI(self):
        types = [
            "Balcony", "Closet", "Western-style-room", "Japanese-style-room",
            "Dining room", "Kitchen", "Corridor", "Washroom", "Bathroom",
            "Toilet"
        ]
        self.model.removeRows(0, self.model.rowCount())
        for i, x in enumerate(types):
            item = QtGui.QStandardItem()
            item.setCheckable(True)
            item.setText(x)
            self.model.appendRow(item)
        return

    def update(self):
        for i in range(self.model.rowCount()):
            self.model.item(i).setCheckState(False)
        type = self.house.get_type()
        if (type == -1):
            return
        item = self.model.item(type)
        item.setCheckState(True)
