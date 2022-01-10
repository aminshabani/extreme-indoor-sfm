from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPixmap
import qdarkstyle
import os, sys


class FlagListWidget():
    def __init__(self, MW, app, house):
        self.house = house
        self.app = app
        self.list = qtw.QListView(MW)
        self.list.setWindowTitle('Flags List')
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        #self.list.clicked.connect(self.app.update_folder)
        self.model = QtGui.QStandardItemModel(self.list)
        self.list.setModel(self.model)

        self.list.setObjectName('sca_flags')

        # Duplex, Partiallly covered, additional impossible panos but some connected panos
        # ignored because completely outside or noises, lack of pano but possible, without overlap
        flags = ['duplex', 'impossible', 'ignored', 'hard', 'non overlap']
        self.model.removeRows(0, self.model.rowCount())
        for i, x in enumerate(flags):
            item = QtGui.QStandardItem()
            item.setCheckable(True)
            item.setText(x)
            self.model.appendRow(item)

        self.updateUI()

    def updateUI(self):
        self.size = [
            self.app.MW.width() * 0.002, 25 + self.app.MW.height() * 0.51,
            self.app.MW.width() * 0.09,
            self.app.MW.height() * 0.15
        ]
        self.list.setGeometry(self.size[0], self.size[1], self.size[2],
                              self.size[3])

    def update(self):
        for i in range(self.model.rowCount()):
            self.model.item(i).setCheckState(False)
        flags = self.house.get_flags()
        for index in flags:
            item = self.model.item(index)
            item.setCheckState(True)
