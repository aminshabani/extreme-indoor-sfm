from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5.QtWidgets as qtw
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPixmap
import qdarkstyle
import os, sys


class ExtendedQLabel(qtw.QLabel):
    clicked = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("QLabel::hover" "{" "background-color : gray;" "}")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.clicked.emit(self.objectName())
        else:
            self.clicked.emit(self.objectName())


class FileListWidget():
    def __init__(self, MW, app):
        #super().__init__(MW)
        self.app = app
        self.list = qtw.QListView(MW)
        self.list.setWindowTitle('Houses List')
        self.list.clicked.connect(self.app.update_folder)
        self.model = QtGui.QStandardItemModel(self.list)
        self.list.setModel(self.model)

        self.list.setObjectName('sca_folders')
        self.initUI()
        self.updateUI()

    def initUI(self):
        self.labels = []
        self.dir = None

    def get_row(self, indx):
        self.list.setCurrentIndex(
            self.model.indexFromItem(self.model.item(indx)))
        return self.model.indexFromItem(self.model.item(indx))

    def updateUI(self):
        self.size = [
            self.app.MW.width() * 0.002, 25,
            self.app.MW.width() * 0.09,
            self.app.MW.height() * 0.5
        ]
        self.list.setGeometry(self.size[0], self.size[1], self.size[2],
                              self.size[3])

    def update(self, dir):
        self.updateUI()
        self.model.removeRows(0, self.model.rowCount())
        self.dir = dir
        fileList = os.listdir(dir)
        fileList.sort()
        self.labels = []
        for i, x in enumerate(fileList):
            item = QtGui.QStandardItem()
            item.setText(x)
            if (os.path.exists("clean_data/{}/labels.json".format(x))):
                item.setForeground(QtGui.QColor("green"))
            self.model.appendRow(item)
            self.labels.append(item)
