import sys
import os
import argparse

import PanoAnnotator.data as data
import PanoAnnotator.configs.Params as pm
import PanoAnnotator.utils as utils
import PanoAnnotator.views as views
import qdarkstyle
import HorizonNet.layout_viewer as layout_viewer
#import estimator

from PyQt5 import QtCore, QtGui, QtWidgets

from PanoAnnotator.views.PanoView import PanoView
from PanoAnnotator.views.MonoView import MonoView
from PanoAnnotator.views.ResultView import ResultView
from PanoAnnotator.views.LabelListView import LabelListView

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QProgressDialog
from PyQt5.QtCore import QCoreApplication


class PanoAnnotator(QMainWindow, views.MainWindowUi):
    def __init__(self, app, parent):
        super(PanoAnnotator, self).__init__()
        self.app = app
        self.pr = parent
        self.setupUi(self)
        self.actionOpenImage.triggered.connect(self.openImageFile)
        self.actionOpenJson.triggered.connect(self.openJsonFile)

        self.actionSaveFile.triggered.connect(self.saveSceneFile)

        self.mainScene = data.Scene(self)

        if pm.isDepthPred:
            # import PanoAnnotator.estimator as estimator
            # self.depthPred = estimator.DepthPred()
            self.depthPred = layout_viewer.get_depth
        else:
            self.depthPred = None

        self.panoView.setMainWindow(self)
        self.monoView.setMainWindow(self)
        self.resultView.setMainWindow(self)
        self.labelListView.setMainWindow(self)

    def openImageFile(self):
        filePath, ok = QFileDialog.getOpenFileName(self, "open",
                                                   pm.fileDefaultOpenPath,
                                                   "Images (*.png *.jpg)")
        if ok:
            self.openImage(filePath)
            # self.mainScene = self.createNewScene(filePath)
            # self.mainScene.initLabel()
            # self.initViewsByScene(self.mainScene)
        else:
            print('open file error')
        return ok

    def openImage(self, filepath):
        self.filepath = filepath
        self.mainScene = self.createNewScene(self.filepath)
        if (os.path.exists("{}.json".format(self.filepath[:-4]))):
            self.mainScene.loadLabel("{}.json".format(self.filepath[:-4]))
        else:
            self.mainScene.loadOldLabel("{}.json".format(self.filepath[:-4]))
            # self.mainScene.initLabel()
        self.initViewsByScene(self.mainScene)

    def openJsonFile(self):
        filePath, ok = QFileDialog.getOpenFileName(self, "open",
                                                   pm.fileDefaultOpenPath,
                                                   "Json (*.json)")
        if ok:
            imagePath = os.path.join(os.path.dirname(filePath),
                                     pm.colorFileDefaultName)
            self.mainScene = self.createNewScene(imagePath)
            self.mainScene.loadLabel(filePath)
            self.initViewsByScene(self.mainScene)
        else:
            print('open file error')
        return ok

    def saveSceneFile(self):

        curPath = self.mainScene.getCurrentPath()
        savePath = "{}.json".format(self.filepath[:-4])
        #utils.saveSceneAsMaps(savePath, self.mainScene)
        utils.saveSceneAsJson(savePath, self.mainScene)
        self.close()

    def createNewScene(self, filePath):
        scene = data.Scene(self)
        scene.initScene(filePath, self.depthPred)
        return scene

    def initViewsByScene(self, scene):
        self.panoView.initByScene(scene)
        self.monoView.initByScene(scene)
        self.resultView.initByScene(scene)
        self.labelListView.initByScene(scene)

    def moveMonoCamera(self, coords):
        self.monoView.moveCamera(coords)

    def updateViews(self):
        self.panoView.update()
        self.monoView.update()
        self.resultView.update()

    def updateListView(self):
        self.labelListView.refreshList()

    def updataProgressView(self, val):
        self.progressView.setValue(val)
        QCoreApplication.processEvents()

    def refleshProcessEvent(self):
        QCoreApplication.processEvents()

    def closeEvent(self, event):
        # if self.depthPred:
        # self.depthPred.sess.close()
        event.accept()
        return

    def keyPressEvent(self, event):
        print("main")
        key = event.key()

    def run(self, path):
        print(path)
        self.showMaximized()
        self.openImage(path)
