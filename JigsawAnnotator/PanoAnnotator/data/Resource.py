import os
import numpy as np

import PanoAnnotator.configs.Params as pm
import PanoAnnotator.utils as utils

from PIL import Image as Image
from PIL import ImageEnhance
from PyQt5.QtGui import QPixmap


class Resource(object):
    def __init__(self, name):

        self.name = name

        self.path = ''
        self.image = None  #(w,h)
        self.data = None  #(h,w)
        self.pixmap = None

    def initByImageFile(self, filePath):

        if os.path.exists(filePath):
            self.path = filePath
            self.image = Image.open(filePath).convert('RGB')
            enhancer = ImageEnhance.Contrast(self.image)
            img2 = enhancer.enhance(2)
            enhancer = ImageEnhance.Sharpness(img2)
            img2 = enhancer.enhance(2)
            self.data = np.asarray(self.image).astype(np.float)
            self.image = img2
            if pm.isGUI:
                self.pixmap = QPixmap(filePath)
            return True
        else:
            print("No default {0} image found".format(self.name))
            return False

    def initByImageFileDepth(self, filePath):

        if os.path.exists(filePath):
            self.path = filePath
            self.image = Image.open(filePath)
            self.data = np.asarray(self.image).astype(np.float)
            return True
        else:
            print("No default {0} image found".format(self.name))
            return False
