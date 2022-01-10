import numpy as np
import matplotlib.pyplot as plt
import sys, glob, os
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QPixmap, QTransform

# room_types:
# 0) balcony 1) outdoor 2) western-style-room 3) japanese-style-room
# 4) Dining room 5) Kitchen 6) corridor 7) washroom 8) bathroom 9)toilet


# ricoh_data = json.load(open('{}/{}.json'.format(PR_RICOH_DIR, self.house_name)))
# ricoh_data = ricoh_data['images']
# ricoh_data = [x for x in ricoh_data if x['file_name'][:-4] == self.name]
# ricoh_data = ricoh_data[0]['room_type']
# mapping = {'Washing_room': 7, 'Bathroom': 8, 'Kitchen': 5, 'Balcony': 0, 'Toilet': 9,
#            'Japanese-style_room': 3, 'Verandah': 0, 'Western-style_room': 2, 'Entrance': 6}
# self.type = mapping[ricoh_data]         

class Room_type_annotator():
    def __init__(self, app, house_dir):
        self.app = app
        self.house_dir = house_dir
        self.dialog = None
        img_files = glob.glob("{}/aligned_*.png".format(house_dir))
        img_files.sort()
        self.type_list = []
        self.is_closed = False
        if (os.path.exists("{}/room_types.txt".format(house_dir))):
            return
        for img in img_files:
            self.name = img.split('/')[-1][8:-4]
            self.openPano(img)
        if (self.is_closed):
            return
        with open("{}/room_types.txt".format(house_dir), 'w') as tmpfile:
            for t in self.type_list:
                tmpfile.write("pano: {} \t type: {} \n".format(t[0], t[1]))

    def keyPressEvent(self, e):
        # print(e.key())
        if (e.key() == 81):  # = q for quit
            if (e.modifiers() & QtCore.Qt.ControlModifier):
                self.is_closed = True
                self.dialog.close()
                QCoreApplication.quit()
            else:
                self.dialog.close()
        elif (e.key() > 47 and e.key() < 58):  # = numbers
            print('room assigned of type {}'.format(e.text()))
            self.type_list.append([self.name, int(e.text())])
            self.dialog.close()

    def openPano(self, image_dir):
        pano = QPixmap(image_dir)
        dialog = QtWidgets.QDialog()
        dialog.resize(self.app.MWsize[0], self.app.MWsize[0] / 1024 * 512)
        imagelabel = QtWidgets.QLabel(dialog)
        imagelabel.setPixmap(pano.scaledToWidth(self.app.MWsize[0]))
        dialog.setWindowTitle('Pano room type annotator')
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.keyPressEvent = self.keyPressEvent
        self.dialog = dialog

        label = QtWidgets.QLabel(dialog)
        label.setText(
            "0) balcony 1) closet 2) western-style-room 3) japanese-style-room 4) Dining room 5) Kitchen 6) corridor 7) washroom 8) bathroom 9)toilet"
        )
        size = [self.app.MW.width() * 0.002, 25, self.app.MWsize[0], 25]
        label.setGeometry(size[0], size[1], size[2], size[3])

        dialog.exec_()
