from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QProgressDialog
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QIcon, QPixmap
import qdarkstyle
import os, sys
from utils.fileListWidget import FileListWidget
from utils.flagListWidget import FlagListWidget
from utils.typeListWidget import TypeListWidget
from utils.imageListWidget import ImageListWidget
from utils.roomViewWidget import RoomViewWidget
from utils.floorPlanWidget import FloorPlanWidget
from utils.house import House
from screeninfo import get_monitors
from utils.room_type_annotator import Room_type_annotator
from PanoAnnotator.PanoAnnotator import PanoAnnotator


class Ui_MainWindow():
    def setupUi(self, MainWindow):
        monits = get_monitors()
        self.MWsize = [monits[0].width * 0.8, monits[0].height * 0.8]
        MainWindow.setObjectName("MainWindow")
        self.MW = MainWindow
        self.panoAnnotator = PanoAnnotator(self, self.MW)
        MainWindow.resize(self.MWsize[0], self.MWsize[1])

        self.view_fp = True
        self.view_door_colors = True
        self.view_room_colors = True
        self.ignore_rotations = False
        self.ignore_centers = False

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.keyPressEvent = self.keyPressEvent
        '''
        self.fpmap = QtWidgets.QLabel(self.centralwidget)
        self.fpmap.setGeometry(QtCore.QRect(1200, 10, 600, 800))
        self.fpmap.setObjectName("fpmap")
        '''

        self.house = House()
        self.folders_layout = FileListWidget(MainWindow, self)
        self.flag_layout = FlagListWidget(MainWindow, self, self.house)
        self.type_layout = TypeListWidget(MainWindow, self, self.house)
        self.image_layout = ImageListWidget(MainWindow, self, self.house)
        self.room_layout = RoomViewWidget(MainWindow, self, self.house)
        self.fp_layout = FloorPlanWidget(MainWindow, self, self.house)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, self.MWsize[0], 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        MainWindow.statusBar()

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")

        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")

        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")

        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")

        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")

        self.actionEdit = QtWidgets.QAction(MainWindow)
        self.actionEdit.setObjectName("actionEdit")

        self.actionViewFP = QtWidgets.QAction(MainWindow)
        self.actionViewFP.setObjectName("actionViewFP")
        self.actionViewDC = QtWidgets.QAction(MainWindow)
        self.actionViewDC.setObjectName("actionViewDC")
        self.actionViewRC = QtWidgets.QAction(MainWindow)
        self.actionViewRC.setObjectName("actionViewRC")
        self.actionViewIC = QtWidgets.QAction(MainWindow)
        self.actionViewIC.setObjectName("actionViewIC")
        self.actionViewIR = QtWidgets.QAction(MainWindow)
        self.actionViewIR.setObjectName("actionViewIR")

        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExit)

        self.menuEdit.addAction(self.actionEdit)

        self.menuView.addAction(self.actionViewFP)
        self.menuView.addAction(self.actionViewRC)
        self.menuView.addAction(self.actionViewDC)
        self.menuView.addAction(self.actionViewIC)
        self.menuView.addAction(self.actionViewIR)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        '''
        self.label.setText(_translate("MainWindow", "Reference Image"))
        self.label_2.setText(_translate("MainWindow", "Target Image"))

        self.pushButton.setText(_translate("MainWindow", "Show"))
        self.pushButton.setShortcut(_translate("MainWindow", "Ctrl+q"))
        self.pushButton.clicked.connect(self.show_func)
        '''

        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuView.setTitle(_translate("MainWindow", "View"))

        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+o"))
        self.actionOpen.triggered.connect(self.openDir)

        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+s"))
        self.actionSave.triggered.connect(self.saveDir)

        self.actionEdit.setText(_translate("MainWindow", "Edit"))
        self.actionEdit.setShortcut(_translate("MainWindow", "Ctrl+e"))
        self.actionEdit.triggered.connect(self.editLayout)

        self.actionViewFP.setText(_translate("MainWindow", "ViewFP"))
        self.actionViewFP.setShortcut(_translate("MainWindow", "Ctrl+v"))
        self.actionViewFP.triggered.connect(self.editViewFP)

        self.actionViewRC.setText(_translate("MainWindow", "ViewRC"))
        self.actionViewRC.setShortcut(_translate("MainWindow", "Ctrl+r"))
        self.actionViewRC.triggered.connect(self.editViewRC)

        self.actionViewIC.setText(_translate("MainWindow", "ViewIC"))
        self.actionViewIC.setShortcut(_translate("MainWindow", "Alt+c"))
        self.actionViewIC.triggered.connect(self.editViewIC)

        self.actionViewIR.setText(_translate("MainWindow", "ViewIR"))
        self.actionViewIR.setShortcut(_translate("MainWindow", "Alt+r"))
        self.actionViewIR.triggered.connect(self.editViewIR)

        self.actionViewDC.setText(_translate("MainWindow", "ViewDC"))
        self.actionViewDC.setShortcut(_translate("MainWindow", "Ctrl+d"))
        self.actionViewDC.triggered.connect(self.editViewDC)

        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+q"))
        self.actionExit.triggered.connect(QCoreApplication.quit)

        self.file_path = 'clean_data'
        self.folders_layout.update(self.file_path)
        self.current_index = -1
        self.MW.setFocus()

    def update_ui(self):
        self.flag_layout.update()
        self.room_layout.update()
        self.image_layout.update()
        self.fp_layout.update()
        self.type_layout.update()

        self.flag_layout.updateUI()
        self.room_layout.updateUI()
        self.image_layout.updateUI()
        self.fp_layout.updateUI()
        self.type_layout.updateUI()
        self.folders_layout.updateUI()

    def saveDir(self):
        self.house.save_data()

    def openDir(self):
        filePath = QFileDialog.getExistingDirectory(self.MW, "open",
                                                    os.getcwd())
        if (len(filePath) == 0):
            return
        self.folders_layout.update(filePath)
        self.update_ui()
        self.file_path = filePath

    def editViewFP(self):
        self.view_fp = not self.view_fp
        self.update_ui()

    def editViewDC(self):
        self.view_door_colors = not self.view_door_colors
        self.update_ui()

    def editViewRC(self):
        self.view_room_colors = not self.view_room_colors
        self.update_ui()

    def editViewIC(self):
        self.ignore_centers = not self.ignore_centers
        self.update_ui()

    def editViewIR(self):
        self.ignore_rotations = not self.ignore_rotations
        self.update_ui()

    def editLayout(self):
        print("editing {}/{}/aligned_{}.png".format(self.house.dir,
                                                    self.house.house_name,
                                                    self.house.current_pano))
        self.panoAnnotator.run("{}/{}/aligned_{}.png".format(
            self.house.dir, self.house.house_name, self.house.current_pano))
        self.update_ui()

    def update_folder(self, index):
        self.MW.statusBar().showMessage("loading folder " +
                                        str(self.house.house_name))
        self.current_index = index.row()
        if (not os.path.exists(
                os.path.join(self.file_path, index.data(), "room_types.txt"))):
            Room_type_annotator(self, "{}/{}".format(self.file_path,
                                                     index.data()))

        self.house.reset(self.file_path, index.data(), self.ignore_rotations, self.ignore_centers)
        self.update_ui()
        self.MW.setFocus()

    def update_image(self, image_indx):
        self.house.current_pano = int(image_indx[5:])
        self.MW.statusBar().showMessage(
            "loading image {} ...".format(image_indx))
        self.room_layout.update()

    def keyPressEvent(self, e):
        #print(e.key())
        if (QtWidgets.QApplication.keyboardModifiers() ==
                QtCore.Qt.ControlModifier):
            self.flag_layout.update()
            self.house.update_flag(e.key() - 48)

        if (QtWidgets.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier
            ):
            self.type_layout.update()
            self.house.set_type(e.key() - 48)

        if (e.key() == 82):  # = r for rotate
            self.house.rotate()
            self.MW.statusBar().showMessage("rotating the image...")

        elif (e.key() == 78):  # = n for next
            if (QtWidgets.QApplication.keyboardModifiers() ==
                    QtCore.Qt.ShiftModifier):
                item = self.folders_layout.get_row(self.current_index + 1)
                self.update_folder(item)
            else:
                if (self.house.current_pano + 1 < len(self.house.pano_list)):
                    self.house.current_pano += 1

        elif (e.key() == 80):  # = p for prev
            if (QtWidgets.QApplication.keyboardModifiers() ==
                    QtCore.Qt.ShiftModifier):
                if (self.current_index > 0):
                    item = self.folders_layout.get_row(self.current_index - 1)
                    self.update_folder(item)
            else:
                if (self.house.current_pano > 0):
                    self.house.current_pano -= 1

        elif (e.key() == 61 or e.key() == 43):  # = + for zoom
            self.house.zoom(is_zoom=True)

        elif (e.key() == 45 or e.key() == 95):  # = - for zoom
            self.house.zoom(is_zoom=False)

        elif (e.key() == 65):  # = a for add
            self.house.add_pano()
        elif (e.key() == 68):  # = d for del
            self.house.remove_pano()
        elif (e.key() == 72):  # = h for left
            self.house.move_pano('l')
        elif (e.key() == 74):  # = j for down
            self.house.move_pano('d')
        elif (e.key() == 75):  # = k for up
            self.house.move_pano('u')
        elif (e.key() == 76):  # = l for right
            self.house.move_pano('r')
        # elif(e.key()==81): # = q for quit

        self.update_ui()

    def openPano(self):
        dialog = QtWidgets.QDialog()
        dialog.resize(self.MWsize[0], self.MWsize[0] / 1024 * 512)
        imagelabel = QtWidgets.QLabel(dialog)
        imagelabel.setPixmap(self.house.get_original_pano().scaledToWidth(
            self.MWsize[0]))
        dialog.setWindowTitle('Pano View Window')
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.exec_()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
