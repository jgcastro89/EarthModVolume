import sys
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
from pyqtgraph.dockarea import *

from model_logi import ModelLogic


class ModelGUI(QtGui.QMainWindow):

    def __init__(self):
        """
        model gui
        Contains the graphical aspects of user interface.
        Handles construction of:
            MainWindow
            tabWidgets
            DockArea
            DockWidgets
            GridLayouts
        """

        super(ModelGUI, self).__init__()

        pg.setConfigOption('background', '#f8f8ff')
        pg.setConfigOption('foreground', 'k')

        # creating instance of ModelLogic
        self.MLogic = ModelLogic()

        # set minimum window size
        self.setMinimumSize(1000, 500)
        self.setStyleSheet('QMainWindow{background-color: lightsteelblue}')

        # dock area to contain all dock widgets
        self.model_ui = DockArea()

        self.gridLayout = QtGui.QGridLayout()           # setting up gridLayout
        self.tabsWidget = QtGui.QTabWidget()            # tab widget
        self.tabsWidget.setLayout(self.gridLayout)      # setting layout for tabs
        self.setCentralWidget(self.tabsWidget)          # setting tabs_widget to center

        # calling method to construct tabs
        self.modelTabLayout = None
        self._construct_tabs()

        # calling method to construct dock widgets
        self._model_docks()
        self._add_menu_buttons()

        # calling method to add objects to docks
        self._add_objects_to_docks()

        # adding model_ui to modelTabLayout and centering/showing window
        self.modelTabLayout.addWidget(self.model_ui)
        self.center()
        self.show()

    def _construct_tabs(self):
        """
        Constructs Qt.Gui.QWidgets (tabs)
        places widgets in corresponding tabs
        :return:
        """
        modelTab = QtGui.QWidget()
        self.modelTabLayout = QtGui.QGridLayout(modelTab)
        self.tabsWidget.addTab(modelTab, "Model")

    def _model_docks(self):
        """
        Constructs Qt.GuiDock widgets
        Dock widgets each contain their respective objects (i.e. maps, models, cross sections, etc.).
        DockWidgets are placed in DockArea (model_ui)
        :return:
        """
        self.modelMenuDock = Dock("Menu", size=(0, 1))
        self.modelVolumeDock = Dock("Volumetric Model", size=(4, 10))
        self.modelImageDock = Dock("Map View", size=(4, 4))
        self.modelCrossSectionDock = Dock("Cross-section", size=(4,4))

        self.model_ui.addDock(self.modelCrossSectionDock, 'top')
        self.model_ui.addDock(self.modelVolumeDock, 'bottom')
        self.model_ui.addDock(self.modelImageDock, 'right', self.modelVolumeDock)
        self.model_ui.addDock(self.modelMenuDock, 'left', self.modelCrossSectionDock)

    def _add_menu_buttons(self):
        """
        import Qt.Gui PushButtons from
        add push buttons to modelMenuDock
        :return:
        """
        self.modelMenuDock.addWidget(self.MLogic.LoadCloudData, row=0, col=0, colspan=3)
        # self.modelMenuDock.addWidget(self.MLogic.LoadDEMData, row=1, col=0, colspan=3)
        self.modelMenuDock.addWidget(self.MLogic.set_penalty, row=2, col=0, colspan=1)
        self.modelMenuDock.addWidget(self.MLogic.set_kernel, row=2, col=1, colspan=1)
        self.modelMenuDock.addWidget(self.MLogic.set_gamma, row=2, col=2, colspan=1)
        self.modelMenuDock.addWidget(self.MLogic.train_model, row=3, col=0, colspan=3)
        self.modelMenuDock.addWidget(self.MLogic.set_x_grid_size, row=4, col=0, colspan=1)
        self.modelMenuDock.addWidget(self.MLogic.set_y_grid_size, row=4, col=1, colspan=1)
        self.modelMenuDock.addWidget(self.MLogic.set_z_grid_size, row=4, col=2, colspan=1)
        self.modelMenuDock.addWidget(self.MLogic.predict_volumetric_model, row=5, col=0, colspan=3)
        self.modelMenuDock.addWidget(self.MLogic.NeuralNetApprox, row=6, col=0, colspan=3)

    def _add_objects_to_docks(self):
        """
        Objects are imported from volume_viewer.py
        Objects include:
            maps
            volumetric models
            plots
            tables
        :return:
        """
        self.modelVolumeDock.addWidget(self.MLogic.VM.VolumeModelView)
        self.modelImageDock.addWidget(self.MLogic.MAP.imv1)
        self.modelCrossSectionDock.addWidget(self.MLogic.MAP.cross_section)

    def center(self):
        """
        center and resize QtGui.DesktopWidget on screen
        :return:
        """
        screen = QtGui.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)


def main():
    app = QtGui.QApplication(sys.argv)
    earthModeling = ModelGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()