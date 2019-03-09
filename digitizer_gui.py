import sys

from pyqtgraph import QtGui, QtCore
from pyqtgraph.dockarea import *

from digitizer_logi import DigitizerLogic


class DigitizerGUI(QtGui.QMainWindow):

    def __init__(self):
        """
        digitizer gui:
        Contains the graphical aspects of the user interface.
        Handles construction of:
            MainWindow
            tabWidgets
            DockArea
            DockWidgets
            GridLayouts
        Does not construct QPushButtons. QPushButtons and their respective logic are
         imported from a separate python file/class.
        """
        super(DigitizerGUI, self).__init__()

        # creating instance of DigitizerLogic
        self.DLogic = DigitizerLogic()

        # set minimum window size
        self.setMinimumSize(1000, 500)

        # dock areas to contain all dock widgets
        self.digitizer_ui = DockArea()

        self.gridLayout = QtGui.QGridLayout()           # setting up gridLayout
        self.tabsWidget = QtGui.QTabWidget()            # tab widget
        self.tabsWidget.setLayout(self.gridLayout)      # setting layout for tabs
        self.setCentralWidget(self.tabsWidget)          # setting tabs_widget to center

        # calling method to construct tabs
        self.digitizerTabLayout = None                   # Qt.Gui.GridLayout for digitizer_tab
        self._construct_tabs()                           # calling method to construct tabs_widgets

        # calling method to construct dock widgets
        self._digitizer_docks()
        self._add_menu_buttons()

        # calling method to add objects to docks
        self._add_objects_to_docks()

        self.digitizerTabLayout.addWidget(self.digitizer_ui)
        self.setStyleSheet('QMainWindow{background-color: lightsteelblue}')
        self.center()
        self.show()

    def _construct_tabs(self):
        """
        constructs Qt.Gui.QWidgets (tabs)
        places widgets in corresponding tabs
        :return:
        """
        # digitizerTab: will contain all DockWidgets related to digitizer
        digitizerTab = QtGui.QWidget()
        modelTab = QtGui.QWidget()
        self.digitizerTabLayout = QtGui.QGridLayout(digitizerTab)
        self.tabsWidget.addTab(digitizerTab, "Digitizer")
        self.tabsWidget.addTab(modelTab, "Model")

        # model_tab = QtGui.QWidget()

    def _digitizer_docks(self):
        """
        constructs Qt.GuiDock widgets
        Dock widgets each contain their respective objects
        Objects can be buttons, maps, volumetric models, images, etc.
        DockWidgets are placed in DockArea (digitizer_ui)
        :return:
        """
        self.digitizerMenuDock = Dock("Menu", size=(1, 1))
        self.digitizerImageDock = Dock("Digitizer", size=(4, 10))
        self.digitizerDataFrameDock = Dock("Polygons DataFrame", size=(4, 4))

        self.digitizer_ui.addDock(self.digitizerImageDock, 'bottom')
        self.digitizer_ui.addDock(self.digitizerMenuDock, 'top', self.digitizerImageDock)
        self.digitizer_ui.addDock(self.digitizerDataFrameDock, 'right', self.digitizerMenuDock)

    def _model_docks(self):
        """
        constructs Qt.GuiDock widgets for model dock
        Dock widgets each contain their repsective objects
        Objects can be buttons, maps, volumetric models, images, ect.
        DockWidgets are placed in DockArea (model_ui)
        :return:
        """
        self.modelMenuDock = Dock("Menu", size=(1, 1))
        self.modelVolumeDock = Dock("Volumetric Model", size=(4, 10))
        self.modelImageDock = Dock("Map View", size=(4, 4))
        self.modelCrossSectionDock = Dock("Cross-section", size=(4,4))

    def _add_menu_buttons(self):
        """
        import Qt.Gui PushButtons from digitizer_logi.py
        Add pushButtons to digitizerMenuDock
        :return:
        """
        self.digitizerMenuDock.addWidget(self.DLogic.LoadFirstImageButton, row=0, col=0, colspan=3)
        self.digitizerMenuDock.addWidget(self.DLogic.LoadNewImageButton, row=1, col=0, colspan=3)
        # self.digitizerMenuDock.addWidget(self.DLogic.SetXRange, row=2, col=0, colspan=1)
        # self.digitizerMenuDock.addWidget(self.DLogic.SetYRange, row=2, col=1, colspan=1)
        # self.digitizerMenuDock.addWidget(self.DLogic.SetZRange, row=2, col=2, colspan=1)
        self.digitizerMenuDock.addWidget(self.DLogic.DrawPolygon, row=3, col=0, colspan=3)
        self.digitizerMenuDock.addWidget(self.DLogic.FinishPolygon, row=4, col=0, colspan=3)
        self.digitizerMenuDock.addWidget(self.DLogic.SavePoints, row=5, col=0, colspan=3)

    def _add_objects_to_docks(self):
        """
        Objects are imported from digitizer_logi.py
        Objects include:
            maps
            volumetric models
            plots
            tables
        :return:
        """
        self.digitizerImageDock.addWidget(self.DLogic.BP.plot_view)
        self.digitizerDataFrameDock.addWidget(self.DLogic.TableView)
        # self.digitizerDataFrameDock.addWidget(self.DLogic.BP_DarkPlot.plot_view)

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
    earthModeling = DigitizerGUI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
