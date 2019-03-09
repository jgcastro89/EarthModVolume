import pdb
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import QtGui


class VolumeModel(QtGui.QMainWindow):
    def __init__(self):

        self.VolumeModelView = gl.GLViewWidget()
        self.VolumeModelView.opts['distance'] = 300

        self.y_grid = None
        self.x_grid = None
        self.z_grid = None

        self.xDim = None
        self.yDim = None
        self.zDim = None

        self.voxelModel = None
        self.dem_surf = None

    def init_model(self, xDim, yDim, zDim, voxels, dem=None):

        self._clear_view()

        self.y_grid = gl.GLGridItem()
        self.x_grid = gl.GLGridItem()
        self.z_grid = gl.GLGridItem()

        self.VolumeModelView.addItem(self.x_grid)
        self.VolumeModelView.addItem(self.y_grid)
        self.VolumeModelView.addItem(self.z_grid)

        self.xDim = xDim
        self.yDim = yDim
        self.zDim = zDim

        self._rotate_grid()
        self._blank_volume_model()
        self._scale_gird()
        self._translate_grid()

        # if dem is not None:
        #    self.generate_surface_model(dem)

        self.generate_volumetric_model(voxels)

    def _blank_volume_model(self):

        empty_data = (self.xDim/2, self.yDim/2, self.zDim/2)
        empty_data = np.zeros(empty_data)
        self.volume = np.empty(empty_data.shape + (4,), dtype=np.ubyte)

    def _scale_gird(self):

        self.x_grid.scale(self.xDim / 10, self.yDim / 10, 1)
        self.y_grid.scale(self.zDim / 20, self.yDim / 10, 1)
        self.z_grid.scale(self.xDim / 10, self.zDim / 20, 1)

    def _rotate_grid(self):

        self.y_grid.rotate(90, 0, 1, 0)
        self.z_grid.rotate(90, 90, 1, 0)

    def _translate_grid(self):

        self.y_grid.translate(-self.xDim, 0, self.zDim / 2)
        self.z_grid.translate(0, self.yDim, self.zDim / 2)

    def generate_surface_model(self, dem_surface):

        self.dem_surf = dem_surface

        for i in range(0, self.xDim):
            DEM = (dem_surface[i] * 0.3048 / 8)
            DEM = [int(j) for j in DEM]
            for j in range(len(DEM)):
                self.volume[i][j][0:DEM[j]] = [255, 139, 47, 100]

        """
        axis = gl.GLAxisItem()
        self.VolumeModelView.addItem(axis)

        voxelModel = gl.GLVolumeItem(self.volume, smooth=True, sliceDensity=1)
        voxelModel.translate(-(self.xDim / 2), -(self.yDim / 2), 0)
        self.VolumeModelView.addItem(voxelModel)
        """

    def generate_volumetric_model(self, voxels):

        self.voxels = voxels.reshape(self.xDim/2, self.yDim/2, self.zDim/2)

        self.voxels = np.kron(self.voxels, np.ones((2,2,2)))
        self.volume = np.kron(self.volume, np.ones((2,2,2,1)))

        unit0Index = np.asarray(np.where(self.voxels == 0)).T
        unit1Index = np.asarray(np.where(self.voxels == 1)).T
        unit2Index = np.asarray(np.where(self.voxels == 2)).T
        unit3Index = np.asarray(np.where(self.voxels == 3)).T
        unit4Index = np.asarray(np.where(self.voxels == 4)).T
        unit5Index = np.asarray(np.where(self.voxels == 5)).T
        unit6Index = np.asarray(np.where(self.voxels == 6)).T

        self.volume[unit0Index[:, 0], unit0Index[:, 1], unit0Index[:, 2]] = [106, 25, 205, 100]
        self.volume[unit1Index[:, 0], unit1Index[:, 1], unit1Index[:, 2]] = [160, 82, 45, 100]
        self.volume[unit2Index[:, 0], unit2Index[:, 1], unit2Index[:, 2]] = [147, 112, 216, 100]
        self.volume[unit3Index[:, 0], unit3Index[:, 1], unit3Index[:, 2]] = [70, 130, 180, 100]
        self.volume[unit4Index[:, 0], unit4Index[:, 1], unit4Index[:, 2]] = [32, 178, 170, 100]
        self.volume[unit5Index[:, 0], unit5Index[:, 1], unit5Index[:, 2]] = [0, 255, 0, 100]
        self.volume[unit6Index[:, 0], unit6Index[:, 1], unit6Index[:, 2]] = [0, 0, 0, 0]

        """
        for i in range(0, self.xDim):
            for j in range(0, self.yDim):
                for l in range(0, self.zDim):
                    elif self.voxels[i][j][l] == -1:
                        self.volume[i][j][l] = [106, 25, 205, 100]

                    if self.voxels[i][j][l] == 0:
                        self.volume[i][j][l] = [160, 82, 45, 100]

                    elif self.voxels[i][j][l] == 1:
                        self.volume[i][j][l] = [147, 112, 216, 100]
                        #self.volume[i][j][l] = [160, 82, 45, 100]

                    elif self.voxels[i][j][l] == 2:
                        self.volume[i][j][l] = [70, 130, 180, 100]
                        #self.volume[i][j][l] = [160, 82, 45, 100]

                    elif self.voxels[i][j][l] == 3:
                        self.volume[i][j][l] = [32, 178, 170, 100]

                    elif self.voxels[i][j][l] == 4:
                        self.volume[i][j][l] = [0, 255, 0, 100]

                    #elif self.voxels[i][j][l] == 5:
                    #    self.volume[i][j][l] = [100, 100, 100, 50]

                    elif self.dem_surf is None:
                        self.volume[i][j][l] = [0, 0, 0, 0]

        """
        self.voxelModel = gl.GLVolumeItem(self.volume, smooth=True, sliceDensity=1)
        self.voxelModel.translate(-(self.xDim / 2), -(self.yDim / 2), 0)
        self.VolumeModelView.addItem(self.voxelModel)

    def _clear_view(self):

        try:
            self.VolumeModelView.removeItem(self.voxelModel)
            # self.VolumeModelView.removeItem(self.dem_surf)
            self.VolumeModelView.removeItem(self.x_grid)
            self.VolumeModelView.removeItem(self.y_grid)
            self.VolumeModelView.removeItem(self.z_grid)
        except ValueError:
            pass


class Geol_Map():
    def __init__(self):

        self.imv1 = pg.ImageView()
        self.imv2 = pg.ImageView()
        self.cross_section = pg.GraphicsWindow()
        self.slice_plot = self.cross_section.addPlot()
        self.roi = pg.LineSegmentROI([[0, 50], [100, 50]], pen='r')
        self.xDim = None
        self.yDim = None
        self.image = None
        self.vox_model = None

        self.roi.sigRegionChanged.connect(self.update_cross_section)

    def init_map(self, xDim, yDim, vox_model):
        self.xDim = xDim
        self.yDim = yDim
        self.vox_model = vox_model
        self.map_view()

    def map_view(self):

        self.imv1.addItem(self.roi)
        self.imv1.setImage(self.vox_model, xvals=np.linspace(0., 25., self.vox_model.shape[2]), axes={'t': 2, 'x': 0, 'y': 1, 'c': 3})
        self.imv1.view.invertY(False)
        self.imv2.view.invertY(False)

        north_arrow = pg.ArrowItem(angle=90, tipAngle=45, baseAngle=10, headLen=50, tailWidth=8, pen='r')
        self.imv1.addItem(north_arrow)

    def update_cross_section(self):
        slice = self.roi.getArrayRegion(self.vox_model, self.imv1.imageItem, axes=(0, 1))

        self.imv2.setImage(slice)
        self.slice_plot.clear()
        self.image = pg.ImageItem(slice)
        self.slice_plot.addItem(self.image)

