import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from matplotlib.mlab import griddata

import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
from pyqtgraph.Point import Point as qtPoint

from PIL import Image, ImageDraw
from resizeimage import resizeimage
from shapely.geometry import LineString, Point, shape, Polygon

from twoD_plots import BlankPlot  # , BlankPlotDark
from pandas_model import PandasModel


class DigitizerLogic(QtGui.QMainWindow):
    def __init__(self):
        super(DigitizerLogic, self).__init__()

        # creating instance of BlankPlot
        self.BP = BlankPlot()

        # creating an empty Table widget
        self.TableView = QtGui.QTableView()

        # QPushButtons to be imported by digitizer_gui.py
        self.LoadNewImageButton = QtGui.QPushButton('Load New Image')
        self.LoadFirstImageButton = QtGui.QPushButton('Load Image')
        self.SetXRange = QtGui.QPushButton('Set X Range')
        self.SetYRange = QtGui.QPushButton('Set Y Range')
        self.SetZRange = QtGui.QPushButton('Set Z Range')
        self.DrawPolygon = QtGui.QPushButton('Draw Polygon')
        self.FinishPolygon = QtGui.QPushButton('Finish Polygon')
        self.SavePoints = QtGui.QPushButton('Save Points')

        # coordinate ranges for x,y,z
        self.x_range_lon = []
        self.y_range_lat = []
        self.z_range_elv = []

        #
        self.intClassValue = None  # integer value for classification
        self.mousePoint = None  # (x,y) tuple containing pixel coordinates
        self.polygon_coordinates = []  # empty list to store coordinate points selected by user
        self.generated_data_df = None  #
        self.generated_longitude = np.array([])  #
        self.generated_latitude = np.array([])  #
        self.generated_zvalues = np.array([])  #
        self.generated_classifier = np.array([])

        # hide LoadNewImageButton at startup
        self.LoadNewImageButton.hide()
        self.FinishPolygon.setEnabled(False)

        # signals for QPushButtons
        self.LoadFirstImageButton.clicked.connect(self._open_file)
        self.LoadNewImageButton.clicked.connect(self._clear_plots)
        self.SetXRange.clicked.connect(self._set_x_range)
        self.SetYRange.clicked.connect(self._set_y_range)
        self.SetZRange.clicked.connect(self._set_z_range)
        self.DrawPolygon.clicked.connect(self._construct_new_polygon)
        self.FinishPolygon.clicked.connect(self._save_polygon)
        self.SavePoints.clicked.connect(self._save_points_to_csv)

        # signal for BlankPlot scene
        self.BP.universal_plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.BP.universal_plot.scene().sigMouseClicked.connect(self._select_points)

    def _open_file(self):
        """
        Opens a file dialog which requests user inputs.
        Input parameters are:
            file_name : image
        :return:
        """
        fileName = QtGui.QFileDialog.getOpenFileName(parent=self, caption='OpenFile')
        image = Image.open(str(fileName))
        image.load()
        image = rotate(image, -90)
        image = np.asarray(image, dtype='int32')
        image_show = pg.ImageItem(image)
        image_show.setImage(opacity=1)
        image_show.scale(1, 1)
        image_show.setZValue(-100)

        self.BP.universal_plot.addItem(image_show)
        self.LoadFirstImageButton.hide()
        self.LoadNewImageButton.show()

        self._set_x_range(image)
        self._set_y_range(image)
        self._set_z_range(image)

    def _clear_plots(self):
        """
        Clears BlankPlot, resets coordinate ranges in preperation to digitize a new cross section
        :return:
        """
        self.BP.universal_plot.clear()
        self.BP.insert_infinite_lines()
        self.x_range_lon = []
        self.y_range_lat = []
        self.z_range_elv = []
        self.polygon_coordinates = []

        self._open_file()

    def _set_x_range(self, image):
        """
        Opens a dialog requesting user inputs.
        Input parameters:
            x_range : range of x coordinates i.e (longitude)
                        initial longitude - final longitude
        :return:
        """
        x_range_min, ok = QtGui.QInputDialog.getDouble(self, "Set the initial coordinate value for X (longitude)",
                                                       "Enter a min longitude:", 000.000, -360, 360, 6
                                                       )

        x_range_max, ok = QtGui.QInputDialog.getDouble(self, "Set the final coordinate value for X (longitude",
                                                       "Enter a max longitude:", 000.000, x_range_min, 360, 6
                                                       )
        self.x_range_lon = np.linspace(x_range_min, x_range_max, len(image))

    def _set_y_range(self, image):
        """
        Opens a dialog requesting user inputs.
        Input parameters:
            y_range : range of y coordinates i.e (latitude)
                        initial latitude - final latitude
        :return:
        """
        y_range_min, ok = QtGui.QInputDialog.getDouble(self, "Set the initial coordinate value for Y (latitude)",
                                                       "Enter a min latitude:", 000.000, -360, 360, 6
                                                       )

        self.y_range_lat.append(y_range_min)

        y_range_max, ok = QtGui.QInputDialog.getDouble(self, "Set the final coordinate value for Y (latitude)",
                                                       "Enter a max latitude:", 000.000, -360, 360, 6
                                                       )
        self.y_range_lat.append(y_range_max)

    def _set_z_range(self, image):
        """
        Opens a dialog requesting user inputs.
        Input parameters:
            z_range : range of y coordinates
                        initial z - final z
        :return:
        """
        z_range_min, ok = QtGui.QInputDialog.getDouble(self, "Set the initial coordinate value for Z",
                                                       "Enter a min Z:", 000.000, -10000000, 10000000, 6
                                                       )

        z_range_max, ok = QtGui.QInputDialog.getDouble(self, "Set the final coordinate value for Z",
                                                       "Enter a max Z:", 000.000, z_range_min, 10000000, 6
                                                       )
        self.z_range_elv = np.linspace(z_range_min, z_range_max, len(image[0]))

    def _construct_new_polygon(self):
        """
        Opens a dialog requesting user input.
        Input parameters:
            classifier : and integer value which will be associated with polygon to be drawn
        :return:
        """

        integer, ok = QtGui.QInputDialog.getInt(self, "Set a Classifier value", "Set an int value for polygon")

        if ok:
            self.intClassValue = integer
            self.FinishPolygon.setEnabled(True)

    def _mouse_moved(self, pos):
        """
        Updates the location of cross-hairs (vLine/hLine) when mouse moves over the plot scene.
        ========== Parameters ==========
        pos : tuple
            A position tuple containing the x and y pixel coordinates of mouse pointer
        :param pos:
        :return:
        """
        vb = self.BP.universal_plot.vb
        if self.BP.universal_plot.sceneBoundingRect().contains(pos):
            self.mousePoint = vb.mapSceneToView(pos)

            self.BP.vLine.setPos(self.mousePoint.x())
            self.BP.hLine.setPos(self.mousePoint.y())

            # updating coordinate values shown on plot
            self.BP.coordinate_labels.setText("<span style='color: crimson'> x=%1i :"
                                              "<span style='color: crimson'> y=%1i </span>"
                                              % (self.mousePoint.x(), self.mousePoint.y()))

    def _select_points(self):
        """
        Stores the selected (x, y) pixel vertices into a list.
        intClassValue must be set otherwise no action will be taken
        :return:
        """
        if self.intClassValue is not None:
            self.polygon_coordinates.append(self.mousePoint.x())
            self.polygon_coordinates.append(self.mousePoint.y())

            points = np.array([[self.mousePoint.x()],
                               [self.mousePoint.y()]
                               ])

            # plots selected points in real time
            self._plot_scatter_points(points)

            if len(self.polygon_coordinates) >= 2:
                self._plot_line(self.polygon_coordinates)

    def _save_polygon(self):
        """
        Constructs a polygon from polygon_coordinates (list).
        Generate random points within said polygon and store the results into a list
        :return:
        """
        self.polygon_coordinates = np.asarray(self.polygon_coordinates).reshape(-1, 2)
        polygon_object = Polygon(self.polygon_coordinates)

        self._plot_polygon(polygon_object)
        self._update_table(polygon_object)

        self.intClassValue = None

        self.polygon_coordinates = []

    def _save_points_to_csv(self):
        """
        Saves generated data in dataframe to a csv file in user specified directory
        :return:
        """
        #directory_name = QtGui.QFileDialog.getExistingDirectory(self, "Select Directory")
        self.generated_data_df.to_csv("volumetricData.csv", index=True, sep=',')

    def _plot_scatter_points(self, points):
        """
        Plots scatter points onto blank plot to be displayed above image
        :param points:
        :return:
        """
        self.BP.universal_plot.plot(points[0], points[1], pen=None, symbolPen=None, symbolSize=10,
                                    symbolBrush=(220, 20, 60, 100))


        # x_axis_range = np.linspace(self.x_range_lon[0], self.x_range_lon[1], 950)
        # ticks = [list(zip(range(950), x_axis_range))]
        # x_axis = self.BP.universal_plot.getAxis('top')
        # x_axis.setTicks(ticks)
        # self.BP.universal_plot.showAxis('top', show=True)

    def _plot_line(self, points):
        """
        Plots a line plot from onto black plot to be displayed above image
        :param points:
        :return:
        """
        points = np.asarray(points).reshape(-1, 2).T

        self.BP.universal_plot.plot(points[0], points[1], pen=(220, 20, 60, 100))

    def _plot_polygon(self, polygon_object):
        """
        Plots polygon_object on plot and displays polygon over image
        :param polygon_object:
        :return:
        """
        x_vertices, z_vertices = polygon_object.exterior.xy

        fr = pg.PolyLineROI(np.vstack((x_vertices, z_vertices)).T, closed=True, pen=(0, 0, 255, 100), movable=False,
                            removable=False)

        self.BP.universal_plot.addItem(fr)

    def _update_table(self, polygon_object):
        """
        Displays the final/true data points on table
        Before displaying the data, we must first generate N number of random points withing
            the polygon.
        Then we map the x-axis pixel coordinates to decimal degree longitude coordinates.
        Next we take the newly generated longitude coordinates and compute their corresponding latitude
            coordinates using user inputs for longitude and latitude ranges and the equation of a line.
        Next we map the z-axis pixel coordinates to meters.
        Finally we generate a dataframe containing the processed data points and display them.
        :param polygon_object: Polygon to be used for generating N random points within.
        :return:
        """
        random_points = np.asarray([self.random_points_inside_polygon(polygon_object) for j in range(1000)]).T

        lon = [self.x_range_lon[random_points[0][j]] for j in range(len(random_points[0]))]
        z_value = [self.z_range_elv[random_points[1][j]] for j in range(len(random_points[1]))]
        classifier = [self.intClassValue] * 1000

        lat = self.compute_latitudes(self.x_range_lon[0], self.x_range_lon[-1],
                                     self.y_range_lat[0], self.y_range_lat[1],
                                     lon)

        self.generated_longitude = np.hstack((self.generated_longitude, lon))
        self.generated_latitude = np.hstack((self.generated_latitude, lat))
        self.generated_zvalues = np.hstack((self.generated_zvalues, z_value))
        self.generated_classifier = np.hstack((self.generated_classifier, classifier))

        self.generated_data_df = pd.DataFrame({'Longitude': self.generated_longitude,
                                               'Latitude': self.generated_latitude,
                                               'Z_value': self.generated_zvalues,
                                               'Classifier': self.generated_classifier
                                               })

        model = PandasModel(self.generated_data_df)
        self.TableView.setModel(model)
        self.TableView.resizeColumnsToContents()

    def _plot_random_points(self, polygon_object):
        """
        Plot N random points within polygon on secondary plot
        :param x_vertices: pixel vertices for x-axis
        :param z_vertices: pixel vertices for z-axis
        :return:
        """
        x_vertices, z_vertices = polygon_object.exterior.xy

        x_vertices, z_vertices = self.displace_vertices(x_vertices, z_vertices, np.min(self.x_range_lon),
                                                        np.min(self.z_range_elv))

        random_points = np.asarray([self.random_points_inside_polygon(polygon_object) for j in range(10000)])

        """
        self.BP_DarkPlot.universal_plot.plot(random_points.T[0]+np.min(self.x_range_lon) , random_points.T[1]+np.min(self.z_range_elv), pen=None,
                                             symbolPen=None, symbolSize=10, symbolBrush=(255, 255, 0, 100))

        self.BP_DarkPlot.universal_plot.plot(x_vertices, z_vertices, pen=(70, 130, 180), symbolPen=(0, 0, 0, 100),
                                             symbolSize=10, symbolBrush=(0, 255, 255, 100))
        """

    # static methods are located here
    # Methods beyond this point can be imported and used for other applications

    @staticmethod
    def random_points_inside_polygon(polygon_object):
        """
        This function takes in a polygon object and generates a specified number of random
        points (integer) within the given polygon.
        :return:
        """
        min_x, min_y, max_x, max_y = polygon_object.bounds

        while True:
            x = np.random.random_integers(min_x, max_x)
            y = np.random.random_integers(min_y, max_y)
            rand_point = Point([x, y])
            if rand_point.within(polygon_object):
                return qtPoint([x, y])

    @staticmethod
    def compute_latitudes(min_x, max_x, min_y, max_y, longitude_list):
        """
        Computes the corresponding latitude values give user inputs for minimum/maximum decimal degree
            longitude/latitude coordinates for specified image.
        :param max_y: user input for max latitude coordinate in decimal degrees
        :param min_y: user input for min latitude coordinate in decimal degress
        :param min_x: user input for minimum longitude coordinate in decimal degrees
        :param max_x: user input for maximum longitude coordinate in decimal degrees
        :param longitude_list: polygon x-axis vertices in decimal degrees
        :return:
        """
        lat = []

        slope = (max_y - min_y) / (max_x - min_x)

        for i in range(len(longitude_list)):
            result = slope * longitude_list[i] - slope*min_x + min_y
            lat.append(result)

        return np.asarray(lat)

    @staticmethod
    def displace_vertices(x_vertices, z_vertices, x_min=0, z_min=0):
        """
       Adds vertical and horizontal displacement to x_vertices and z_vertices to
       illustrate polygons in real space as opposed to pixel values.
       :param x_vertices: polygon vertices for x-axis
       :param z_vertices: polygon vertices for z-axis
       :param x_min: minimum longitude value from user input
       :param z_min: minimum z value from user input
       :return:
       """
        for i in range(len(x_vertices)):
            x_vertices[i] += x_min

        for i in range(len(z_vertices)):
            z_vertices[i] += z_min

        return x_vertices, z_vertices
