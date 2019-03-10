import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from pyqtgraph import QtGui

from volume_model import VolumeModel, Geol_Map
# from surface_model import GridSurface


class ModelLogic(QtGui.QMainWindow):
    def __init__(self):
        super(ModelLogic, self).__init__()

        # creating instance of VolumeModel
        self.VM = VolumeModel()
        self.MAP = Geol_Map()

        # penalty parameter for SVM
        self._penalty = 1.0
        self._gamma = 0.003
        self._kernel = 'rbf'

        self.xDim = None
        self.yDim = None
        self.zDim = None

        self._trainingData_x = None
        self._trainingData_y = None

        self._x_grid = None
        self._y_grid = None
        self._z_grid = None
        self.gridData = None

        self.scatter_df = None
        self.dem_file = None
        self.dem_model = None

        # QPushButtons to be imported into model_gui
        self.LoadCloudData = QtGui.QPushButton('Load Point Cloud Data')
        self.NeuralNetApprox = QtGui.QPushButton('Load Existing Model')
        self.LoadDEMData = QtGui.QPushButton('Load DEM Model')
        self.set_x_grid_size = QtGui.QPushButton('Set X Grid Size')
        self.set_y_grid_size = QtGui.QPushButton('Set Y Grid Size')
        self.set_z_grid_size = QtGui.QPushButton('Set Z Grid Size')
        self.set_penalty = QtGui.QPushButton('Penalty : C')
        self.set_gamma = QtGui.QPushButton('Gamma')
        self.set_kernel = QtGui.QPushButton('Kernel')
        self.train_model = QtGui.QPushButton('Train SVM Model')
        self.predict_volumetric_model = QtGui.QPushButton('Predict Volumetric Model')
        self.set_x_grid_size.setEnabled(False)
        self.set_y_grid_size.setEnabled(False)
        self.set_z_grid_size.setEnabled(False)
        self.set_penalty.setEnabled(False)
        self.set_gamma.setEnabled(False)
        self.set_kernel.setEnabled(False)
        self.train_model.setEnabled(False)
        self.predict_volumetric_model.setEnabled(False)

        # Signals for QPushButtons
        self.LoadCloudData.clicked.connect(self._classify_data)
        self.set_penalty.clicked.connect(self._get_svm_penalty_parameter)
        self.set_gamma.clicked.connect(self._get_svm_gamma_parameter)
        self.set_kernel.clicked.connect(self._get_svm_kernel_parameter)
        self.set_x_grid_size.clicked.connect(self.get_int_attr_x)
        self.set_y_grid_size.clicked.connect(self.get_int_attr_y)
        self.set_z_grid_size.clicked.connect(self.get_int_attr_z)
        self.train_model.clicked.connect(self._train_model)
        self.predict_volumetric_model.clicked.connect(self._predict_model)
        self.LoadDEMData.clicked.connect(self._load_dem_model)

    def _classify_data(self):
        self._load_csv()
        self._get_svm_penalty_parameter()
        self._get_svm_gamma_parameter()
        self._get_svm_kernel_parameter()
        self.get_int_attr_x()
        self.get_int_attr_y()
        self.get_int_attr_z()
        self._stack_training_data()

        self.set_x_grid_size.setEnabled(True)
        self.set_y_grid_size.setEnabled(True)
        self.set_z_grid_size.setEnabled(True)
        self.set_penalty.setEnabled(True)
        self.set_gamma.setEnabled(True)
        self.set_kernel.setEnabled(True)
        self.train_model.setEnabled(True)
        self.predict_volumetric_model.setEnabled(True)

    def _load_csv(self):

        fileName = QtGui.QFileDialog.getOpenFileName(parent=self, caption='OpenFile')

        self.scatter_df = pd.read_csv(str(fileName), header=0)

    def _load_dem_model(self):
        dem_directory = QtGui.QFileDialog.getOpenFileName(parent=self, caption="OpenFile")

        self.dem_file = str(dem_directory)

    def _get_svm_penalty_parameter(self):

        num, ok = QtGui.QInputDialog.getDouble(self, "Penalty Parameter for SVM", "Enter floating point number",
                                               1.0, 0.000000000001, 1000000, 12)

        if num > 0 and ok:
            self._penalty = num
            self.set_penalty.setText('C = ' + str(num))
        else:
            self._get_svm_penalty_parameter()

    def _get_svm_gamma_parameter(self):

        num, ok = QtGui.QInputDialog.getDouble(self, "Gamma Parameter for SVM", "Enter floating point number",
                                               0.0003, 0.000000000001, 1000000, 12)

        if num > 0 and ok:
            self._gamma = num
            self.set_gamma.setText('Gamma = ' + str(num))
        else:
            self._get_svm_gamma_parameter()

    def _get_svm_kernel_parameter(self):

        kernels = ("rbf", "linear", "poly", "sigmoid", "precomputed")

        kernel, ok = QtGui.QInputDialog.getItem(self, "Kernel Parameter for SVM", "Select a Kernel", kernels, 0, False)

        self._kernel = kernel
        self.set_kernel.setText('Kernel = ' + kernel)

    def get_int_attr_x(self):
        """
        This method assigns an integer value for the x-axis grid size. The value is stored in set_x_grid_size.
        Modifications are needed for when a user clicks cancel instead of ok.
        :return:
        """
        num, ok = QtGui.QInputDialog.getInt(self, "Set Grid Size X", "Enter an Integer", 100, 100)
        input = str(num)

        if num > 99 and ok:
            self.set_x_grid_size.setText(input)
            self.xDim = num
        else:
            self.get_int_attr_x()

    def get_int_attr_y(self):
        """
        This method assigns an integer value for the y-axis grid size. Tha value is stored in set_y_grid_size.
        Modifications are needed for when a user clicks cancel.
        :return:
        """
        num, ok = QtGui.QInputDialog.getInt(self, "Set Grid Size Y", "Enter an Integer", 100, 100)
        input = str(num)

        if num > 99 and ok:
            self.set_y_grid_size.setText(input)
            self.yDim = num
        else:
            self.get_int_attr_y()

    def get_int_attr_z(self):
        """
        This method assigns an integer for the z-axis grid size. The values is stored in set_z_grid_size.
        This method is currently not in use. It's application might take place when developing volumetric models.
        :return:
        """
        num, ok = QtGui.QInputDialog.getInt(self, "Set Grid Size Z", "Enter an Integer", 25, 25)
        input = str(num)

        if num > 24 and ok:
            self.set_z_grid_size.setText(input)
            self.zDim = num
        else:
            self.get_int_attr_z()

    def _stack_training_data(self):

        self.scatter_df['Longitude'] = (self.scatter_df['Longitude'] - self.scatter_df['Longitude'].min())  * 1110
        self.scatter_df['Latitude'] = (self.scatter_df['Latitude'] - self.scatter_df['Latitude'].min())  * 1110
        self.scatter_df['Z_value'] = (self.scatter_df['Z_value']) / 10

        self._trainingData_x = np.vstack((self.scatter_df['Latitude'], self.scatter_df['Longitude']))
        self._trainingData_x = np.vstack((self._trainingData_x, self.scatter_df['Z_value'])).T

        self._trainingData_y = self.scatter_df['Classifier']

    def _train_model(self):

        self.svm_model = SVC(C=self._penalty, gamma=self._gamma, kernel=str(self._kernel), probability=True)
        self.svm_model.fit(self._trainingData_x, self._trainingData_y)

        #from neupy import algorithms
        #self.svm_model = algorithms.PNN(std=12, verbose=True)
        #self.svm_model.fit(self._trainingData_x, self._trainingData_y)

    def _reshape_data(self):

        self._y_grid = np.linspace(self.scatter_df['Longitude'].min(), self.scatter_df['Longitude'].max(), self.xDim)
        self._x_grid = np.linspace(self.scatter_df['Latitude'].min(), self.scatter_df['Latitude'].max(), self.yDim)
        self._z_grid = np.linspace(self.scatter_df['Z_value'].min(), self.scatter_df['Z_value'].max(), self.zDim)

        self._x_grid = np.asarray(self._x_grid)
        self._y_grid = np.asarray(self._y_grid)
        self._z_grid = np.asarray(self._z_grid)

    def _grid_data(self):

        self._reshape_data()

        X, Y, Z = np.meshgrid(self._x_grid, self._y_grid, self._z_grid, indexing='ij')
        X = np.dstack(X.T).reshape(-1, 1)
        Y = np.dstack(Y.T).reshape(-1, 1)
        Z = np.dstack(Z.T).reshape(-1, 1)
        self.gridData = np.hstack((X, Y))
        self.gridData = np.hstack((self.gridData, Z))

    def _generate_dem_model(self):

        if self.dem_file is not None:
            dem_model = GridSurface(self.xDim, self.yDim, self.zDim, self.scatter_df, str(self.dem_file))
            self.dem_model = dem_model.dem_surface

    def _predict_model(self):

        self._grid_data()
        # self._generate_dem_model()

        voxels = self.svm_model.predict(self.gridData)

        # self._marchineCubes(voxels)
        # self._compressModelIntoFunction(voxels)
        # self._sklearn_mlp(voxels)

        """
        SVM produces results with too much smoothing when original training data is too sparse
        here is an expiramental attempt at generating a model with harp faults by using the partial results
        from SVM model, combining that with the original training data for overburden, and running this new
        data through a probabilistic neural network.
        """

        """
        SvmBedrockIndex = np.asarray(np.where(voxels != 5))
        SvmBedrockIndex = SvmBedrockIndex.reshape(SvmBedrockIndex.size)

        OriginOverburdenIndex = np.asarray(np.where(self._trainingData_y == 5))
        OriginOverburdenIndex = OriginOverburdenIndex.reshape(OriginOverburdenIndex.size)

        #OverBurden_y = self._trainingData_y[OriginOverburdenIndex]
        #OverBurden_x = self._trainingData_x[OriginOverburdenIndex]

        #voxels = voxels[SvmBedrockIndex]
        #self.gridData = self.gridData[SvmBedrockIndex]

        PnnTrainingSet = np.vstack((self.gridData[SvmBedrockIndex], self._trainingData_x[OriginOverburdenIndex]))
        PnnClassSet = np.hstack((voxels[SvmBedrockIndex], self._trainingData_y[OriginOverburdenIndex]))

        from neupy import algorithms
        pnn = algorithms.PNN(std=10, verbose=True)
        pnn.fit(PnnTrainingSet, PnnClassSet)
        voxels = pnn.predict(self.gridData)
        """

        self.VM.init_model(self.xDim*2, self.yDim*2, self.zDim*2, voxels)

        self.MAP.init_map(self.xDim*2, self.yDim*2, self.VM.volume)

    def _compressModelIntoFunction(self, voxels):

        import tensorflow as tf
        import pdb

        mlp = tf.keras.models.Sequential([
            tf.keras.layers.Dense(100, activation=tf.nn.sigmoid, input_dim=3),
            tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
        ])

        mlp.compile(optimizer='adadelta',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        mlp.fit(self.gridData, voxels, epochs=20)

        voxels = mlp.predict_classes(self.gridData)

        pdb.set_trace()

    def _sklearn_mlp(self, voxels):

        from sklearn.neural_network import MLPClassifier
        import pdb

        mpl = MLPClassifier(hidden_layer_sizes=(20, 20, 20, 20), activation='relu',  # solver='lbfgs',
                            max_iter=600, verbose=True, learning_rate='adaptive', tol=0.0001,
                            alpha=0.0001, learning_rate_init=0.0001, )  # , learning_rate='adaptive')

        mpl.fit(self.gridData, voxels)

        pdb.set_trace()

        voxels = mpl.predict(self.gridData)

    def _marchineCubes(self, voxels):

        from skimage import measure
        import pdb

        voxels = voxels.reshape(100,100,25)

        verts, faces = measure.marching_cubes(voxels, 4, spacing=(0.5, 0.5, 0.5))

        x = verts[:, 0].astype(int)
        y = verts[:, 1].astype(int)
        z = verts[:, 2].astype(int)

        FaceColor = voxels[x, y, z]

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='colorMap', lw=0)
        plt.show()
