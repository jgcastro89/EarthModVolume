from pyqtgraph import QtGui, QtCore


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a Qt table view with a pandas dataframe
    """

    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        self.header_labels = data.columns

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])

            if role == QtCore.Qt.BackgroundRole:
                # return self._data.Color[index.row()]
                return QtGui.QColor(QtCore.Qt.white)

            if role == QtCore.Qt.ForegroundRole:
                pass
                # return self._data.Color[index.row()]

        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.header_labels[section]
        return QtCore.QAbstractTableModel.headerData(self, section, orientation, role)