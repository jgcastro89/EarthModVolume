# EarthModVolume
A python powered tool that uses the support vector machine algorithm to gtenerate volumetric models from geologic cross sections.

https://pubs.usgs.gov/of/2007/1285/pdf/Smirnoff.pdf?fbclid=IwAR2nGeIJ8JcQgTYhM-H0rxoyIgNm8OxV82Fzsy_LRUhqFRmhJFlxxAJN6SQ

To start using the dashboard type "python model_gui.py" into a terminal window.

This tool requres the instalation of pyqtgraph, pandas, numpy, scipy, sklearn, and matplotlib. These packages and their dependencies are best handeled by anaconda. The following is a step by step procedure on how to install these packages.

    install anaconda (2.7) (https://www.anaconda.com/download/)
    install pyqtgraph through anaconda (https://anaconda.org/anaconda/pyqtgraph)
    install sklearn through anaconda (https://anaconda.org/anaconda/scikit-learn)
    matplotlib, numpy, scipy, and pandas should already be installed with anaconda. You can check by typing "conda list" into a terminal window.

Should you run into problems after isntalling the requered packages, the following are the exact versions of said packages. pyqtgraph 0.10.0,
pandas 0.23.1, numpy 1.15.4, scipy 1.1.0, scikit-learn 0.18.2, matplotlib 1.5.1.

A custom digitizer was developed with the Python programming language,
leveraging bindings to QT for the graphic user interface, Numpy for fast array manipulation,
Shapely generating polygons, Scipy and PIL for image processing, and Pandas for data
manipulation. The software takes cropped images of geologic cross-sections as input, then prompts the user
to assign 3 coordinates (longitude, latitude, elevation) to reference the image in 3D space. The
longitude and latitude inputs must be in decimal degrees, and elevations must be in meters. Once
the image has been referenced, the user can select an integer to represent a geologic formation
and begin drawing polygons.

Smirnoff et al. noticed the success of reconstruction for a particular class is directly
proportional to the number of those class points in the training set with the SVM algorithm
(Smirnoff et al. 2008). A solution to this problem was inspired by Monte Carlo Integration. For
each polygon that is digitized, the software will generate 10,000 random points within the
relevant polygon. Each data point contains a coordinate (longitude, latitude,
elevation) and the assigned integer used to represent the geologic formation. This newly
generated data will later serve as the training dataset for SVM, and PNN classifiers.

The project is a work in progress. I'm working on automating a few workflows, and eventually introducing the PNN algorithm as an alternative for generating volumetric models from geologic cross-sections. If you would like to have a copy of my masters thesis, please feel free to send me an email: at joey-92@live.com or jgcastro@miners.utep.edu.
