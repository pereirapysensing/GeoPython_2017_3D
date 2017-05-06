# -*- coding: utf-8 -*-
'''
Working with 3D point clouds with Python: GeoPython 2017, Basel - Switzerland

@author:  Joao Paulo Pereira
          University of Freiburg
          Chair of Remote Sensing and Landscape Information Systems - FeLis                  
                    ---------------------------------------

This material was developed especificaly for the workshop "Working with 
3D point clouds with Python" presented at the GeoPython 2017 conference.

The goal here is to present the basics on 3D point cloud processing using
Python programming language. Along this workshop, participants will have
contact with the following topics:
    - where can one download free 3D point cloud data
    - where 3D point clouds come from
    - how to import LAS files and use it as a numpy array
    - how to manipulate LAS files in python
    - how to visualize 3D point clouds
    - how to produce images from 3D point clouds

In case you whish to contact me, please send me an e-mail:
            joao.pereira@felis.uni-freiburg.de
            pereira.jpa00@gmail.com
###############################################################################
####################                                       ####################
####################              INSTRUCTIONS             ####################
####################                                       ####################
###############################################################################
            
This script must be accompanied by the Auxiliary3DFunctions.py. If that is not
the case, this script will not work. In addition, lines starting with '#!!!'
were diactivated on purpose. If you wish to visualize the 3D point clouds,
please reactivate the lines by removing the '#!!!'.

Have fun  =) !!!!
'''
# Import libraries
import os
os.chdir("F:\Joao\MEGA\Publications\GeoPython 2017\Workshop\GeoPython_2017_3D".replace("\\","/"))
try:
    import Auxiliary3DFunctions as af
except ImportError:
    print """Auxiliary3DFunctions was not found. Please contact 
    João Paulo Pereira at joao.pereira@felis.uni-freiburg.de"""
import math
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
### Create a laspy object

# This object is what you will use for the rest of the time. You can access 
# las file attributes(e.g. Classification, return number, intensity)
las, x, y, z = af.import_las("data/points.las")
print 'Data type: {}'.format(type(las))
print 'Point cloud size: {} points'.format(len(x))

##############################################################################
### Now that we have the laspy object, we can access several attributes

# Let's take a look how is the elevetation distribution.
# Plotting the histogram for the elevation
af.plot_las(z,
            num_bins=100,
            color='red',
            xlabel='Elevation (meters)',
            ylabel='Frequency',
            title='Elevation distribution in the LAS file')
###############################################################################
#### Let's see the point cloud to identify what is wrong
#
af.view_las(x, y, z, z, title='Elevation')
#
###############################################################################
#### Let's remove some of this noise first and then plot the histogram again.
#
## Remove all points with elevation equal or abouve 100 meters
mask = np.where(z>=100)
xc, yc, zc = (np.delete(x, mask),
              np.delete(y, mask),
              np.delete(z, mask))
#
## Plotting the histogram for the elevation, now without the noise
af.plot_las(zc,
            num_bins=100,
            color='orange',
            xlabel='Elevation (meters)',
            ylabel='Frequency',
            title='Elevation distribution without outlier in the LAS file')
#
## ##############################################################################
## ### Now, the point cloud should look fine
##
af.view_las(xc,yc,zc,zc, title='Elevation')
##
## ##############################################################################
## ### Now, we should see if our LiDAR data is classified...
##
## # Create a new variable with the classification information
las_class = np.array(las.classification, dtype='int8')
las_class = np.delete(las_class, mask) #Remove the noise points
print 'Smaller class (Unclassified):', min(las_class)
print 'Bigger class (Noise):', max(las_class)
##
## ##############################################################################
## ### Now, let's the point cloud with the classification information
##
af.view_las(xc,yc, zc, las_class, title='Classification')
##
## ##############################################################################
## ### Prepare the data to generate images
## # Point density from the metadata 24,41 points/m^2
##
## Get first returns for the Digital Surface Model (DSM)
first_mask = np.where(las.return_num!=1)
x_first, y_first, z_first = (np.delete(x, first_mask),
                             np.delete(y, first_mask),
                             np.delete(z, first_mask))
print len(z_first)
#
z_first_mask = np.where(z_first>=100)
x_first, y_first, z_first = (np.delete(x_first, z_first_mask),
                             np.delete(y_first, z_first_mask),
                             np.delete(z_first, z_first_mask))
print len(z_first)
##
## ##############################################################################
## # Plotting the histogram for the elevation, now without the noise
af.plot_las(z_first,
         num_bins=100,
         color='brown',
         xlabel='Elevation (meters)',
         ylabel='Frequency',
         title='Elevation distribution from first returns')
##
##############################################################################
# Now to calculate the DSM we need first to find the correct spatial resolution
# using the point density information from the LAS file metadata (24,41 points/m^2)
ideal_resolution = round(math.sqrt(1./24.41) + 0.1,2)

# Then we apply the las2grid function to calculte the dsm
image = af.las2grid(x_first,
                    y_first,
                    z_first,
                    cell=ideal_resolution,
                    NODATA=0.0,
                    target=r'F:/geopython/dsm_python.asc')
#
###############################################################################
# Let's work now with the digital terrain model (DTM).
#First we need to filter the point cloud and select only points with
# classification 2 (ground)

# creates mask for classification 2                    
ground_mask = np.where(las.classification==2)

#apply mask to data
x_ground, y_ground, z_ground = (x[ground_mask],
                                y[ground_mask],
                                z[ground_mask])

af.plot_las(z_ground,
            num_bins=100,
            color='grey',
            xlabel='Elevation (meters)',
            ylabel='Frequency',
            title='Elevation distribution from ground points')

# we need the dsm to get the geoinformation
source='F:/geopython/dsm_python.asc'

# calculate the dtm using the function lasinterpolated
dtm = af.lasinterpolated(source,
                         "F:/geopython/dtm_python.tif",
                         x_ground,
                         y_ground,
                         z_ground, 
                         method='nearest', 
                         fill_value=0.0,
                         EPSG=26910,
                         contour=False, 
                         plot=True)
###############################################################################
# We can also calculate other products like slope, aspect and shaded images.
products = af.las2raster(source,
                         slopegrid = "F:/geopython/slope.asc",
                         aspectgrid = "F:/geopython/aspect.asc",
                         shadegrid = "F:/geopython/relief.asc")

# from the dsm we can calculate contour lines
af.las2contour(source, "F:/geopython/contour", 1, 10)
#
#################################################################################            
# Also create RGB images using different products combinations
af.raster2color(source,
                "F:/geopython/relief.asc",
                "F:/geopython/slope.asc",
                target="F:/geopython/colored.tif", 
                EPSG=26910)
"""
###############################################################################
#####################                                    ######################
#####################              THANK YOU             ######################
#####################                                    ######################
###############################################################################
"""
