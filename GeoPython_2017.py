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
import Auxiliary3DFunctions as af
import numpy as np
import math

##############################################################################
### Create a laspy object

# This object is what you will use for the rest of the time. You can access 
# las file attributes(e.g. Classification, return number, intensity)
las, x, y, z = af.import_las("data/points.las")
print 'Type:', type(las)
print 'Size:', len(x), 'points'

##############################################################################
### Now that we have the laspy object, we can access several attributes

# Let's take a look how is the elevetation distribution.
# Plotting the histogram for the elevation
# af.plot_las(z,
#          num_bins=100,
#          color='red',
#          xlabel='Elevation (meters)',
#          ylabel='Frequency',
#          title='Elevation distribution in the LAS file')
##############################################################################
### Let's see the point cloud to identify what is wrong

# af.view_las(x, y, z, z, title='Elevation')

##############################################################################
### Let's remove some of this noise first and then plot the histogram again.

# Remove all points with elevation equal or abouve 100 meters
# mask = np.where(z>=100)
# xc, yc, zc = (np.delete(x, mask),
#               np.delete(y, mask),
#               np.delete(z, mask))

# Plotting the histogram for the elevation, now without the noise
# af.plot_las(zc,
#          num_bins=100,
#          color='yellow',
#          xlabel='Elevation (meters)',
#          ylabel='Frequency',
#          title='Elevation distribution without outlier in the LAS file')

# ##############################################################################
# ### Now, the point cloud should look fine
#
# af.view_las(xc,yc,zc,zc, title='Elevation')
#
# ##############################################################################
# ### Now, we should see if our LiDAR data is classified...
#
# # Create a new variable with the classification information
# las_class = np.array(las.classification, dtype='int8')
# las_class = np.delete(las_class, mask) #Remove the noise points
# print 'Smaller class:', min(las_class)
# print 'Bigger class:', max(las_class)
#
# ##############################################################################
# ### Now, let's the point cloud with the classification information
#
# af.view_las(x,y,z,z, title='Elevation')
#
# ##############################################################################
# ### Prepare the data to generate images
# # Point density from the metadata 24,41 points/m^2
# ideal_resolution = round(math.sqrt(1./24.41) + 0.1,2)
#
# # Get first returns for the Digital Surface Model (DSM)
# first_mask = np.where(las.return_num!=1)
# x_first, y_first, z_first = (np.delete(x, first_mask),
#                             np.delete(y, first_mask),
#                             np.delete(z, first_mask))
# print len(z_first)
#
# z_first_mask = np.where(z_first>=100)
# x_first, y_first, z_first = (np.delete(x_first, z_first_mask),
#                              np.delete(y_first, z_first_mask),
#                              np.delete(z_first, z_first_mask))
# print len(z_first)
#
# ##############################################################################
# # Plotting the histogram for the elevation, now without the noise
# af.plot_las(z_first,
#          num_bins=100,
#          color='cyan',
#          xlabel='Elevation (meters)',
#          ylabel='Frequency',
#          title='Elevation distribution from first returns')
#
# image = af.las2grid(x_first,
#                     y_first,
#                     z_first,
#                     cell=0.5,
#                     NODATA=0.0,
#                     target=r'/media/pereira/Data1/dsm_python.asc')
#
# ##############################################################################
#
# ground_mask = np.where(las.classification==2)
#
# x_ground, y_ground, z_ground = (np.array(x[ground_mask]),
#                                 np.array(y[ground_mask]),
#                                 np.array(z[ground_mask]))
# print len(z_ground)
#
# z_ground_mask = np.where(z_ground>=100)
# x_ground, y_ground, z_ground = (np.delete(x_ground, z_ground_mask),
#                                 np.delete(y_ground, z_ground_mask),
#                                 np.delete(z_ground, z_ground_mask))
# print len(z_ground)
#
# source='D:/dtm_python.asc'
#
# af.las2raster(source,
#            slopegrid = "/media/pereira/Data1/dtm_slope.asc",
#            aspectgrid = "/media/pereira/Data1/dtm_aspect.asc",
#            shadegrid = "/media/pereira/Data1/dtm_relief.asc")
#
# af.las2contour(source, "/media/pereira/Data1/dtm_contour", 1, 10)
#
# ##############################################################################
#
# af.plot_las(z_ground,
#          num_bins=100,
#          color='grey',
#          xlabel='Elevation (meters)',
#          ylabel='Frequency',
#          title='Elevation distribution from ground points')
#
# image = af.las2grid(x_ground,
#                     y_ground,
#                     z_ground,
#                     cell=0.5,
#                     NODATA=0.0,
#                     target=r'/media/pereira/Data1/dtm_python.asc')
#
# ##############################################################################
#
# source='D:/dsm_python.asc'
#
# af.las2raster(source,
#            slopegrid = "/media/pereira/Data1/slope.asc",
#            aspectgrid = "/media/pereira/Data1/aspect.asc",
#            shadegrid = "/media/pereira/Data1/relief.asc")
#
# af.las2contour(source, "/media/pereira/Data1/contour", 1, 10)
#
# af.raster2color(source, target = "/media/pereira/Data1/lidar.bmp")