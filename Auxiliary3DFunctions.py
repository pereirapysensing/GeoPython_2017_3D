# -*- coding: utf-8 -*-
"""
Working with 3D point clouds with Python: GeoPython 2017, Basel - Switzerland

@author:  Joao Paulo Pereira
          University of Freiburg
          Chair of Remote Sensing and Landscape Information Systems - FeLis                  
                    ---------------------------------------
                    
Auxiliary3DFunctions.py is a script developed specially for the workshop 
presented at the GeoPython 2017 conference intitled 'Working with 3D point 
clouds with Python'. Some of the functions were written by the author and
others were obtained from the book

                 Learning Geospatial Analysis with Python 2ed
                 By Joel Lawhead, Dec 2015, Packt Publishing.
                 
                    ---------------------------------------
"""
# Start importing the main packages
import os
from linecache import getline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf
import colorsys
from scipy import interpolate
import gdal, ogr, osr
from mayavi.mlab import points3d, colorbar
try:
    import laspy
except ImportError:
    print ('Error, Module Laspy is required. Starting installation with pip.')
    import pip
    pip.main(['install', 'laspy'])    
try:
    import Image
    import ImageOps
except:
    from PIL import Image, ImageOps


def import_las(path, array=True):
    """
    import_las(...)
        import_las(path, array=True)
        
        Imports LAS files returning a laspy object that can be manipulated into
        numpy arrays.
        
        Parameters
        ----------
        path : string 
            String containing the LAS file.
        
        array : bool
            This function returns originaly a laspy object. However, if you want
            the laspy object and the individual arrays for x, y and z, just put 
            array=True. Default = True.
        
        Examples
        --------
        >>> las = import_las("C:/Data/points.las", array=False)
        >>> type(las)
        <class 'laspy.file.File'>
        
        >>> las, x, y, z = import_las("C:/Data/points.las", array=True)
        >>> type(las)
        <class 'laspy.file.File'>
        >>> map(type,(x,y,z))
        [<type 'numpy.ndarray'>, <type 'numpy.ndarray'>, <type 'numpy.ndarray'>]
    
    """
    # Change backslashes to slashes to avoid path issues
    las_path = path.replace('\\','/')
    # Open the LAS file
    las = laspy.file.File(las_path)
    # When array== True, the function will return the laspy object along with 
    # the 3D coordinates organized in 3 different arrays. If false, returns
    # only the laspy object.
    if array==True:
        x = las.x
        y = las.y
        z = las.z
        return las, x, y, z
    else:
        return las

def view_las(x, y, z, s, color='jet', title='Classes', classes=7, point_size=1):
    """
    view_las(...)
        view_las(x, y, z, s, color, title, classes, point_size)
        
        Return a Mayavi interactive scene using (x,y,z) to plot points of size 
        point_size, where s will be used to atribute the color to the points.
        
        Parameters
        ----------
        x : array_like, float32 or float16
            Numpay array with the East coordinates.
        
        y : array_like, float32 or float16
            Numpay array with the North coordinates. Must have the same size 
            as x.
        
        z : array_like, float32 or float16
            Numpy array with the Elevation values. Must have the same size as 
            x and y.
        
        s : array_like, float32 or float16
            Numpy array with the same size as x and y. S will be used to color 
            the point cloud. In case you want to use the Elevation, just repete 
            z like: view_las(x, y, z, z, ...).
                                  
        color :  string
            Type of colormap to use. Type string.
        
        title : string
            Title of the legend. Type string.
        
        classes : integer 
            How many classes of color should the legend show. Type integer.
        
        point_size : float
            Size of the point ploted. Type float. Default = 1.0.
    """
    vis = points3d(x, y, z, s, colormap=color, mode='point')
    vis.actor.property.set(representation='p', point_size=point_size)
    colorbar(title=title, orientation='vertical', nb_labels=classes)

def plot_las(data, num_bins=100, color='red', xlabel=None, ylabel=None, title=None):
    """
    plot_las(...)
        plot_las(data, 
                 num_bins=100, 
                 color='red', 
                 xlabel=None, 
                 ylabel=None, 
                 title=None)
        
        Plot a histogram with a normal distribution curve based on the data mean
        and standard deviation.
        
        Parameters
        ----------
        
        data: numpy_array
            Numpy array with the variable to be assessed.
        
        num_bins : integer
            Number of bins in the histogram. Default = 100.
        
        color: string.
            Color of the bins.
        
        xlabel : string.
            Title of the X axis. Default = None.
        
        ylabel : string.
            Title of the Y axis. Default = None.
        
        title : string.
            Title of the histogram. Default = None.
        
        Notes
        -----
        If xlabel, ylabel and title are not provided, the histogram will be
        plotted normally, however, without any of those elements.
    """
    num_bins = num_bins
    fig, ax = plt.subplots()
    mu = np.mean(data)
    sigma = np.std(data)
    # Creates the histogram
    n, bins, patches = ax.hist(data, 
                               num_bins, 
                               color=color, 
                               histtype='stepfilled', 
                               normed=True)
    # add a 'best fit' line
    k = normpdf(bins, mu, sigma)
    ax.plot(bins, k, '--', linewidth=3)
    if xlabel != ylabel != title != None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    plt.show()

def las2grid(x, y, z, cell=0.5, NODATA=-9999, target='C:/grid.asc'):
    """
    las2grid(...)
        plot_las(las, x, y, z, cell=0.5, NODATA=-9999, target='C:/grid.asc')
        
        Creates a grid file in ASCII format using the X, Y and Z coordinates as
        input.
        
        Parameters
        ----------
        x : array_like, float32 or float16
            Numpay array with the East coordinates.
        
        y : array_like, float32 or float16
            Numpay array with the North coordinates. Must have the same size 
            as x.
        
        z : array_like, float32 or float16
            Numpy array with the Elevation values. Must have the same size as 
            x and y.
        
        cell : float type
            Size of the pixel in the grid. Using 0.1 means having a 10cm image 
            resolution. Default = 0.5.
        
        NODATA: float or integer type.
            Value to be used to fill NODATA locations. Default = -9999.
        
        target : string type.
            Location where to save the ASCII file. Default = 'C:/grid.asc'.
        
    """
    # Get the x axis distance
    xdist = max(x) - min(x)
    # Get the y axis distance
    ydist = max(y) - min(y)
    # Number of columns for our grid
    cols = int((xdist) / cell)
    # Number of rows for our grid
    rows = int((ydist) / cell)
    cols += 1
    rows += 1
    print cols
    print rows
    # Track how many elevation
    # values we aggregate
    count = np.zeros((rows, cols)).astype(np.float32)
    # Aggregate elevation values
    zsum = np.zeros((rows, cols)).astype(np.float32)
    # Y resolution is negative
    ycell = -1 * cell
    # Project x,y values to grid
    projx = (x - min(x)) / cell
    projy = (y - min(y)) / ycell
    # Cast to integers and clip for use as index
    ix = projx.astype(np.int32)
    iy = projy.astype(np.int32)
    # Loop through x,y,z arrays, add to grid shape,
    # and aggregate values for averaging
    for x_,y_,z_ in np.nditer([ix, iy, z]):
        count[y_, x_]+=1
        zsum[y_, x_]+=z_
    # Change 0 values to 1 to avoid numpy warnings, 
    # and NaN values in array
    nonzero = np.where(count>0, count, 1)
    # Average our z values
    zavg = zsum/nonzero
    # Interpolate 0 values in array to avoid any
    # holes in the grid
    mean = np.ones((rows,cols)) * np.mean(zavg)
    left = np.roll(zavg, -1, 1)
    lavg = np.where(left>0,left,mean)
    right = np.roll(zavg, 1, 1)
    ravg = np.where(right>0,right,mean)
    interpolate = (lavg+ravg)/2
    fill1=np.where(zavg>0,zavg,interpolate)
    fill2=np.where(zavg==0,NODATA,interpolate)
    fill = fill1 + fill2
    # Create our ASCII DEM header
    header = "ncols        {}\n".format(fill.shape[1])
    header += "nrows        {}\n".format(fill.shape[0])
    header += "xllcorner    {}\n".format(min(x))
    header += "yllcorner    {}\n".format(min(y))
    header += "cellsize     {}\n".format(cell)
    header += "NODATA_value      {}\n".format(NODATA)
    # Open the output file, add the header, save the array
    with open(target, "w") as f:
        f.write(header)
        np.savetxt(f, fill, fmt="%1.2f")
    
    return fill

def lasinterpolated(x,y,z,xnumIndexes,ynumIndexes, method, fill_value, contour=False, plot=True):
    """
    lasinterpolated(...)
        lasinterpolated(x, 
                        y, 
                        z,
                        xnumIndexes,
                        ynumIndexes, 
                        method, 
                        fill_value, 
                        contour=False, 
                        plot=True)
        
        Generates an interpolated image from X, Y and Z using different methods.
        
        Parameters
        ----------
        x : array_like, float32 or float16.
            Numpay array with the East coordinates.
        
        y : array_like, float32 or float16.
            Numpay array with the North coordinates. Must have the same size 
            as x.
        
        z : array_like, float32 or float16.
            Numpy array with the Elevation values. Must have the same size as 
            x and y.
            
        xnumIndexes : integer type.
            Size fo the final image on the North axis, ie the Y in the las file.
        
        ynumIndexes :  integer type.
            Size fo the final image on the East axis, ie the X in the las file.
        
        method : {'linear', 'nearest', 'cubic'}.
            Method of interpolation. One of
    
            ``nearest``
              return the value at the data point closest to
              the point of interpolation.  See `NearestNDInterpolator` for
              more details.
        
            ``linear``
              tesselate the input point set to n-dimensional
              simplices, and interpolate linearly on each simplex.  See
              `LinearNDInterpolator` for more details.
        
            ``cubic`` (1-D)
              return the value determined from a cubic
              spline.
        
            ``cubic`` (2-D)
              return the value determined from a
              piecewise cubic, continuously differentiable (C1), and
              approximately curvature-minimizing polynomial surface. See
              `CloughTocher2DInterpolator` for more details.
            
        fill_value : float.
            Value used to fill in for requested points outside of the
            convex hull of the input points.  If not provided, then the
            default is ``nan``. This option has no effect for the
            'nearest' method.  
        
        contour : bool type.
            If True, calculates the contour based on the interpolated images.
        
        plot : bool type.
            Plots the interpolated image if set to True.
    """
    # Prepares the grid that will be used as base for the interpolation 
    xi = np.linspace(np.min(x), np.max(x),ynumIndexes)
    yi = np.linspace(np.min(y), np.max(y),xnumIndexes)
    XI, YI = np.meshgrid(xi, yi)
    
    # Set the 2D coodinates in a single array to be used in the interpolation
    points = np.vstack((x,y)).T
    values = np.asarray(z)
    points = np.asarray(points)
    
    # Interpolates the surface based on the 2D coordinates (points) and the
    # elevation values (values) using as the grid calculated before (XI, YI) as
    # base.
    DEM = interpolate.griddata(points, 
                               values, 
                               (XI,YI), 
                               method=method, 
                               fill_value=fill_value)
    # Calculates the contour lines based on the elevation values when contour
    # is set to true.
    if contour==True:
        levels = np.arange(np.min(DEM),np.max(DEM),2)
        plt.contour(DEM, levels,linewidths=0.8,colors='k')
    
    # Option to plot the final image to check the results 
    if plot==True:
        plt.imshow(DEM,cmap ='RdYlGn_r',origin='lower')
        plt.colorbar()
    
    return DEM
        
def las2raster(source, 
               slopegrid, 
               aspectgrid, 
               shadegrid, 
               azimuth=315.0, 
               altitude=45.0,
               z=1.0,
               scale=1.0,
               NODATA=-9999):  
    """
    las2raster(...)
        las2raster(source, 
               slopegrid, 
               aspectgrid, 
               shadegrid, 
               azimuth=315.0, 
               altitude=45.0,
               z=1.0,
               scale=1.0,
               NODATA=-9999)
        
        Creates three diferent products from a grid ASCII file. Use las2grid to
        produce the grid.
        
        
        Parameters
        ----------
        source : string type.
            Path of the grid file. You can use las2grid to produce the grid.
        
        slopegrid : string type.
            Path where the slope image shoud be saved.
        
        aspectgrid: string type.
            Path where the aspect image shoud be saved.
        
        shadegrid : string type.
            Path where the shade image shoud be saved.
        
        azimuth : float type.
            Azimuth to ajust the ilumination angle for the shaded image.
        
        altitude : float type.
            Angle from the artificial sun in comparison to the terrain.
        
        z : float type.
            Exaggeration in the elevation.
        
        scale : float type.
            Scale between different axis.
        
        NODATA: float or integer type.
            Value to be used to fill NODATA locations. Default = -9999.

    """
    # Needed for numpy conversions
    deg2rad = 3.141592653589793 / 180.0
    rad2deg = 180.0 / 3.141592653589793
    
    # Parse the header using a loop and
    # the built-in linecache module
    hdr = [getline(source, i) for i in range(1, 7)]
    values = [float(h.split(" ")[-1].strip()) for h in hdr]
    cols, rows, lx, ly, cell, nd = values
    xres = cell
    yres = cell * -1
    
    # Load the dem into a numpy array
    arr = np.loadtxt(source, skiprows=6)
    
    # Exclude 2 pixels around the edges which are usually NODATA.
    # Also set up structure for a 3x3 windows to process the slope
    # throughout the grid
    window = []
    for row in range(3):
        for col in range(3):
            window.append(arr[row:(row + arr.shape[0] - 2),
                          col:(col + arr.shape[1] - 2)])
    # Process each cell
    x = ((z * window[0] + z * window[3] + z * window[3] + z * window[6]) -
         (z * window[2] + z * window[5] + z * window[5] + z * window[8])) / \
        (8.0 * xres * scale)
    
    y = ((z * window[6] + z * window[7] + z * window[7] + z * window[8]) -
         (z * window[0] + z * window[1] + z * window[1] + z * window[2])) / \
         (8.0 * yres * scale)
    
    # Calculate slope
    slope = 90.0 - np.arctan(np.sqrt(x * x + y * y)) * rad2deg
    
    # Calculate aspect
    aspect = np.arctan2(x, y)
    
    # Calculate the shaded relief
    shaded = np.sin(altitude * deg2rad) * np.sin(slope * deg2rad) + \
         np.cos(altitude * deg2rad) * np.cos(slope * deg2rad) * \
         np.cos((azimuth - 90.0) * deg2rad - aspect)
    shaded = shaded * 255
    
    # Rebuild the new header
    header = "ncols        {}\n".format(shaded.shape[1])
    header += "nrows        {}\n".format(shaded.shape[0])
    header += "xllcorner    {}\n".format(lx + (cell * (cols - shaded.shape[1])))
    header += "yllcorner    {}\n".format(ly + (cell * (rows - shaded.shape[0])))
    header += "cellsize     {}\n".format(cell)
    header += "NODATA_value      {}\n".format(NODATA)
    
    # Set no-data values
    for pane in window:
        slope[pane == nd] = NODATA
        aspect[pane == nd] = NODATA
        shaded[pane == nd] = NODATA
    
    # Open the output file, add the header, save the slope grid
    with open(slopegrid, "wb") as f:
        f.write(header)
        np.savetxt(f, slope, fmt="%4i")
    
    # Open the output file, add the header, save the aspect grid
    with open(aspectgrid, "wb") as f:
        f.write(header)
        np.savetxt(f, aspect, fmt="%4i")
    
    # Open the output file, add the header, save the array
    with open(shadegrid, "wb") as f:
        f.write(header)
        np.savetxt(f, shaded, fmt="%4i")

def las2contour(source, target, interval, base): 
    """
    las2contour(...)
        las2contour(source, target, interval, base)
        
        Generates contour lines from a grid ASCII file and save them in a ESRI
        shapefile.
        
        Parameters
        ----------
        source : string type.
            Path of the grid file. You can use las2grid, las2raster or 
            lasinterpolated to produce the grid.
        
        target : string type.
            Path where the contour should be saved.
        
        interval : float type.
            Interval in meters between lines.
            
         base : float type.
             Value of which the algorithm should start creating lines.
             For example:
                 If the minimum value of your data is 650 meters, there is no
                 point in putting the base = 1. You should put starting from 645
                 or 649.
                 On the other hand, if you want to create the contour lines from 
                 a nDSM, you should put base = 0. Contour lines will start to be
                 written from the elevation 0m.
            
    """
    # Imports the driver to create the shapefile
    ogr_driver = ogr.GetDriverByName('ESRI Shapefile')
    
    # Check if the file already exists, removing it in case it does
    if os.path.exists(target + ".shp"):
        os.remove(target + ".shp")
        ogr_ds = ogr_driver.CreateDataSource(target + ".shp")
    
    # Create new shapefile in the destination folder
    ogr_ds = ogr_driver.CreateDataSource(target + ".shp")
    ogr_lyr = ogr_ds.CreateLayer(target, geom_type=ogr.wkbLineString25D)
    field_defn = ogr.FieldDefn('ID', ogr.OFTInteger)
    ogr_lyr.CreateField(field_defn)
    field_defn = ogr.FieldDefn('ELEV', ogr.OFTReal)
    ogr_lyr.CreateField(field_defn)
    
    # imports the image used as base for the contour calculation
    ds = gdal.Open(source)
    
    # Calculates contour based on the input image using the ContourGenerate
    # function from gdal package.
    
    # gdal.ContourGenerate() arguments
    # Band srcBand,
    # double contourInterval,
    # double contourBase,
    # double[] fixedLevelCount,
    # int useNoData,
    # double noDataValue,
    # Layer dstLayer,
    # int idField,
    # int elevField
    gdal.ContourGenerate(ds.GetRasterBand(1), interval, base, [], 0, 0, ogr_lyr, 0, 1)
    ogr_ds = None
    del ogr_ds

def raster2color(red, green, blue, target, EPSG): 
    """
    raster2color(...)
        raster2color(source, target)
        
        Generates a 3D band image from the source files (red, green, blue).
        
        Parameters
        ----------
        red : string type.
            Path of the image to be used in the RED channel (Band = 1).
        
        green : string type.
            Path of the image to be used in the GREEN channel (Band = 2).
            
        blue : string type.
            Path of the image to be used in the Blue channel (Band = 3).
        
        target : string type.
            Path where the colored image should be saved.
        
        EPSG : interget type.
            EPSG number for the coordinate reference system. If you don't know 
            the number you can surch for it here --> http://spatialreference.org/ref/epsg/.
    """
    # Load the ASCII DEM into a numpy array
    arr = np.loadtxt(red, skiprows=6)
    arr_ = np.loadtxt(green, skiprows=6)
    arr__ = np.loadtxt(blue, skiprows=6)
    
    # Convert the numpy array to a PIL image
    im = Image.fromarray(arr).convert('L')
    im_ = Image.fromarray(arr_).convert('L')
    im__ = Image.fromarray(arr__).convert('L')
    
    # Rotate the image to fit to the correct orientation
    im1_ = np.array(im)[::-1]
    im2_ = np.array(im_)[::-1]    
    im3_ = np.array(im__)[::-1]
    
    # get the geo infomation
    ds = gdal.Open(red)
    geotransform = ds.GetGeoTransform()
    
    # Create and save the GeoTiff image with the correct EPSG
    rows = im1_.shape[0]
    cols = im1_.shape[1] 
    dst_ds = gdal.GetDriverByName('GTiff').Create(target, cols, rows, 3, gdal.GDT_Byte)
    
    # specify coords
    dst_ds.SetGeoTransform(geotransform)  
    
    # write RGB bands to the raster
    dst_ds.GetRasterBand(1).WriteArray(im1_) 
    dst_ds.GetRasterBand(2).WriteArray(im2_)
    dst_ds.GetRasterBand(3).WriteArray(im3_)
    
    # establish encoding
    srs = osr.SpatialReference()     
    
    # Set lat/long vased on the EPSG
    srs.ImportFromEPSG(EPSG)                
    dst_ds.SetProjection(srs.ExportToWkt())
    
    # write to disk
    dst_ds.FlushCache()                    
    dst_ds = None 