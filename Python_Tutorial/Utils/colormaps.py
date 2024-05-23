import sys, os
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
 
# setting path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom Colors
LIGHTGRAY = [0.85,0.85,0.85]

# Custom Colormaps
highContrast_b2r_black = LinearSegmentedColormap.from_list('highContrast_b2r_black', [
                                        (0.0,'darkblue'),(0.1,'blue'),(0.4, 'cyan'),
                                        (0.5,'black'),(0.6,'yellow' )
                                        ,(0.9,'red'),(1.0,'darkred')])
mpl.colormaps.register(highContrast_b2r_black)

b2r_black = LinearSegmentedColormap.from_list('b2r_black', [
                                        (0.0,'blue'),
                                        (0.5,'black'),
                                        (1.0,'red')])
mpl.colormaps.register(b2r_black)

highContrast_white = LinearSegmentedColormap.from_list('highContrast_b2r_white', [
                                        (0.0,'cyan'),(0.2,'blue'),(0.3, 'darkblue'),
                                        (0.5,'white'),(0.7,'darkred' )
                                        ,(0.8,'red'),(1.0,'yellow')])
mpl.colormaps.register(highContrast_white)

# Function to invert a colormap
def invert_cmap(cmap_name):
    cmap = mpl.colormaps[cmap_name]
    rgb = cmap(np.linspace(0,1,10))[:,:3]
    rgb_i = 1-rgb
    outmap = LinearSegmentedColormap.from_list(cmap_name, rgb_i, N=256)
    mpl.colormaps.register(outmap)
    return outmap

gray2cool = LinearSegmentedColormap.from_list('gray2cool', [
                                        (0.0,'white'),(0.5,'black'),
                                        (0.65,'darkblue'), (0.85,'blue'), (1,'cyan')])
mpl.colormaps.register(gray2cool)
mpl.colormaps.register(cmap=gray2cool.reversed())

gray2blue = LinearSegmentedColormap.from_list('gray2blue', [
                                        (0.0,'white'), (0.25,'gray'), (0.7,'blue'), (0.9,'darkblue'), (1,'midnightblue')])
mpl.colormaps.register(gray2blue)
mpl.colormaps.register(cmap=gray2blue.reversed())

gray2hot = LinearSegmentedColormap.from_list('gray2hot', [
                                        (0.0,'white'),(0.5,'black'),
                                        (0.65,'darkred'), (0.85,'red'), (1,'yellow')])
mpl.colormaps.register(gray2hot)
mpl.colormaps.register(cmap=gray2hot.reversed())

gray2red = LinearSegmentedColormap.from_list('gray2red', [
                                        (0.0,'white'), (0.25,'gray'), (0.7,'red'), (0.9,'firebrick'), (1,'maroon')])
mpl.colormaps.register(gray2red)
mpl.colormaps.register(cmap=gray2red.reversed())

gray2lime = LinearSegmentedColormap.from_list('gray2lime', [
                                        (0.0,'white'),(0.5,'black'),
                                        (0.65,'darkgreen'), (0.85,'green'), (1,'lime')])
mpl.colormaps.register(gray2lime)
mpl.colormaps.register(cmap=gray2lime.reversed())

gray2green = LinearSegmentedColormap.from_list('gray2green', [
                                        (0.0,'white'), (0.25,'gray'), (0.7,'forestgreen'), (0.9,'green'), (1,'darkgreen')])
mpl.colormaps.register(gray2green)
mpl.colormaps.register(cmap=gray2green.reversed())

cool2gray2hot = LinearSegmentedColormap.from_list('cool2gray2hot', [(0.0, 'cyan'), (0.05,'blue'), (0.1,'darkblue'),
                                        (0.2,'black'), (0.5,'white'), (0.8,'black'),
                                        (0.9,'darkred'), (0.95,'red'), (1,'yellow')])
mpl.colormaps.register(cool2gray2hot)
mpl.colormaps.register(cmap=cool2gray2hot.reversed())

kindlemann_colors = 1./255.*np.array([[0,0,0],
                    [27,1,29],
                    [36,3,55],
                    [39,4,79],
                    [38,5,105],
                    [31,6,133],
                    [25,8,158],
                    [8,21,175],
                    [8,46,165],
                    [7,64,147],
                    [6,78,131],
                    [6,90,118],
                    [5,101,109],
                    [5,112,100],
                    [6,122,89],
                    [6,132,75],
                    [7,142,57],
                    [7,152,37],
                    [8,162,15],
                    [20,172,8],
                    [43,181,9],
                    [74,188,9],
                    [107,195,9],
                    [142,200,10],
                    [177,203,10],
                    [212,205,10],
                    [247,205,83],
                    [251,211,169],
                    [252,221,203],
                    [254,232,224],
                    [254,244,240],
                    [255,255,255]])
kindlemann_scalar = [0, 0.032258065, 0.064516129, 0.096774194, 0.129032258, 0.161290323, 0.193548387, 0.225806452, 0.258064516, 0.290322581, 0.322580645, 0.35483871, 0.387096774, 0.419354839, 0.451612903, 0.483870968, 0.516129032, 0.548387097, 0.580645161, 0.612903226, 0.64516129, 0.677419355, 0.709677419, 0.741935484, 0.774193548, 0.806451613, 0.838709677, 0.870967742, 0.903225806, 0.935483871, 0.967741935, 1]

kindlemann = LinearSegmentedColormap.from_list('kindlemann', list(zip(kindlemann_scalar, kindlemann_colors)), N=256)
mpl.colormaps.register(kindlemann)
mpl.colormaps.register(kindlemann.reversed())

def stitchColormaps(data, cm1, cm2, stitch_at='mid', name='stitched_cmap'):
    
    # Get the data midpoint
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    if stitch_at == 'mid':
        dmid = (dmin+dmax)/2
        prop = 0.5
    else:
        if stitch_at > dmax or stitch_at < dmin:
            dmid = (dmin+dmax)/2
            prop = 0.5
        else:
            dmid = stitch_at
            prop = np.abs(dmid-dmin)/np.abs(dmax-dmin)
    
    # Vectors based on proportion
    vec1 = np.linspace(0,1,int(np.round(512*prop)))
    vec2 = np.linspace(0,1,int(np.round(512*(1-prop))))
        
    # Get the colors of the colormaps
    colors1 = plt.cm.get_cmap(cm1)
    colors2 = plt.cm.get_cmap(cm2)
    cmap1 = colors1(vec1)
    cmap2 = colors2(vec2)
    
    # stack them and create a colormap
    cmap_vec = np.vstack((cmap1, cmap2))    
    mymap = LinearSegmentedColormap.from_list(name, cmap_vec)
    
    return mymap
    
    