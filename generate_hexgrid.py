
import numpy as np
import gdspy
from numba import jit,njit
import matplotlib.pyplot as plt

@jit()
def rot(alpha):
    return np.matrix( [[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]] )

@jit()
def get_circle(r,n=12):
    v = np.array( [r,0] )

    x = np.zeros(0)
    y = np.zeros(0)
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    return x,y


@njit()
def get_hexgrid(size,dist):
    d = dist# (dist+2*r)
    #a = np.sqrt(3)*d
    dx = np.sqrt(3)
    dy = 1/2

    #x_pos = np.arange(d,size-d,0.5*d*dx)
    #y_pos = np.arange(d,size-d,2*d*dy)
    x_pos = np.arange(0, size,1.0)*0.5*d*dx
    y_pos = np.arange(0, size,1.0)*2*d*dy

    x = np.zeros(len(x_pos)*len(y_pos))
    y = np.zeros(len(x_pos)*len(y_pos))
    #xy =

    counter = 0

    for i in range(len(x_pos)):
        x0 = x_pos[i]
        for j in range(len(y_pos)):
            y0 = y_pos[j]
            if (i % 2):
                if (j+2) % 3:
                    x[counter] = x0
                    if not i % 2:
                        y[counter] = y0
                    else:
                        y[counter] = y0 + d * dy
                    counter += 1
            elif (j % 3):
                x[counter] = x0
                if not i % 2:
                    y[counter] = y0
                else:
                    y[counter] = y0 + d * dy
                counter += 1

    return x[:counter],y[:counter]

@jit()
def add_field(cell,hx,hy,x,y,r,layer=1):

    for j in range(len(hx)):
        x2, y2 = get_circle(r, 32)
        x2 += x
        y2 += y
        x2 += hx[j]
        y2 += hy[j]
        points = np.vstack((x2, y2)).T
        # Create the polygon on layer 1.
        poly1 = gdspy.Polygon(points, layer)

        # Add the new polygon to the cell.
        cell.add(poly1)



def add_hexgrid_field(cell,x,y,dist,r,n,layer=1):
    hx, hy = get_hexgrid(n, dist)
    # plt.scatter(hx,hy)
    # plt.show()
    # print((len(hx),len(hy)))
    #hx /= 1000
    #hy /= 1000
    add_field(cell,hx,hy,x,y,r,layer)



def make_hexgrid_cell(name):
    cell = gdspy.Cell(name)

    space = 10+15

    r = (60 / 2) * 1e-3
    dist = np.arange(20, 100, 10) * 1e-3 + 2*r
    n = int(50000/dist.max())-2

    for j in range(len(dist)):
        add_hexgrid_field(cell,space*j,0,dist[j],r,n)

make_hexgrid_cell("HEXGRID")


# Output the layout to a GDSII file (default to all created cells).
# Set the units we used to micrometers and the precision to picometers.
gdspy.write_gds('sdickreuter_hexgrid.gds', unit=1.0e-6, precision=1.0e-12)

# ------------------------------------------------------------------ #
#      VIEWER
# ------------------------------------------------------------------ #

# View the layout using a GUI.  Full description of the controls can
# be found in the online help at http://gdspy.sourceforge.net/
gdspy.LayoutViewer()
