
import numpy as np
import gdspy


def rot(alpha):
    return np.matrix( [[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]] )

def get_circle(r,n=12):
    v = np.array( [r,0] )

    x = np.zeros(0)
    y = np.zeros(0)
    for i in range(n):
        x2, y2 = (v*rot(2*np.pi/n*i)).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    return x,y

def get_hexamer(dist,r):
    x = np.zeros(0)
    y = np.zeros(0)
    v = np.array( [0.5*(dist+2*r)/np.sin(np.pi/6),0] )
    m = 6
    for i in range(m):
        x2, y2 = (v*rot(2*np.pi/m*i)).A1

        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )

    return x,y

def get_dimer(dist,r):
    x1 = np.zeros(1)
    y1 = np.zeros(1)
    x2 = np.zeros(1)
    y2 = np.zeros(1)
    x1 -= (r+dist/2)
    x2 += (r+dist/2)
    x = np.hstack( (x1,x2) )
    y = np.hstack( (y1,y2) )
    return x,y

def get_trimer(dist,r):
    x = np.zeros(0)
    y = np.zeros(0)
    v = np.array( [0,0.5*(dist+2*r)/np.sin(np.pi/3)] )
    m = 3
    for i in range(m):
        x2, y2 = (v*rot(2*np.pi/m*i)).A1
        x = np.hstack( (x,x2) )
        y = np.hstack( (y,y2) )
    return x,y

def get_triple(dist,r):
    x1 = np.zeros(1)
    y1 = np.zeros(1)
    x2 = np.zeros(1)
    y2 = np.zeros(1)
    x3 = np.zeros(1)
    y3 = np.zeros(1)
    x1 -= 2*r+dist
    x2 += 2*r+dist

    x = np.hstack( (x1,x2,x3) )
    y = np.hstack( (y1,y2,y3))
    return x,y


def get_triple_rotated(dist,r,alpha = 0):
    x1 = np.zeros(1)
    y1 = np.zeros(1)
    x2 = np.zeros(1)
    y2 = np.zeros(1)
    x3 = np.zeros(1)
    y3 = np.zeros(1)
    x1 -= 2*r+dist

    #v = np.array([r - dose_check_radius, 0])
    v = np.array([2*r+dist, 0])
    x_rot, y_rot = (v * rot(alpha)).A1

    x2 += x_rot
    y2 -= y_rot

    x = np.hstack( (x1,x2,x3))
    y = np.hstack( (y1,y2,y3))

    return x,y

def get_triple00(dist,r):
    return get_triple_rotated(dist,r,0)

def get_triple30(dist,r):
    return get_triple_rotated(dist,r,2*np.pi/12)

def get_triple60(dist,r):
    return get_triple_rotated(dist,r,2*np.pi/6)

def get_triple90(dist,r):
    return get_triple_rotated(dist,r,2*np.pi/4)

def get_asymdimer(dist,r):
    x1 = np.zeros(1)
    y1 = np.zeros(1)
    x2 = np.zeros(1)
    y2 = np.zeros(1)
    r2 = 1.5*r
    x1 -= r+dist/2
    x2 += r2+dist/2

    x = np.concatenate((x1,x2))
    y = np.concatenate((y1,y2))
    return x,y

def get_single(dist,r):
    x1 = np.zeros(1)
    y1 = np.zeros(1)
    return x1,y1


def add_field(cell,px,py,x,y,r,layer=1):

    fx = np.linspace(0.5, 14.0, 5)
    fy = fx.copy()

    fx, fy = np.meshgrid(fx, fy)
    fx = fx.ravel()
    fy = fy.ravel()

    for j in range(len(fx)):

        for i in range(len(px)):
            x2, y2 = get_circle(r, 64)
            x2 += x
            y2 += y
            x2 += 0.5
            y2 += 0.5
            x2 += px[i]
            y2 += py[i]
            x2 += fx[j]
            y2 += fy[j]
            points = np.vstack((x2, y2)).T
            # Create the polygon on layer 1.
            poly1 = gdspy.Polygon(points, layer)

            # Add the new polygon to the cell.
            cell.add(poly1)


def add_asym_dimer_field(cell,px,py,x,y,r,layer=1):
    r = [r,1.5*r]
    fx = np.linspace(0.5, 14.0, 5)
    fy = fx.copy()

    fx, fy = np.meshgrid(fx, fy)
    fx = fx.ravel()
    fy = fy.ravel()

    for j in range(len(fx)):

        for i in range(len(px)):
            x2, y2 = get_circle(r[i], 64)
            x2 += x
            y2 += y
            x2 += 0.5
            y2 += 0.5
            x2 += px[i]
            y2 += py[i]
            x2 += fx[j]
            y2 += fy[j]
            points = np.vstack((x2, y2)).T
            # Create the polygon on layer 1.
            poly1 = gdspy.Polygon(points, layer)

            # Add the new polygon to the cell.
            cell.add(poly1)

def add_hexamer_field(cell,x,y,dist,r,layer=1):
    hx, hy = get_hexamer(dist, r)
    add_field(cell,hx,hy,x,y,r,layer)

def add_dimer_field(cell,x,y,dist,r,layer=1):
    hx, hy = get_dimer(dist, r)
    add_field(cell,hx,hy,x,y,r,layer)

def add_trimer_field(cell, x, y, dist, r, layer=1):
    hx, hy = get_trimer(dist, r)
    add_field(cell, hx, hy, x, y, r, layer)

def add_triple00_field(cell, x, y, dist, r, layer=1):
    hx, hy = get_triple00(dist, r)
    add_field(cell, hx, hy, x, y, r, layer)

def add_triple30_field(cell, x, y, dist, r, layer=1):
    hx, hy = get_triple30(dist, r)
    add_field(cell, hx, hy, x, y, r, layer)

def add_triple60_field(cell, x, y, dist, r, layer=1):
    hx, hy = get_triple60(dist, r)
    add_field(cell, hx, hy, x, y, r, layer)

def add_triple90_field(cell, x, y,dist, r, layer=1):
    hx, hy = get_triple90(dist, r)
    add_field(cell, hx, hy, x, y, r, layer)

def add_asymdimer(cell, x, y,dist, r, layer=1):
    hx, hy = get_asymdimer(dist, r)
    add_asym_dimer_field(cell, hx, hy, x, y, r, layer)

def add_single(cell, x, y, dist, r, layer=1):
    hx, hy = get_single(dist, r)
    add_field(cell, hx, hy, x, y, r, layer)

def add_markerh(cell,x,y,layer=1):
    rect = gdspy.Rectangle((0, 0), (15, 2), layer)
    rect.translate(x, y)
    cell.add(rect)
    rect = gdspy.Rectangle((0, 3), (15, 5), layer)
    rect.translate(x, y)
    cell.add(rect)
    rect = gdspy.Rectangle((0, 6), (15, 8), layer)
    rect.translate(x, y)
    cell.add(rect)
    rect = gdspy.Rectangle((0, 9), (15, 11), layer)
    rect.translate(x, y)
    cell.add(rect)
    rect = gdspy.Rectangle((0, 12), (15, 14), layer)
    rect.translate(x, y)
    cell.add(rect)

def add_markerv(cell,x,y,layer=1):
    rect = gdspy.Rectangle((0, 0), (2, 15), layer)
    rect.translate(x, y)
    cell.add(rect)
    rect = gdspy.Rectangle((3, 0), (5, 15), layer)
    rect.translate(x, y)
    cell.add(rect)
    rect = gdspy.Rectangle((6, 0), (8, 15), layer)
    rect.translate(x, y)
    cell.add(rect)
    rect = gdspy.Rectangle((9, 0), (11, 15), layer)
    rect.translate(x, y)
    cell.add(rect)
    rect = gdspy.Rectangle((12, 0), (14, 15), layer)
    rect.translate(x, y)
    cell.add(rect)



# def make_cell(name,function):
#     cell = gdspy.Cell(name)
#
#     n = 5
#     space = 10+15
#
#     gx = np.arange(0,space*n,space)
#
#     for i in range(len(gx)):
#         add_markerv(cell,gx[i]+35,n*space+15,2)
#
#     for i in range(len(gx)):
#         add_markerh(cell,0,gx[i],2)
#
#     gy = gx.copy()
#     gx += 35
#
#     gx,gy = np.meshgrid(gx,gy)
#
#     r = np.array([30,40,50,60,70])/2*1e-3
#     dist = np.array([10,20,30,40,50])*1e-3
#
#     for i in range(gx.shape[0]):
#         for j in range(gx.shape[1]):
#             function(cell,gx[i,j],gy[i,j],dist[j],r[gx.shape[0]-i-1])
#
# make_cell("HEXAMER",add_hexamer_field)
# make_cell("DIMER",add_dimer_field)
# make_cell("TRIMER",add_trimer_field)
# make_cell("TRIPEL00",add_triple00_field)
# make_cell("TRIPEL30",add_triple30_field)
# make_cell("TRIPEL60",add_triple60_field)
# make_cell("TRIPEL90",add_triple90_field)


def make_cell2(name,functions):
    cell = gdspy.Cell(name)

    n = len(functions)
    space = 10+15
    #dist = np.array([10,20,30,40,50])*1e-3
    dist = np.arange(20,80,2)*1e-3
    #dist = np.arange(40, 60, 1) * 1e-3
    #dist = np.append(dist,np.arange(40, 100, 2) * 1e-3)

    r = (60/2)*1e-3

    for i in range(n):
        x = 0
        y = (n-i-1) * space + 15
        if functions[i] is not None:
            add_markerh(cell, x, y, 2)
            x += space + 15
            for j,d in enumerate(dist):
                functions[i](cell, x,y , d, r)
                x += space
                if (j+1)%5 == 0:
                    x+=15

    y = (n +1-1) * space + 30
    x = space + 15
    for j in range(len(dist)):
        if j % 5 == 0:
            add_markerv(cell, x, y, 2)
        x += space
        if (j + 1) % 5 == 0:
            x += 15

    x = 0
    y = (n + 2-1) * space + 30
    add_single(cell,x,y,0.0,r)

make_cell2("OLIGOMERS", [add_dimer_field,add_trimer_field,add_hexamer_field,None,add_asymdimer,None,add_triple00_field,add_triple30_field,add_triple60_field,add_triple90_field])


# Output the layout to a GDSII file (default to all created cells).
# Set the units we used to micrometers and the precision to picometers.
gdspy.write_gds('sdickreuter_patterns4.gds', unit=1.0e-6, precision=1.0e-12)

# ------------------------------------------------------------------ #
#      VIEWER
# ------------------------------------------------------------------ #

# View the layout using a GUI.  Full description of the controls can
# be found in the online help at http://gdspy.sourceforge.net/
gdspy.LayoutViewer()
