#
#
#  Working version, uses fast interpolation
#
#
#  Install interp3D library:
#     https://github.com/jglaser/interp3d
#

import numpy as np
import sys
from io import BytesIO
from flask import Flask, render_template, send_file, make_response, request

import struct
import array as arr

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


#--------------------------------------------
#
#--------------------------------------------
def read_bvolume(filename):
    
    F = open(filename,'rb')

    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)
    
    den = arr.array('B')
    den.fromfile(F,sizeX*sizeY*sizeZ)    
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.uint8)
    
    return den

#--------------------------------------------
#  Input angle in DEGREES
#--------------------------------------------
def rotate_around_axis(x,y,z, axis, angle):

    cangle = np.cos(np.deg2rad(angle))
    sangle = np.sin(np.deg2rad(angle))
    
    if axis.capitalize() == 'X':
        xr = x
        yr = y*cangle - z*sangle
        zr = y*sangle + z*cangle
        
    if axis.capitalize() == 'Y':
        xr =  x*cangle + z*sangle
        yr =  y
        zr = -x*sangle + z*cangle

    if axis.capitalize() == 'Z':
        xr = x*cangle - y*sangle
        yr = x*sangle + y*cangle
        zr = z
        
    return xr,yr,zr


#--------------------------------------------
#
#--------------------------------------------
def sdss_radec_to_xyz(max_l, ra, dec, zred, H_o):

    x0    = 0.5                     #--- X Origin of survey inside box
    y0    = 0.0                     #--- Y Origin of survey inside box
    z0    = 0.04                    #--- Z Origin of survey inside box    
    
    #-- Speed of light
    c_l = 299792.458
    
    #--- Unitary vector
    dec2 = np.deg2rad(dec)
    ra2  = np.deg2rad(ra)
    cx = np.cos(dec2)*np.cos(ra2)
    cy = np.cos(dec2)*np.sin(ra2)
    cz = np.sin(dec2)

    #--- Distance [Mpc]
    x_Mpc  =  cx * zred * c_l/H_o
    y_Mpc  =  cy * zred * c_l/H_o
    z_Mpc  =  cz * zred * c_l/H_o

    #--- Rotate points to (custom) survey box
    x,y,z = rotate_around_axis(x_Mpc,y_Mpc,z_Mpc,'z',-90.0)

    return x,y,z


#--------------------------------------------
#
#--------------------------------------------
def get_sphere_simple(max_l, vol_den, ra, dec, reds, n_lon, n_lat, radius):
    HubbleParam = 73.0
    c_light     = 300000.0
    
    radius_z = radius * HubbleParam / c_light
    print('>>> ', radius, radius_z)
    
    x0    = 0.5   #--- X Origin of survey inside box
    y0    = 0.0   #--- Y Origin of survey inside box
    z0    = 0.04  #--- Z Origin of survey inside box
    
    #--- RA range
    ra_arr  = np.arange(0,360, 360/n_lon)

    #--- RA range
    dec_arr  = np.arange(-90, 90, 180/n_lat)

    #--- Galaxy position inside grid
    xg,yg,zg = sdss_radec_to_xyz(max_l,ra,dec,reds, HubbleParam)
    
    ima = np.zeros((n_lon, n_lat), dtype=np.float32)
    for i in range(n_lon):
        for j in range(n_lat):           
            
            #--- Compute 3D coordinates of sphere
            xl,yl,zl = sdss_radec_to_xyz(max_l, ra_arr[i],dec_arr[j],radius_z, HubbleParam)
            xt = xl+xg
            yt = yl+yg
            zt = zl+zg
            
            #--- Unitary position and add grid offset
            xl = xt/max_l + x0
            yl = yt/max_l + y0
            zl = zt/max_l + z0

            #--- Sample volume
            ima[i,j] = vol_den[(xl*max_l).astype(int),(yl*max_l).astype(int),(zl*max_l).astype(int)]
            ima[i,j] = np.power(ima[i,j], 1.5)

    mini = (ima == np.min(ima)).nonzero()[0]
    mini_inv = (ima == np.min(ima)).nonzero()[0]
    ima[mini] = np.min(ima[mini_inv])

    ima = np.transpose(ima)
    return ima



#--------------------------------------------
#
#--------------------------------------------
def get_reds_slice_simple(max_l, vol_den, ra, dec, reds, ra_delta, dec_delta, n_ra, n_dec):
    HubbleParam = 73.0
    c_light     = 300000.0

    x0    = 0.5   #--- X Origin of survey inside box
    y0    = 0.0   #--- Y Origin of survey inside box
    z0    = 0.04  #--- Z Origin of survey inside box
    
    #--- RA range
    ra1 = ra - ra_delta/2
    ra2 = ra + ra_delta/2
    ra_arr  = np.arange(ra1,ra2, (ra2-ra1)/n_ra)

    #--- RA range
    dec1 = dec - dec_delta/2
    dec2 = dec + dec_delta/2
    dec_arr  = np.arange(dec1,dec2, (dec2-dec1)/n_dec)

    ima = np.zeros((n_ra, n_dec), dtype=np.float32)
    for i in range(n_ra):
        for j in range(n_dec):
            #--- Compute 3D coordinates of LOS
            xl,yl,zl = sdss_radec_to_xyz(max_l, ra_arr[i],dec_arr[j],reds, HubbleParam)
            #--- Unitary position and add grid offset
            xl = xl/max_l + x0
            yl = yl/max_l + y0
            zl = zl/max_l + z0

            #--- Sample volume
            ima[i,j] = vol_den[(xl*max_l).astype(int),(yl*max_l).astype(int),(zl*max_l).astype(int)]
            ima[i,j] = np.power(ima[i,j], 1.5)

    mini = (ima == np.min(ima)).nonzero()[0]
    mini_inv = (ima == np.min(ima)).nonzero()[0]
    ima[mini] = np.min(ima[mini_inv])
    
    ima = np.transpose(ima)
    return ima



#========================================================================
#
#   Flask server
#
#========================================================================

app = Flask(__name__)



#========================================================================
#
#   Read dataset and prepare for interpolation
#
#========================================================================

PATH_DEN = '../Data/'
FILE_DEN = 'DR13_D_all_random_boundary.ENS-256.SIG-0.5.mask.log10.bvol'  # Log10 clipped in rage (-2,3)


max_l = 512.0

#--- Read volume as global variable
print('>>> Reading volume...')
#vol_den = read_bvolume(PATH_DEN + FILE_DEN)

vol_den = np.load(PATH_DEN + FILE_DEN + '.npy')
print('>>>    ready!')

#np.save(FILE_DEN + '.npy', vol_den)


#--------------------------------------------
#
#  Sphere around galaxy
#  -------
#
#  VOID_04
#  http://127.0.0.1:5000/sky_sphere/?n_lon=256&n_lat=128&ra=155.6469&dec=45.6392&reds=0.020&radius=10	
#
#  Polar ring:
#  http://127.0.0.1:5000/sky_sphere/?n_lon=256&n_lat=128&ra=157.0802&dec=62.5840&reds=0.0178&radius=20
#--------------------------------------------
@app.route('/sky_sphere/', methods=['GET'])
def sky_sphere():

    dec    = float(request.args.get('dec'))
    ra     = float(request.args.get('ra'))
    reds   = float(request.args.get('reds'))
    radius = float(request.args.get('radius'))
    n_lon  = int(request.args.get('n_lon'))
    n_lat  = int(request.args.get('n_lat'))
    print('>>> Requested sphere around galaxy at ra, dec, z:', ra, dec, reds)

    ima = get_sphere_simple(max_l, vol_den, ra=ra, dec=dec, reds=reds, n_lon=n_lon, n_lat=n_lat, radius=radius)
    
    x1 = 0
    x2 = 360
    y1 = -90
    y2 = 90
    fig, ax = plt.subplots(1)
    plt.rcParams['figure.figsize'] = [16, 8]
    ax.set_xlabel('lat')
    ax.set_ylabel('lon')
    plt.imshow(ima,interpolation="none", origin='lower',cmap='terrain',extent=[x1,x2,y1,y2])
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    
    return send_file(img, mimetype='image/png')


#--------------------------------------------
#
#  Redshift (sky) projection (RA,DEC)
#  -------
#
#  VOID_04
#  http://127.0.0.1:5000/slice_reds/?ra_delta=30&dec_delta=30&n_ra=64&n_dec=64&ra=155.6469&dec=45.6392&reds=0.020	
#  COMA
#  http://127.0.0.1:5000/slice_reds/?ra_delta=30&dec_delta=30&n_ra=64&n_dec=64&ra=194.9529&dec=27.9805&reds=0.0231
#
#--------------------------------------------
@app.route('/slice_reds/', methods=['GET'])
def slice_reds():

    dec       = float(request.args.get('dec'))
    ra        = float(request.args.get('ra'))
    reds      = float(request.args.get('reds'))
    ra_delta  = float(request.args.get('ra_delta'))
    dec_delta = float(request.args.get('dec_delta'))
    n_ra      = int(request.args.get('n_ra'))
    n_dec     = int(request.args.get('n_dec'))
    print('>>> Requested a slice centered at ra, dec, z:', ra, dec, reds)
    print('>>> Requested a slice with geometry         :', ra_delta, dec_delta, n_ra, n_dec)
    
    ima = get_reds_slice_simple(max_l, vol_den, ra=ra, dec=dec, reds=reds, ra_delta=ra_delta, dec_delta=dec_delta, n_ra=n_ra, n_dec=n_dec)

    x1 = -ra_delta/2
    x2 =  ra_delta/2
    y1 = -dec_delta/2
    y2 =  dec_delta/2
    fig, ax = plt.subplots(1)
    plt.rcParams['figure.figsize'] = [6, 6]
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\delta$')
    ima = np.flip(ima,1)
    plt.imshow(ima,interpolation="none", origin='lower',cmap='terrain',extent=[x1,x2,y1,y2])
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    
    return send_file(img, mimetype='image/png')


#--------------------------------------------
#   Download cutout as fits file (in development...)
#
#  See https://gist.github.com/nkint/684f7205c75d497947fe624ba3c098a7
#      https://gist.github.com/sergeyk/4536515
#      https://stackoverflow.com/questions/54899367/send-numpy-array-as-bytes-from-python-to-js-through-flask/54955161
#
#--------------------------------------------
@app.route('/download')
def download_file():
	path = "dust.npy"
	return send_file(path, as_attachment=True)

#--------------------------------------------
#
#--------------------------------------------
@app.route('/image/')
def images():
    return render_template("image.html")



#if __name__ == '__main__':
#    app.run(debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
