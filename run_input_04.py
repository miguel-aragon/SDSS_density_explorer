#
#
#  Working version, uses fast interpolation
#
#
#
#
#  Search dataset instead of loading into ram:
#  https://www.devdungeon.com/content/working-binary-data-python#seek_file_position
#

import numpy as np
import sys
from io import BytesIO
from flask import Flask, render_template, send_file, make_response, request

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from interp3d import interp_3d
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, '/Users/miguel/.local/python_utils/')
sys.path.insert(0, '/Users/miguel/.local/python_utils/SDSS/')
from Volumes.volumes import read_fvolume
from sdss_radec_to_xyz import *
from sdss_get_los import *




#--------------------------------------------
#
#--------------------------------------------
def get_sphere_simple(vol_den, interp, ra, dec, reds, n_lon, n_lat, radius):
    HubbleParam = 73.0
    c_light     = 300000.0
    
    radius_z = radius * HubbleParam / c_light
    print('>>> ', radius, radius_z)
    
    max_l = 1024.0 
    x0    = 0.5   #--- X Origin of survey inside box
    y0    = 0.0   #--- Y Origin of survey inside box
    z0    = 0.04  #--- Z Origin of survey inside box
    
    #--- RA range
    ra_arr  = np.arange(0,360, 360/n_lon)

    #--- RA range
    dec_arr  = np.arange(-90, 90, 180/n_lat)

    #--- Galaxy position inside grid
    xg,yg,zg = sdss_radec_to_xyz(ra,dec,reds, HubbleParam)
    print('>>> xg,yg,zg: ', xg,yg,zg)
    
    ima = np.zeros((n_lon, n_lat))
    for i in range(n_lon):
        for j in range(n_lat):           
            
            #--- Compute 3D coordinates of sphere
            xl,yl,zl = sdss_radec_to_xyz(ra_arr[i],dec_arr[j],radius_z, HubbleParam)
            xt = xl+xg
            yt = yl+yg
            zt = zl+zg
            
            #--- Unitary position and add grid offset
            xl = xt/max_l + x0
            yl = yt/max_l + y0
            zl = zt/max_l + z0

            #--- Sample volume
            #vol_den_ijk = vol_den[(xl*1024).astype(int),(yl*1024).astype(int),(zl*1024).astype(int)]
            vol_den_ijk = interp((xl*1024, yl*1024, zl*1024))
            ima[i,j] = np.power(vol_den_ijk,0.2)

    ima = ima / np.max(ima)*255
    ima = np.transpose(ima)
    return ima



#--------------------------------------------
#
#--------------------------------------------
def get_reds_slice_simple(vol_den, interp, ra, dec, reds, ra_delta, dec_delta, n_ra, n_dec):
    HubbleParam = 73.0
    c_light     = 300000.0

    max_l = 1024.0 
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

    ima = np.zeros((n_ra, n_dec))
    for i in range(n_ra):
        for j in range(n_dec):
            #--- Compute 3D coordinates of LOS
            xl,yl,zl = sdss_radec_to_xyz(ra_arr[i],dec_arr[j],reds, HubbleParam)
            #--- Unitary position and add grid offset
            xl = xl/max_l + x0
            yl = yl/max_l + y0
            zl = zl/max_l + z0

            #--- Sample volume
            #vol_den_ijk = vol_den[(xl*1024).astype(int),(yl*1024).astype(int),(zl*1024).astype(int)]
            vol_den_ijk = interp((xl*1024, yl*1024, zl*1024))            
            ima[i,j] = np.power(vol_den_ijk,0.2)

    ima = ima / np.max(ima)*255
    ima = np.transpose(ima)
    return ima

#--------------------------------------------
#   Makes a slice at dec
#--------------------------------------------
def make_dec_slice(_ra, _dec, _ra_delta, _ra_n):
    #--- Make array of ra positions at equator
    ra1 = _ra - _ra_delta/2
    ra2 = _ra + _ra_delta/2
    ras  = np.arange(ra1,ra2, (ra2-ra1)/_ra_n) - _ra
    
    #--- From (ra, dec=0) plane to unitary vector
    cx = np.cos(np.deg2rad(_dec))*np.cos(np.deg2rad(ras))
    cy = np.cos(np.deg2rad(_dec))*np.sin(np.deg2rad(ras))
    cz = ras*0
    
    #--- Rotate to dec position
    cxa,cya,cza = rotate_around_axis(cx, cy, cz,  'Y', -_dec)
    cxr,cyr,czr = rotate_around_axis(cxa,cya,cza, 'Z',  _ra)
        
    #--- Back to ra, dec, convert to degrees
    dec2  = np.rad2deg(np.arcsin(czr))
    ra2   = np.rad2deg(np.arctan2(cyr, cxr))

    #--- Fix angles
    for i,ra_i in enumerate(ras):
        if ra2[i] < 0:
            ra2[i] = ra2[i] + 360
    
    return ra2, dec2



#--------------------------------------------
#
#--------------------------------------------

def get_dec_slice_correct(vol_den, interp, ra, dec, reds, ra_delta, reds_delta, n_ra, n_reds):
    HubbleParam = 73.0
   
    z1, z2 = reds=reds-reds_delta/2, reds+reds_delta/2  # Initial redshift
    z_arr = np.arange(z1,z2,(z2-z1)/n_reds)

    #--- Construct correct radec slice
    ra_new, dec_new =  make_dec_slice(ra, dec, ra_delta, n_ra)

    ima = np.zeros((n_ra, n_reds))
    for i in range(n_ra):
        #--- Compute 3D coordinates of LOS
        xl,yl,zl = sdss_get_los(ra_new[i],dec_new[i],z1,z2,n_reds, HubbleParam)
        #--- Sample volume with LOS
        #ima[i,:] = np.power(vol_den[(xl*1024).astype(int),(yl*1024).astype(int),(zl*1024).astype(int)],0.2)
        for j in range(n_reds):
            vol_den_ijk = interp((xl[j]*1024, yl[j]*1024, zl[j]*1024))  
            ima[i,j] = np.power(vol_den_ijk,0.2)


    ima = ima / np.max(ima)*255
    ima = np.transpose(ima)
    return ima


#========================================================================


PATH_DEN = '/Users/miguel/Desktop/SDSS_density_explorer/Data/'
#FILE_DEN = 'DR13_D_all-XX_G15.Y-0.den-mean.fvol'
#FILE_DEN = 'DR13_D_all-01.masked.fvol'
FILE_DEN = 'DR13_D_all_random_boundary.ENS-256.SIG-0.5.fvol'

#--- Read volume as global variable
print('>>> Reading volume...')
vol_den = read_fvolume(PATH_DEN + FILE_DEN)
print('>>>    ready!')


#--- Cython implementation of interpolation uses float_t (double)
vol_den = np.asfarray( vol_den, dtype='float' )

#--- Hard-coded grid size, bad programming.
x = np.linspace(0,1023,1024)
y = np.linspace(0,511,512)
z = np.linspace(0,511,512)

X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
interp = interp_3d.Interp3D(vol_den, x,y,z)


#========================================================================

app = Flask(__name__)


#--------------------------------------------
#
#--------------------------------------------
@app.route('/profile/', methods=['GET'])
def profile():

    dec = float(request.args.get('dec'))
    ra  = float(request.args.get('ra'))
    print('>>> Requested slice centered at ra, dec: ', ra, dec)
    
    #ima = get_dec_slice_simple( vol_den, ra=190, dec=dec, ra_delta=80, n_ra=256, n_z=256)

    ima = get_dec_slice_correct(vol_den, ra=ra, dec=dec, ra_delta=90, n_ra=256, n_z=256)

    fig, ax = plt.subplots(1)
    plt.plot(ima[:,127],color='red')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    
    return send_file(img, mimetype='image/png')

#--------------------------------------------
#
#  Sphere around galaxy
#  -------
#
#  VOID_04
#  http://127.0.0.1:5000/sky_sphere/?n_lon=256&n_lat=128&ra=155.6469&dec=45.6392&reds=0.020&radius=10	
#
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

    ima = get_sphere_simple(vol_den, interp, ra=ra, dec=dec, reds=reds, n_lon=n_lon, n_lat=n_lat, radius=radius)
    
    x1 = 0
    x2 = 360
    y1 = -90
    y2 = 90
    fig, ax = plt.subplots(1)
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
#  Declination slice (RA,z)
#  -------
#  VOID_04
#  http://127.0.0.1:5000/slice_dec/?ra_delta=30&reds_delta=0.01&n_ra=64&n_reds=64&ra=155.6469&dec=45.6392&reds=0.020
#
#--------------------------------------------
@app.route('/slice_dec/', methods=['GET'])
def slice_dec():

    #--- Speed of light
    c_l = 299792.458

    ra          = float(request.args.get('ra'))
    dec         = float(request.args.get('dec'))
    reds        = float(request.args.get('reds'))
    ra_delta    = float(request.args.get('ra_delta'))
    reds_delta  = float(request.args.get('reds_delta'))
    n_ra        = int(request.args.get('n_ra'))
    n_reds      = int(request.args.get('n_reds'))
    print('>>> Requested slice centered at ra, dec, z: ', ra, dec, reds)
    
    ima = get_dec_slice_correct(vol_den, interp, ra=ra, dec=dec, reds=reds, ra_delta=ra_delta, reds_delta=reds_delta, n_ra=n_ra, n_reds=n_reds)

    x1 = -ra_delta/2
    x2 =  ra_delta/2
    y1 = -(reds_delta/2)*c_l
    y2 =  (reds_delta/2)*c_l
    print('plot ', x1,x2,y1,y2)
    fig, ax = plt.subplots(1)
    ax.set_xlabel('ra')
    ax.set_ylabel('km/s')
    plt.imshow(ima,interpolation="none", origin='lower',cmap='terrain',extent=[x1,x2,y1,y2],aspect='auto')
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
    
    # def get_reds_slice_simple(vol_den, ra, dec, reds, ra_delta, dec_delta, n_ra=32, n_dec=32):
    ima = get_reds_slice_simple(vol_den, interp, ra=ra, dec=dec, reds=reds, ra_delta=ra_delta, dec_delta=dec_delta, n_ra=n_ra, n_dec=n_dec)

    x1 = -ra_delta/2
    x2 =  ra_delta/2
    y1 = -dec_delta/2
    y2 =  dec_delta/2
    fig, ax = plt.subplots(1)
    ax.set_xlabel('ra')
    ax.set_ylabel('dec')
    plt.imshow(ima,interpolation="none", origin='lower',cmap='terrain',extent=[x1,x2,y1,y2])
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    
    return send_file(img, mimetype='image/png')


#--------------------------------------------
#   Download cutout as fits file
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




if __name__ == '__main__':
    app.run(debug=True)

