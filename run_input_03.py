#
#
#  Check where it is running and the crerate a url with ra and dec coordinates:
#
#
#     //--- Get image
#     http://127.0.0.1:5000/fig/?ra=190&dec=1
#
#     //--- Get profile
#     http://127.0.0.1:5000/profile/?ra=190&dec=1
#
# -----------------------------------------------------------------
#
#  http://hplgit.github.io/web4sciapps/doc/pub/._web4sa_flask004.html
#  http://dataviztalk.blogspot.com/2016/01/serving-matplotlib-plot-that-follows.html
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

sys.path.insert(0, '/Users/miguel/.local/python_utils/')
sys.path.insert(0, '/Users/miguel/.local/python_utils/SDSS/')
from Volumes.volumes import read_fvolume
from sdss_radec_to_xyz import *
from sdss_get_los import *

#--------------------------------------------
#
#--------------------------------------------
def get_slice_simple(vol_den, ra, dec, ra_delta, n_ra=256, n_z=256):
    HubbleParam = 73.0
    c_light     = 300000.0
    
    z1, z2 = 0.001, 0.1  # Initial redshift
    n_z = 256   # Number of steps to compute line of sight
    z_arr = np.arange(z1,z2,(z2-z1)/n_z)
    
    #--- RA range
    ra1 = ra - ra_delta
    ra2 = ra + ra_delta
    ra_arr  = np.arange(ra1,ra2, (ra2-ra1)/n_ra)
    
    ima = np.zeros((n_ra, n_z))
    for i in range(n_ra):
        #--- Compute 3D coordinates of LOS
        xl,yl,zl = sdss_get_los(ra_arr[i],dec,z1,z2,n_z, HubbleParam)    
        #--- Sample volume with LOS
        ima[i,:] = np.power(vol_den[(xl*1024).astype(int),(yl*1024).astype(int),(zl*1024).astype(int)],0.2)

    ima = ima / np.max(ima)*255
    ima = np.transpose(ima)
    return ima

#--------------------------------------------
#   Makes a slice at dec
#--------------------------------------------
def make_dec_slice(_ra, _dec, _ra_delta, _ra_n):
    #--- Make array of ra positions at equator
    ra1 = _ra - _ra_delta
    ra2 = _ra + _ra_delta
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
def get_slice_correct(vol_den, ra, dec, ra_delta, n_ra=256, n_z=256):
    HubbleParam = 73.0
   
    z1, z2 = 0.001, 0.1  # Initial redshift
    z_arr = np.arange(z1,z2,(z2-z1)/n_z)

    #--- Construct correct radec slice
    ra_new, dec_new =  make_dec_slice(ra, dec, ra_delta, n_ra)

    ima = np.zeros((n_ra, n_z))
    for i in range(n_ra):
        #--- Compute 3D coordinates of LOS
        xl,yl,zl = sdss_get_los(ra_new[i],dec_new[i],z1,z2,n_z, HubbleParam)
        #--- Sample volume with LOS
        ima[i,:] = np.power(vol_den[(xl*1024).astype(int),(yl*1024).astype(int),(zl*1024).astype(int)],0.2)

    ima = ima / np.max(ima)*255
    ima = np.transpose(ima)
    return ima

#========================================================================


PATH_DEN = '/Users/miguel/Desktop/SDSS_density_explorer/Data/'

#--- Read volume as global variable
vol_den = read_fvolume(PATH_DEN + 'DR13_D_all-01.masked.fvol')
print('>>> Reading volume')


app = Flask(__name__)

#--------------------------------------------
#
#--------------------------------------------
@app.route('/profile/', methods=['GET'])
def profile():

    dec = float(request.args.get('dec'))
    ra  = float(request.args.get('ra'))
    print('>>> Requested slice centered at ra, dec: ', ra, dec)
    
    #ima = get_slice_simple( vol_den, ra=190, dec=dec, ra_delta=80, n_ra=256, n_z=256)

    ima = get_slice_correct(vol_den, ra=ra, dec=dec, ra_delta=80, n_ra=256, n_z=256)

    fig, ax = plt.subplots(1)
    plt.plot(ima[:,127],color='red')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    
    return send_file(img, mimetype='image/png')

#--------------------------------------------
#
#--------------------------------------------
@app.route('/fig/', methods=['GET'])
def fig():

    dec = float(request.args.get('dec'))
    ra  = float(request.args.get('ra'))
    print('>>> Requested slice centered at ra, dec: ', ra, dec)
    
    #ima = get_slice_simple( vol_den, ra=190, dec=dec, ra_delta=80, n_ra=256, n_z=256)

    ima = get_slice_correct(vol_den, ra=ra, dec=dec, ra_delta=80, n_ra=256, n_z=256)

    fig, ax = plt.subplots(1)
    plt.imshow(ima,interpolation="none", origin='lower',cmap='gray')
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)
    
    return send_file(img, mimetype='image/png')


#--------------------------------------------
#
#--------------------------------------------
@app.route('/image/')
def images():
    return render_template("image.html")


if __name__ == '__main__':
    app.run(debug=True)

