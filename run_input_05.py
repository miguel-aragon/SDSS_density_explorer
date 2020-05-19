#
#
#  Working version, uses fast interpolation
#
#
#  Install interp3D library:
#     https://github.com/jglaser/interp3d
#
#  SDSS largest survey area at COMA cluster's redshift:
#  slice_reds/?ra_delta=180&dec_delta=100&n_ra=256&n_dec=256&ra=180&dec=35&reds=0.0231
#
#
#  
#

import numpy as np
import sys
from io import BytesIO
from flask import Flask, render_template, send_file, make_response, request

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from interp3d import interp_3d
from scipy.interpolate import RegularGridInterpolator

from compute import *


#========================================================================
#
#   Read dataset and prepare for interpolation
#
#========================================================================

PATH_DEN = '/Users/miguel/Desktop/SDSS_density_explorer/Data/'
#FILE_DEN = 'DR13_D_all-01.masked.fvol'
FILE_DEN = 'DR13_D_all_random_boundary.ENS-256.SIG-0.5.fvol'



#--- Read volume as global variable
print('>>> Reading volume...')
vol_den = read_fvolume(PATH_DEN + FILE_DEN)
print('>>>    ready!')


#np.save(FILE_DEN+'.npy', vol_den)

#--- Cython implementation of interpolation requires float_t (double). TODO: patch to use float
vol_den = np.asfarray( vol_den, dtype='float' )

#--- Hard-coded grid size, bad programming.
x = np.linspace(0,1023,1024)
y = np.linspace(0,511,512)
z = np.linspace(0,511,512)

X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
interp = interp_3d.Interp3D(vol_den, x,y,z)


#========================================================================
#
#   Flask server
#
#========================================================================

app = Flask(__name__)


#--------------------------------------------
#
#  Line of sight profile at ra,dec
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
#  Polar ring:
#  http://127.0.0.1:5000/sky_sphere/?n_lon=256&n_lat=128&ra=157.0802&dec=62.5840&reds=0.0178&radius=20
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
#  RA slice (DEC,z)
#  -------
#  VOID_04
#  http://127.0.0.1:5000/slice_ra/?dec_delta=30&n_dec=64&reds_delta=0.01&n_reds=64&ra=155.6469&dec=45.6392&reds=0.020
#  Polar ring:
#  http://127.0.0.1:5000/slice_ra/?dec_delta=30&reds_delta=0.01&n_dec=64&n_reds=64&ra=157.0802&dec=62.5840&reds=0.0178
#
#--------------------------------------------
@app.route('/slice_ra/', methods=['GET'])
def slice_ra():

    #--- Speed of light
    c_l = 299792.458

    ra          = float(request.args.get('ra'))
    dec         = float(request.args.get('dec'))
    reds        = float(request.args.get('reds'))
    dec_delta   = float(request.args.get('dec_delta'))
    reds_delta  = float(request.args.get('reds_delta'))
    n_dec       = int(request.args.get('n_dec'))
    n_reds      = int(request.args.get('n_reds'))
    print('>>> Requested slice centered at ra, dec, z: ', ra, dec, reds)
    
    ima = get_ra_slice(vol_den, interp, ra=ra, dec=dec, reds=reds, dec_delta=dec_delta, reds_delta=reds_delta, n_dec=n_dec, n_reds=n_reds)

    x1 = -dec_delta/2
    x2 =  dec_delta/2
    y1 = -(reds_delta/2)*c_l
    y2 =  (reds_delta/2)*c_l
    print('plot ', x1,x2,y1,y2)
    fig, ax = plt.subplots(1)
    plt.rcParams['figure.figsize'] = [6, 8]
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel('km/s')
    plt.imshow(ima,interpolation="none", origin='lower',cmap='terrain',extent=[x1,x2,y1,y2],aspect='auto')
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
#  Polar ring:
#  http://127.0.0.1:5000/slice_dec/?ra_delta=30&reds_delta=0.01&n_ra=64&n_reds=64&ra=157.0802&dec=62.5840&reds=0.0178

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
    plt.rcParams['figure.figsize'] = [6, 8]
    ax.set_xlabel(r'$\alpha$')
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
#  COMA
#  http://127.0.0.1:5000/slice_reds/?ra_delta=30&dec_delta=30&n_ra=64&n_dec=64&ra=194.9529&dec=27.9805&reds=0.0231
#  Polar ring:
#  http://127.0.0.1:5000/slice_reds/?ra_delta=30&dec_delta=30&n_ra=64&n_dec=64&ra=157.0802&dec=62.5840&reds=0.0178
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




if __name__ == '__main__':
    app.run(debug=True)

