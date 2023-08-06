import numpy as np
import sys

import struct
import array as arr

from interp3d import interp_3d
from scipy.interpolate import RegularGridInterpolator

#--------------------------------------------
#
#--------------------------------------------
def read_fvolume(filename):
    
    F = open(filename,'rb')

    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)
    
    den = arr.array('f')
    den.fromfile(F,sizeX*sizeY*sizeZ)    
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.float32)    
    
    return den

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
    
    den = arr.array('b')
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

def sdss_get_los(ra,dec,z_lim1,z_lim2,nz, mapp):
    
    dz = (z_lim2-z_lim1)/nz
    
    #--- Create array of redshifts
    z_arr = np.arange(z_lim1,z_lim2, dz,dtype='float')

    #--- Get XYZ positions of sampling points
    xl,yl,zl = sdss_radec_to_xyz(ra, dec, z_arr, mapp)

    #--- Unitary position and add grid offset
    xlu = xl/mapp['max_l'] + mapp['x0']
    ylu = yl/mapp['max_l'] + mapp['y0']
    zlu = zl/mapp['max_l'] + mapp['z0']

    return xlu, ylu, zlu


#--------------------------------------------
# Convert radec to xyz. Note the code rotates the positions along
#   the z axis following my own convention.
#--------------------------------------------
def sdss_radec_to_xyz(ra, dec, zred, mapp):
    
    #--- Unitary vector
    dec2 = np.deg2rad(dec)
    ra2  = np.deg2rad(ra)
    cx = np.cos(dec2)*np.cos(ra2)
    cy = np.cos(dec2)*np.sin(ra2)
    cz = np.sin(dec2)

    #--- Distance [Mpc]
    x_Mpc  =  cx * zred * (mapp['c_l']/mapp['H_o'])
    y_Mpc  =  cy * zred * (mapp['c_l']/mapp['H_o'])
    z_Mpc  =  cz * zred * (mapp['c_l']/mapp['H_o'])

    #--- Rotate points to (custom) survey box
    x,y,z = rotate_around_axis(x_Mpc,y_Mpc,z_Mpc,'z',-90.0)

    return x,y,z


#--------------------------------------------
#
#--------------------------------------------
def get_sphere_simple(vol_den, interp, target, n_lon, n_lat, radius, mapp):
    
    radius_z = radius * mapp['H_o'] / mapp['c_l']
        
    #--- RA range
    ra_arr  = np.arange(0,360, 360/n_lon)

    #--- RA range
    dec_arr  = np.arange(-90, 90, 180/n_lat)

    #--- Galaxy position inside grid
    xg,yg,zg = sdss_radec_to_xyz(target['ra'],target['dec'],target['z'], mapp)
    
    ima = np.zeros((n_lon, n_lat))
    for i in range(n_lon):
        for j in range(n_lat):
            
            #--- Compute 3D coordinates of sphere
            xl,yl,zl = sdss_radec_to_xyz(ra_arr[i],dec_arr[j],radius_z, mapp)
            xt = xl+xg
            yt = yl+yg
            zt = zl+zg
            
            #--- Unitary position and add grid offset
            xl = xt/mapp['max_l'] + mapp['x0']
            yl = yt/mapp['max_l'] + mapp['y0']
            zl = zt/mapp['max_l'] + mapp['z0']

            #--- Sample volume
            #vol_den_ijk = vol_den[(xl*mapp['max_l']).astype(int),(yl*mapp['max_l']).astype(int),(zl*mapp['max_l']).astype(int)]
            vol_den_ijk = interp((xl*mapp['max_l'], yl*mapp['max_l'], zl*mapp['max_l']))
            ima[i,j] = vol_den_ijk

    ima = np.transpose(ima)
    return ima



#--------------------------------------------
#
#--------------------------------------------
def get_reds_slice_simple(vol_den, interp, target, ra_delta, dec_delta, n_ra, n_dec, mapp):
     
    #--- RA range
    ra1 = target['ra'] - ra_delta/2
    ra2 = target['ra'] + ra_delta/2
    ra_arr  = np.arange(ra1,ra2, (ra2-ra1)/n_ra)

    #--- RA range
    dec1 = target['dec'] - dec_delta/2
    dec2 = target['dec'] + dec_delta/2
    dec_arr  = np.arange(dec1,dec2, (dec2-dec1)/n_dec)

    ima = np.zeros((n_ra, n_dec))
    for i in range(n_ra):
        for j in range(n_dec):
            #--- Compute 3D coordinates of LOS
            xl,yl,zl = sdss_radec_to_xyz(ra_arr[i],dec_arr[j], target['z'], mapp)
            #--- Unitary position and add grid offset
            xl = xl/mapp['max_l'] + mapp['x0']
            yl = yl/mapp['max_l'] + mapp['y0']
            zl = zl/mapp['max_l'] + mapp['z0']

            #--- Sample volume
            #vol_den_ijk = vol_den[(xl*mapp['max_l']).astype(int),(yl*mapp['max_l']).astype(int),(zl*mapp['max_l']).astype(int)]
            vol_den_ijk = interp((xl*mapp['max_l'], yl*mapp['max_l'], zl*mapp['max_l']))            
            ima[i,j] = vol_den_ijk

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

def get_dec_slice(vol_den, interp, target, ra_delta, reds_delta, n_ra, n_reds, mapp):
   
    #z1, z2 = reds, reds-reds_delta/2, reds+reds_delta/2  # Initial redshift
    z1 = target['z']-reds_delta/2
    z2 = target['z']+reds_delta/2
    z_arr = np.arange(z1,z2,(z2-z1)/n_reds)

    #--- Construct correct radec slice
    ra_new, dec_new =  make_dec_slice(target['ra'], target['dec'], ra_delta, n_ra)

    ima = np.zeros((n_ra, n_reds))
    for i in range(n_ra):
        #--- Compute 3D coordinates of LOS
        xl,yl,zl = sdss_get_los(ra_new[i],dec_new[i],z1,z2,n_reds, mapp)
        #--- Sample volume with LOS
        #ima[i,:] = np.power(vol_den[(xl*mapp['max_l']).astype(int),(yl*mapp['max_l']).astype(int),(zl*mapp['max_l']).astype(int)],0.2)
        for j in range(n_reds):
            vol_den_ijk = interp((xl[j]*mapp['max_l'], yl[j]*mapp['max_l'], zl[j]*mapp['max_l']))  
            ima[i,j] = vol_den_ijk

    ima = np.transpose(ima)
    
    return ima

#--------------------------------------------
#   Makes a slice at ra
#--------------------------------------------
def make_ra_slice(_ra, _dec, _dec_delta, _dec_n):
    #--- Make array of ra positions at equator
    dec1 = _dec - _dec_delta/2
    dec2 = _dec + _dec_delta/2
    decs  = np.arange(dec1,dec2, (dec2-dec1)/_dec_n)    
    ras   = np.zeros(_dec_n) + _ra
    
    return ras, decs

#--------------------------------------------
#
#--------------------------------------------
def get_ra_slice(vol_den, interp, target, dec_delta, reds_delta, n_dec, n_reds, mapp):
   
    z1 = target['z']-reds_delta/2
    z2 = target['z']+reds_delta/2
    z_arr = np.arange(z1,z2,(z2-z1)/n_reds)

    #--- Construct correct radec slice
    ra_new, dec_new =  make_ra_slice(target['ra'], target['dec'], dec_delta, n_dec)    
    
    ima = np.zeros((n_dec, n_reds))
    for i in range(n_dec):
        #--- Compute 3D coordinates of LOS
        xl,yl,zl = sdss_get_los(ra_new[i],dec_new[i],z1,z2,n_reds, mapp)
        #--- Sample volume with LOS
        #ima[i,:] = np.power(vol_den[(xl*mapp['max_l']).astype(int),(yl*mapp['max_l']).astype(int),(zl*mapp['max_l']).astype(int)],0.2)
        for j in range(n_reds):
            vol_den_ijk = interp((xl[j]*mapp['max_l'], yl[j]*mapp['max_l'], zl[j]*mapp['max_l']))  
            ima[i,j] = vol_den_ijk

    ima = np.transpose(ima)
    return ima
