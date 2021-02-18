
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
import numpy as np
import pandas as pd
import scipy.integrate as integ
from astropy.io import ascii
from scipy import interpolate
import scipy.stats as stats
from astropy.table import Table, Column
import readsnap as rs
# reload(rs)

G = 6.6726e-08
Ro = 6.96e10
Mo = 1.99e33
c = 2.9979e+10 
day = 60*60*24
yr = 356.25*day

DistUnit = Ro
MassUnit = Mo
TimeUnit = np.sqrt(DistUnit**3/(G*MassUnit))
VelUnit = DistUnit/TimeUnit
AngMomUnit = DistUnit*VelUnit*MassUnit
SpinUnit = AngMomUnit*c/(G*Mo**2)

Tday = TimeUnit/(60*60*24)

def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
#     return array[idx]
        return idx




def Get_dynamics(filename):

    """Getting the data from the dynmaics file"""
    
    dyn = np.genfromtxt(filename)    
    
    colnames = ('t','Macc_bh','Engy_bh','PaccX_bh','PaccY_bh','PaccZ_bh','LaccX_bh','LaccY_bh' \
                ,'LaccZ_bh','M_gas','X_gas','Y_gas','Z_gas','PX_gas','PY_gas','PZ_gas' \
                ,'LX_gas','LY_gas','LZ_gas', 'M_star','X_star','Y_star','Z_star','PX_star' \
                ,'PY_star','PZ_star','LX_star','LY_star','LZ_star','M_bh','X_bh','Y_bh' \
                ,'Z_bh','PX_bh','PY_bh','PZ_bh','LX_bh','LY_bh','LZ_bh','Macc_star', 'Engy_star' \
                ,'PaccX_star','PaccY_star','PaccZ_star','LaccX_star','LaccY_star' \
                ,'LaccZ_star','LaccX_starCM','LaccY_starCM','LaccZ_starCM','LaccX_bhCM' \
                ,'LaccY_bhCM','LaccZ_bhCM','rp')
    
    print(np.shape(dyn), len(colnames))
    dat = Table(dyn,names=colnames)
    

    return dat



def snapdata(path,snap):
    data = rs.readsnap(path,snap,0)
    dataBH = rs.readsnap(path,snap,5,skip_bh = 1)
    dataSTAR = rs.readsnap(path,snap,4)
    
    return data,dataBH,dataSTAR

def vrad(data,dataBH):
    
    # get position and velocity from bh frame
    posgasX = data['p'][:,0] - dataBH['p'][0,0] 
    posgasY = data['p'][:,1] - dataBH['p'][0,1] 
    posgasZ = data['p'][:,2] - dataBH['p'][0,2] 

    velgasX = data['v'][:,0] - dataBH['v'][0,0] 
    velgasY = data['v'][:,1] - dataBH['v'][0,1] 
    velgasZ = data['v'][:,2] - dataBH['v'][0,2] 

    # radius and radial velocity
    R = np.sqrt(posgasX**2 + posgasY**2 + posgasZ**2)
    Vrad  = (velgasX*posgasX + velgasY*posgasY + velgasZ*posgasZ)/R
    
    # Angular momentum of each particle
    A = (posgasX*velgasY - posgasY*velgasX)*data['m']
    
    # bin for plotting
    nbin = 400
    Vradb, Rb, bin_id = stats.binned_statistic(R,Vrad,bins=nbin,statistic='mean')    
    Mbin, Rb, bin_id = stats.binned_statistic(R,data['m'],statistic='sum',bins=nbin)
    Abin, Rb, bin_id = stats.binned_statistic(R,A,statistic='sum',bins=nbin)
    Rave = (Rb[:-1] + Rb[1:])*0.5
    
    #getting bh data
    Mbh0 = dataBH['m']

    return R,Vrad,A,Rave,Vradb,(np.cumsum(Mbin)+Mbh0),Abin

def escvel(data,dataBH):
    
    R,Vrad,A,Rave,Vradb,Mbin,Abin = vrad(data,dataBH)
    vesc = np.sqrt(2*G*Mbin*Mo/(Rave*Ro))

    return R,Vrad,A,Rave,Vradb,Mbin,vesc,Abin


def get_r_isco(a):
    
    w1 = 1. + (1. - a**2.)**(1./3.) * ((1. + a)**(1./3.) + (1 - a)**(1./3.))
    w2 = (3.*a**2 + w1**2.)**(1./2.)
    
    # z = r_isco/M
    z = 3 + w2 - ((3 - w1)*(3 + w1 + 2*w2))**(1./2.)
    return z


def disk(Mdirect,Mbound,a_direct):
    
    # get new r_ISCO
    z_1 = get_r_isco(a_direct)
    Mfinal = Mdirect + Mbound
    z = (Mdirect/Mfinal)**2 * z_1
    
    afinal = 1./3. * z**(1./2.) * (4 - np.sqrt(3.*z - 2) )
    
    
    return afinal

    
# function for finding bound particles
def spincal(path,snap,dyn):                         # path is the directory of the snapshots
                                                      # snap is int number of last snapshot
                                                      # dyn is the dynamics file
                                                      
    #get data from snapshot
    data,dataBH,dataSTAR = snapdata(path,snap)
    
    #get BINNED radius, velocity, masscoordinate and escape velocity 
    R,Vrad,A,Rave,Vradbin,M,vesc,Abin = escvel(data,dataBH)
    
    # bound material has velocity below escape velocity
    mfunc    = interpolate.interp1d(Rave, M,bounds_error=None,fill_value='extrapolate')
    vesc     = np.sqrt(2*G*mfunc(R)*Mo/(R*Ro))
    
    bound    = Vrad*VelUnit<vesc
    
    # Angular momentum in bound particles
    Ab    = sum(np.array(A[bound]))
    
    # bound particles with velocity below escape
    Mbound = sum(data['m'][bound])
    print('total bound mass',Mbound)

    
    # find amount of already acrreted angular momentum from dyn file
    idx = find_nearest(dyn['M_bh'],dataBH['m'])
    Mdirect = dataBH['m']
    
    # find bound material within roche lobe
    q = dataBH['m']/(dataSTAR['m'])
    
    f = (0.49*q**(2./3.))/(0.6*q**(2./3.) + np.log(1 + q**(1./3.)))
    separation = np.sqrt((dyn['X_star'][idx]+dyn['X_bh'][idx])**2 + \
                         (dyn['Y_star'][idx]+dyn['Y_bh'][idx])**2 + (dyn['Z_star'][idx]+dyn['Z_bh'][idx])**2)
    r1_lobe = f*separation
    
    # bound mass within roche lobe
    RLbound = np.where((Vrad*VelUnit<vesc) & (R<r1_lobe))
    MRLbound = sum(data['m'][RLbound])
    print('total bound mass in RL',MRLbound)

    #this gives spin:
    Lacc = (np.sqrt(dyn['LaccZ_bh'][idx]**2))*AngMomUnit
    a_direct = Lacc*c/(G*(dyn['M_bh'][idx]*Mo)**2)
    
    # calculate final spin through accretion in disk
    afinal = disk(Mdirect,Mbound,a_direct)
    afinalRL = disk(Mdirect,MRLbound,a_direct)
    
    return afinal,afinalRL,MRLbound,Mdirect



# here is where to put names in..
path  ='/Users/alejandro/Dropbox/Alejandro_CE_SN/Data/NS_MESA10_2021/0129_4/'

dyn = Get_dynamics(path+'dynamics.txt')
snapnumber = 59
spin,spinRL,MRLbound,Mdirect = spincal(path,snapnumber,dyn)
