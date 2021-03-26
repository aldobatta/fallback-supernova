# ==========================================================#
#
#  Make hdf5 file with SPH initial conditions for GADGET
#  Specialized for Exploding Stars and binaries
#
# ==========================================================#

# ==========================================================#
#  Import packages and modules needed

import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from astropy.io import ascii
from astropy.table import Table, Column
import healpy as hp
import time
import matplotlib.pyplot as pl

import NestedPolyStar as nps  # Used to obtain polytropic stars
import SphericalSPHDist as sph # Used to build spherical particle distributions

# Time the execution of the program
start = time.time()

# ====================================================#
# Define physical constants

class Constants():
    def __init__(self):
        self.msun = 1.989e33
        self.rsun = 6.955e10
        self.G  = 6.674e-8
        self.yr = 3.1536e7
        self.day = 60*60*24
        self.h  = 6.6260755e-27
        self.kB = 1.380658e-16
        self.mp = 1.6726219e-24
        self.me = 9.10938356e-28
        self.c  = 2.99792458e10
        self.pc = 3.085677581e18
        self.au = 1.496e13
        self.q = 4.8032068e-10
        self.eV = 1.6021772e-12
        self.sigmaSB = 5.67051e-5
        self.sigmaT = 6.6524e-25

        # print ("Constants defined...")
        return None

c = Constants()

# ====================================================#
#                                                     #
#   Define general properties of the simulation,      #
#   number of particles of all types                  #
#   and configuration of the system                   #
#   (BH-Star binary, Star-Star Binary, single star)   #
#                                                     #
# ====================================================#

# ====================================================#
# Define number of SPH particles

N_p = int(1e6)  # number of SPH particles

Npstring = str(N_p)
N_k = len(Npstring) - 4
N_p_code = Npstring[0]+'0'*N_k+'k'  # used for naming the file


#==========================================================#
# Define what kind of star we want to map with SPH particles
#
# if MakePolytrope = True:
#   the user needs to define M_star and R_star for the code
#   to create a polytropic Star (using Lane-Emden).
#   And specify is a Nested polytrope is wanted or not
#
# if MakePolytrope = False:
#   the user needs to specify what kind of stellar profile
#   will be read and its name.
#   Also define R_out which defines how far out we wil map
#   with SPH particles (low resolution needs R_out < R_star)
#==========================================================#

MakePolytrope = False
NestedPoly = False

if MakePolytrope:
    n1 = 1.5        # Polytropic index of the star
    M_star = 28*c.msun # Mass of the polytropic star
    R_star = 0.7*c.rsun # Radius of the polytropic star

    if NestedPoly:

        n2 = 3.0 # Envelope's Polytropic index (results in R_core ~ 0.3 c.rsun)
        xi_core = 0.7895

        # This polytrope creates a core with the "same" size as a 25 Rsun, 1 Msun star (MESA)
        # n2 = 2.135 # Envelope's Polytropic index (results in R_core ~ 0.02 c.rsun)
        # xi_core = 0.84

        M_core = 0.3*c.msun # Core's mass
else:

# -----------------------------------------------------
# Define stellar profile to be read, M_star and R_star

    Readprofile = True
    Profiletype = 'Heger' # Accepts Profiletypes: MESA, ChrisIC, ChrisSN, Heger
    Profilename = '../stellarprofiles/35OC@presn'

    M, r ,v ,rho, Omega, jprofile, T, P, u = sph.readfile(Profilename,Profiletype,Rotating=True)

    MapEntireStar = False

    if MapEntireStar:
        M_star = M[-1]  # Use entire star for simulation
        R_star = r[-1]  # Use entire star for simulation
        R_out = R_star
        out_id = len(r) - 1

    else:
        R_out = 0.5*c.rsun  # this will be used as the outermost radius of the star
        out_id = sph.find_nearest(r,R_out)
        M_star = M[out_id]
        R_star = r[out_id]


# ======================================================#
# Decide type of simulation
# ======================================================#

# BH inside star --------------------
Collapsar = True  # Creates a BH surrounded by an envelope using a Stellar profile
if Collapsar:
    mBH = 3*c.msun  # Define initial BH mass (removed from the stellar profile)

# SN explosion (and radial velocities) --------------------
SNexplosion = True
SNType = 'Piston'    # Thermal,  Piston or (Lovegrove 2013 -> neutrino mass loss)
if SNexplosion:
    SNEnergy = 5e51    # explosion energy in ergs (only used if not defined by SNE_frac)
    SNE_frac = 0.1    # explosion energy in terms of binding energy of the star
    M_exp = 1.0*c.msun # Innermost mass where explosion energy is deposited
    if SNType == 'Lovegrove':
         M_exp = 0.5*c.msun # Innermost mass lost in neutrinos

# Stellar Rotation --------------------
RigidbodyRot = True    # Assigns constant angular velocity to particles in SPH star
if RigidbodyRot:
    #Omega_star = 2*np.pi / (100.0*c.day)  # define constant angular velocity
    #Omega_star = np.sqrt(c.G*(M_star+15.0)*c.msun/(2.28*c.rsun)**3)   # SS messing up AA's code
    Omega_star = 0.0 * np.sqrt(c.G*M_star/R_star**3)
    
# Binary --------------------
Binary = True    # Creates a Binary (one of them will be an SPH Star)
Tidallock = True  # Only works when Binary is On
if Binary:
    import BinaryBHOrbit as bhb

    addStar = True  # Adds a Star particle as companion
    addBH = False   # Adds a BH (sink) particle as companion
    M_c = 15*c.msun # Define companion's mass

    #=====================
    #Binary properties
    m1 = M_star     # mass of SPH star
    m2 = M_c        # companion's mass
    a = 3.8*c.rsun # orbital separation
    e = 0.0         # eccentricity

    #===========================
    #3rd body trajectory (not used but must be defined)
    m3 = M_star
    r3 = 800 * c.rsun       # distance to binary's CM
    rperi = 100 * c.rsun    # desired perihelium distance
    e3 = 1                  # eccentricity

    #=====================
    # Load class to create initial conditions and evolve them
    bhb_star= bhb.ICs(m1,m2,m3)

    #=====================
    # Get Period and binary orbit
    orb,Period = bhb_star.get_IC(m1,m2,a,e,m3,r3,rperi,e3)

    Omega_orb = 2*np.pi / Period  # Orbital angular velocity

    if Tidallock:
        Omega_star = Omega_orb # tidally lock rotation of SPH star

    # =====================================================================#
    # Define orbital velocities and position (with respect to binary CM)
    pSPH = np.array([orb[0],orb[1],0])  # SPH star's position
    vSPH = [orb[2],orb[3],0]            # SPH star's velocity

    p_star = np.array([orb[4],orb[5],0.0]) # companion's position
    v_star = [orb[6],orb[7],0]             # companion's velocity
    m_star = M_c

    # =====================================================================#
    # Define orbital velocity and position of third body (with respect to binary CM)
    # p3 = np.array([orb[8],orb[9],0])   # Relative position to binary CM
    # v3 = np.array([orb[10],orb[11],0]) # Relative velocity for all particles

else:
    pSPH = np.array([0,0,0])
    vSPH = [0,0,0]



# ======================================================#
# Define Units (if scale_to_units = False) units are CGS)
# ======================================================#

scale_to_units = True
DistUnit = c.rsun
MassUnit = c.msun
TimeUnit = np.sqrt(DistUnit**3 /(c.G*MassUnit))
DensUnit = MassUnit/DistUnit**3
VelUnit = DistUnit/TimeUnit
E_perMassUnit = VelUnit**2
P_Unit = E_perMassUnit*DensUnit

if scale_to_units:
    print '\n-------------------------------------------'
    print 'Scaling distances by ', round(DistUnit,4), ' cm'
    print 'Scaling masses by ', round(MassUnit,4), ' g\n'
    print 'G = 1'
else:
    print '\n-------------------------------------------'
    print 'All final data will be in CGS\n'


# ====================================================#
# Special treatment of SPH particles and filename
# ====================================================#

# ====================================================#
# Apply random rotation to healpix shells

rotshell = True   # Turn on / off

# ====================================================#
# Apply Gaussian distribution to shell's radius

gaussRad = True     # Turn on / off
dr_sigma = 0.1      # 1 sigma of gaussian Dist will be within 0.1 of shell's width
Nsigma = 3.0        # Random Gaussian distribution up to 3 Sigma

# ====================================================#
# Define Filename

# Filename = 'Star_Binary_'+N_p_code+'.hdf5'
if SNexplosion:
    explosion = SNType
else:
    explosion = ''
    
# Name file
if Binary:
    Filename = 'SNBinary_'+N_p_code+explosion+'.hdf5'
    Filename = 'C_5_vhr10.hdf5'
else:
    Filename = 'Star_collapse_'+N_p_code+explosion+'.hdf5'
# print ''
print '================================================================'
print 'Creating initial conditions with roughly '+str(N_p)+' particles'
# print 'in file '+Filename
print '================================================================'

# ==========================================================#
# ====================================================#
#  Here we call the functions (actual program)
# ====================================================#
# ==========================================================#

# ====================================================#
#  First we build a Polytropic star if needed

if MakePolytrope:

    if NestedPoly:

        r1,rho1,M1,K1,r2,rho2,M2,K2 = nps.Get_Nested_Polytropes(R_star,M_core,xi_core,n1=3.0,n2=1.5,Xe=0.7,Ye=0.3)
        r,rho,u,P,M = nps.joinPolytropes(r1,rho1,M1,K1,n1,r2,rho2,M2,K2,n2)

    else:

        r,rho,u,P,M = nps.Get_Polytrope(M_star,R_star,n1)

    # ===================================================================
    # this Table is used later to create the SPH particle distribution

    dat = Table([r,rho,u,P,M],names=('r', 'rho', 'u', 'P', 'M'))
    ascii.write(dat,'PolyStar_cgs.dat')

    if scale_to_units:
        dat_solar = Table([r/DistUnit,rho/DensUnit,u/E_perMassUnit,P/P_Unit,M/MassUnit],names=('r', 'rho', 'u', 'P', 'M'))
        ascii.write(dat_solar,'PolyStar_scaled.dat')

else:

    # print np.shape(r), np.shape(rho), np.shape(u), np.shape(P), np.shape(M)
    dat = Table([r,rho,u,P,M],names=('r', 'rho', 'u', 'P', 'M'))
    ascii.write(dat,'StarProfile_cgs.dat')

    if scale_to_units:
        dat_solar = Table([r/DistUnit,rho/DensUnit,u/E_perMassUnit,P/P_Unit,M/MassUnit],names=('r', 'rho', 'u', 'P', 'M'))
        ascii.write(dat_solar,'StarProfile_scaled.dat')


# ====================================================#
# Get interpolated profiles to build SPH star

rho_int = interp1d(dat['r'],dat['rho'],bounds_error=False, fill_value=dat['rho'][-1])
u_int = interp1d(dat['r'],dat['u'],bounds_error=False, fill_value=dat['u'][0])
M_int = interp1d(dat['r'],dat['M'],bounds_error=False, fill_value=dat['M'][-1])
R_int = interp1d(dat['M'],dat['r'],bounds_error=False, fill_value=dat['r'][0])
# Omega_int = interp1d(dat['r'],dat['r'],bounds_error=False, fill_value=dat['r'][0])


# =============================
# Build SPH star using healpix
# =============================

P_mass = M_star/N_p
print 'Particle mass [solar]', P_mass/c.msun
M_shell_min = 12*P_mass
print 'Lowest shell mass [solar]', M_shell_min/c.msun

r_min = R_int(M_shell_min)
print 'r_min =',r_min/c.rsun

global r_low

r_low = r_min

# ============================================================
# Obtain positions and masses of SPH particles matching rho(r)

xpos,ypos,zpos,mp = sph.getSPHParticles(r_low,P_mass,M_int,rho_int,u_int,R_star,rotshell,gaussRad,Nsigma,dr_sigma,debug=False)


# ======================================================
# Remove gas particles to be replaced by point mass (BH)

if Collapsar:
    R_core = R_int(mBH)
    print ''
    print '-'*40
    print 'Removing mass to be replaced by BH'
    print R_core/DistUnit, 'Core radius'
    Mc, N_bdry, xpos, ypos, zpos, mp = sph.remove_gasparticles(xpos,ypos,zpos,mp,1.05*R_core)
    mBH = Mc # this will be the mass of the BH

# =============================
# Get SPH particle's properties
# =============================

if SNexplosion:
    Min = M[::-1]
    rin = r[::-1]
    E_bind = integ.cumtrapz(-c.G*Min/rin,Min)
    E_bind = E_bind[::-1]
    E_bind = np.append(E_bind,E_bind[-1])
    Eb_int = interp1d(dat['M'],E_bind,bounds_error=False, fill_value=E_bind[-1])

    Eb_env = Eb_int(Mc)
    print Eb_env, "Envelope's Binding energy (outside BH)"

    SNEnergy = SNE_frac*Eb_env  # SN energy
    SNe_pm = SNEnergy/M_exp     # SN energy per unit mass

    if SNType == 'Lovegrove':
        R_core = R_int(mBH + M_exp)
        print ''
        print '-'*40
        print 'Removing mass from neutrino emission Lovegrove (2013)'
        print R_core/DistUnit, 'Core radius'
        Mc, N_bdry, xpos, ypos, zpos, mp = sph.remove_gasparticles(xpos,ypos,zpos,mp,1.05*R_core)

        print Eb_int(mBH + Mc), "Envelope's Binding energy (after neutrino mass loss)"
        print ''
        print 1 - Eb_int(mBH + Mc)/Eb_env,'Fraction of binding energy removed (after mass loss / entire envelope)'

else:
    SNe_pm = 0
    SNType = 'None'
    M_exp = 0




# Merge positions into single array

pos = np.zeros((len(xpos),3))
pos[:,0] = xpos
pos[:,1] = ypos
pos[:,2] = zpos

ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = sph.get_particle_properties(mp,pos,pSPH,vSPH,Omega_star,SNe_pm,SNType,M_exp,mBH,rho_int,u_int,R_int)


# =============================
# Add BH and star particles
# =============================

if Binary:

    if addStar:
        ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = \
        sph.add_Particle(4,p_star,v_star,m_star,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)

    if addBH:
        ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = \
        sph.add_Particle(5,p_star,v_star,m_star,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)

if Collapsar:
    ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = \
    sph.add_Particle(5,pSPH,vSPH,mBH,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)


# =============================
# Save data into an hdf5 file
# =============================

data = Table([ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f]\
             , names=('type','id', 'm', 'x', 'y', 'z', 'vx', 'vy' , 'vz', 'u', 'hsml', 'rho'))

sph.make_hdf5_from_table(data,Filename,scale_to_units,DistUnit,MassUnit)

# ===========================================
## data is already scaled to preferred units
# ===========================================

print ''
print '================================='
print 'Done Creating Initial Conditions'
print 'in file '+Filename
print '================================='

print '\nCreating the ICs took ',round(time.time()- start,4),' seconds \n (' ,round((time.time()- start)/60.,4), ' minutes)'
