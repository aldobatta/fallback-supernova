# ==========================================================#
#
#  Make hdf5 file with SPH initial conditions for GADGET
#  Specialized for Exploding Stars and binaries
#
# ==========================================================#

# ==========================================================#
#  Import packages and modules needed
# Script in python2

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

import SphericalSPHDist_Commented_AVG as sph # Used to build spherical particle distributions
import BinaryBHOrbit_Commented_AVG as bhb

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
        #Alejandro adds
        self.km = 1e5

        # print ("Constants defined...")
        return None

const = Constants()
# AVG - 07/04/20 - 'c' should be called 'class' so we get class.c instead of const.c for speed of light

# ====================================================#
# Define Filename
# Filename = 'NS_1_33_HE16C_SNf_0_8_Mexp_0_67.hdf5' # Template name
testingFlag = False
if testingFlag:
    Filename = 'test.hdf5'
else:
    Filename = 'NS_1_3_MESA10_SNE_frac_0_6_factor_to_cut_0_5.hdf5'

# ====================================================#
# Initial conditions 
M_c = 1.3*const.msun # Define companion's mass (in the case of a non-stellar companion)
a = 1.4*const.rsun # orbital separation
e = 0.0         # eccentricity

# BH inside star --------------------
Collapsar = True  # Creates a BH/NS surrounded by an envelope using a Stellar profile
mBH = 1.3*const.msun  # Define initial BH/NS mass (removed from the stellar profile)

# SN explosion (and radial velocities) --------------------
SNexplosion = True
SNType = 'Piston'    # Thermal,  Piston or (Lovegrove 2013 -> neutrino mass loss)
SNE_frac = 0.6    # explosion energy in terms of binding energy of the star

M_exp = 0.67*const.msun # Innermost mass where explosion energy is deposited

# Natal kick
useNatalKick = False
natalKick = np.array([300.0,300.0,300.0])*const.km # kick in km/s
# ==========================================================#
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

# -----------------------------------------------------
# Define stellar profile to be read, M_star and R_star
# AVG - 07/04/2020 - I deleted options to make polytropes

# AVG - 07/12/2020 - Default when using Heger
# Readprofile = True
# Profiletype = 'Heger' # Accepts Profiletypes: MESA, ChrisIC, ChrisSN, Heger
# Profilename = './stellarProfiles/35OC@presn'
# M, r ,v ,rho, Omega, jprofile, T, P, u = sph.readfile(Profilename,Profiletype,Rotating=True)

# AVG - 07/12/2020 - Testing MESA
Readprofile = True
Profiletype = 'MESA' # Accepts Profiletypes: MESA, ChrisIC, ChrisSN, Heger
Profilename = './stellarProfiles/MESA_10_0_final_profile.data'
print("Using a ", Profiletype, "stellar profile of name:", Profilename)
M, r ,v ,rho, Omega, jprofile, T, P, u = sph.readfile(Profilename,Profiletype,Rotating=False) 

MapEntireStar = False   # AVG - 07/04/2020 - If true, cut off outer radius if the density is too low on the outer particles
factor_to_cut = 0.4 # Default is 1.0, i.e. all the radius
R_out_to_cut = factor_to_cut*const.rsun
print("MapEntireStar = ", MapEntireStar, "and R_out_to_cut = ",R_out_to_cut)
    
if MapEntireStar:
    M_star = M[-1]  # Use entire star for simulation
    R_star = r[-1]  # Use entire star for simulation
    R_out = R_star
    out_id = len(r) - 1

else:
    R_out = R_out_to_cut  # this will be used as the outermost radius of the star
    out_id = sph.find_nearest(r,R_out)
    M_star = M[out_id]
    R_star = r[out_id]
    print("M_star/Msol = ",M_star/const.msun,"R_star/Rsol = ",R_star/const.rsun)


# ======================================================#
# Decide type of simulation
# ======================================================#
if SNexplosion:
    if SNType == 'Lovegrove':
         M_exp = 0.5*const.msun # Innermost mass lost in neutrinos

# Stellar Rotation --------------------
RigidbodyRot = True    # Assigns constant angular velocity to particles in SPH star
if RigidbodyRot:
    Omega_star = 0.0 * np.sqrt(const.G*M_star/R_star**3)
    # AVG - 07/04/20 - Is Omega_star purposedly set to zero? Delete previously commented lines? 
    
# Binary --------------------
Binary = True    # Creates a Binary (one of them will be an SPH Star)
Tidallock = True  # Only works when Binary is On
if Binary:

    addStar = True  # Adds a Star particle as companion         /Single star particle; i.e. similar to a BH
    addBH = False   # Adds a BH (sink) particle as companion
    
    #=====================
    #Binary properties
    m1 = M_star     # mass of SPH star
    m2 = M_c        # companion's mass

    #=====================
    # Load class to create initial conditions and evolve them
    # bhb_star= bhb.ICs(m1,m2,m3) # AVG: anotate if it is specifically for triples, which probably is

    #=====================
    # Get Period and binary orbit
    # orb,Period = bhb_star.get_IC(m1,m2,a,e,m3,r3,rperi,e3)
    orb,Period = bhb.getBinary2(m1,m2,a,e)

    Omega_orb = 2*np.pi / Period  # Orbital angular velocity

    if Tidallock:
        Omega_star = Omega_orb # tidally lock rotation of SPH star

    # =====================================================================#
    # Define orbital velocities and position (with respect to binary CM)
    # AVG: Primary?
    pSPH = np.array([orb[0],orb[1],0])  # SPH star's position
    vSPH = [orb[2],orb[3],0]            # SPH star's velocity

    # AVG: Companion?
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


# AVG - 07/04/20 - CHECKPOINT
# ======================================================#
# Define Units (if scale_to_units = False) units are CGS)
# ======================================================#

# AVG - 17/04/20 - Making code units
scale_to_units = True
DistUnit = const.rsun
MassUnit = const.msun
TimeUnit = np.sqrt(DistUnit**3 /(const.G*MassUnit))
DensUnit = MassUnit/DistUnit**3
VelUnit = DistUnit/TimeUnit
E_perMassUnit = VelUnit**2
P_Unit = E_perMassUnit*DensUnit
sigFigsToPrint = 4

if scale_to_units:
    print '\n-------------------------------------------'
    print 'Scaling distances by ', round(DistUnit,sigFigsToPrint), ' cm'
    print 'Scaling masses by ', round(MassUnit,sigFigsToPrint), ' g\n'
    print 'G = 1'
else:
    print '\n-------------------------------------------'
    print 'All final data will be in CGS\n'


# ====================================================#
# Special treatment of SPH particles and filename
# ====================================================#

# ====================================================#
# Apply random rotation to healpix shells
# AVG: Check what healpix is
rotshell = True   # Turn on / off

# ====================================================#
# Apply Gaussian distribution to shell's radius

gaussRad = True     # Turn on / off
dr_sigma = 0.1      # 1 sigma of gaussian Dist will be within 0.1 of shell's width
Nsigma = 3.0        # Random Gaussian distribution up to 3 Sigma

# ====================================================#
#  Here we call the functions (actual program)
# ====================================================#
# ====================================================#

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
print 'Particle mass [solar]', P_mass/const.msun
M_shell_min = 12*P_mass
print 'Lowest shell mass [solar]', M_shell_min/const.msun

r_min = R_int(M_shell_min)
print 'r_min =',r_min/const.rsun    

global r_low    # AVG: 17/04/20 - does this really need to be global?

r_low = r_min 
print(r_low)

# AVG: 17/04/20 - Is there a way to test that r_min is sensible? Does it matters?

# ============================================================
# Obtain positions and masses of SPH particles matching rho(r)
# AVG: Can we use this twice to make 2 stars?
xpos,ypos,zpos,mp = sph.getSPHParticles(r_low,P_mass,M_int,rho_int,u_int,R_star,rotshell,gaussRad,Nsigma,dr_sigma,debug=False)


# ======================================================
# Remove gas particles to be replaced by point mass (BH)
# AVG - 17/04/20 - Change "Collapsar" to "makeCore" or so to generalize the function
if Collapsar:
    R_core = R_int(mBH)
    print ''
    print '-'*40
    print 'Removing mass to be replaced by BH'
    print R_core/DistUnit, 'Core radius'
    Mc, N_bdry, xpos, ypos, zpos, mp = sph.remove_gasparticles(xpos,ypos,zpos,mp,1.05*R_core)
    mBH = Mc # this will be the inital mass of the BH

# =============================
# Get SPH particle's properties
# =============================

# AVG: Check what is SNexplosion about
if SNexplosion:
    Min = M[::-1]
    rin = r[::-1]
    E_bind = integ.cumtrapz(-const.G*Min/rin,Min)
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

# AVG: Here probably need another similar function to add the second star
# AVG: Checkout: sph.get_particle_properties
ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = sph.get_particle_properties(mp,pos,pSPH,vSPH,Omega_star,SNe_pm,SNType,M_exp,mBH,rho_int,u_int,R_int)

# AVG
GAS_PARTICLE = 0
STAR_PARTICLE = 4
BLACK_HOLE_PARTICLE = 5

# =============================
# Add BH and star particles
# =============================
# This is adding a star
ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = \
        sph.add_Particle(STAR_PARTICLE,p_star,v_star,m_star,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)

# This is adding a black hole
ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f = \
    sph.add_Particle(BLACK_HOLE_PARTICLE,pSPH,vSPH,mBH,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)


# Alejandro: Add natal kick to exploding star (quick hack)
# Works because there is only one BH particle in the simulation
if useNatalKick:
    indexBH = ptype.index(BLACK_HOLE_PARTICLE)
    print 'The particle type 5 is a black hole. Check particle type:', ptype[indexBH]
    print 'The orbital velocity is [cm/s]', vx_f[indexBH], vy_f[indexBH], vz_f[indexBH]
    print 'The natal kick is [cm/s]', natalKick  
    vx_f[indexBH] = vx_f[indexBH]+natalKick[0]
    vy_f[indexBH] = vy_f[indexBH]+natalKick[1]
    vz_f[indexBH] = vz_f[indexBH]+natalKick[2]
    print 'The new velocity is [cm/s]', vx_f[indexBH], vy_f[indexBH], vz_f[indexBH]



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
