import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.optimize import bisect

from mpl_toolkits.mplot3d import Axes3D
from astropy.io import ascii
from astropy.table import Table, Column

import healpy as hp
import seaborn as sns
sns.set_style('ticks')

# Define some constants
# Define length, mass time units

global DUn, MUn, TUn, Erg, c, hp, kb, sb, G, Ro, Mo, pc

G=6.6726e-08
Ro=6.96e10
Mo=1.99e33
# c=2.9979e+10
# pc = 3.0857e18
# DUn = pc/1e3
# MUn = 1e8*Mo
# TUn = np.sqrt(DUn**3/(G*MUn))
# Erg = MUn*DUn**2 / TUn**2
# c  = 2.997e10 * TUn/DUn        #// cm/s   speed of light
# hp = 6.626e-27/(Erg*TUn)     #// erg s  planck
# kb = 1.380e-16 /Erg        #// erg/K  boltzmann
# sb = (5.67e-5 * DUn*DUn*TUn)/Erg #// stephan boltzmann

# class Constants():
#     def __init__(self):
#         self.msun = 1.989e33
#         self.rsun = 6.955e10
#         self.G  = 6.674e-8
#         self.yr = 3.1536e7
#         self.h  = 6.6260755e-27
#         self.kB = 1.380658e-16
#         self.mp = 1.6726219e-24
#         self.me = 9.10938356e-28
#         self.c  = 2.99792458e10
#         self.pc = 3.085677581e18
#         self.au = 1.496e13
#         self.q = 4.8032068e-10
#         self.eV = 1.6021772e-12
#         self.sigmaSB = 5.67051e-5
#         self.sigmaT = 6.6524e-25
#
#
#         print ("Constants defined...")
#         return None
#
#
# c = Constants()


# ============================================
# ============================================
#        Nested polytropes Stuff
# ============================================
# ============================================


def derivs(x,h,n):
#   Solving the ODE   ->  dg/dx = G(g,f,x)  ;  g = df/dx  ;  f = f(x)

#     x       # Current x value
    f = h[0]  # Current f(x) value
    g = h[1]  # Current df/dx value

    G = -( (2./x)*g + f**n )
    dg = G   # dg/dx

    return g , dg    # return derivatives g = df/dx and dg/dx = G(g,f,x), for current step


def RK4(x,dx,h,n):

    step1 = h
    K1 = np.array(derivs(x,step1,n))

    step2 = h + K1*dx/2.
    K2 = np.array(derivs(x,step2,n))

    step3 = h + K2*dx/2.
    K3 = np.array(derivs(x,step3,n))

    step4 = h + K3*dx
    K4 = np.array(derivs(x,step4,n))

    hf= h + (1/6.)*(K1 + 2*(K2 + K3) + K4)*dx

    return hf



def Solve_LaneEmden(n=3.0,dx=1/1e4,Ic=None,tol=5e-4):

    # Solve Lane-Emden equation

    # ==============================
    # Initial/boundary conditions
    # ==============================
    if Ic == None:

        th_0 = 1  # Central density
        dth_0 = 0. # To obtain HyEq dP/dr = 0 at the center
#         th_out = 0. # The pressure outside the star is zero
        r_0 = 1e-5  # initial value for scaled radius

    else:

        th_0 = Ic[0]  # Central density
        dth_0 = Ic[1] # To obtain HyEq dP/dr = 0 at the center
#         th_out = Ic[2] # The pressure outside the star is zero
        r_0 = Ic[2]  # initial value for scaled radius

    # ==============================
    # Initialize variables
    # ==============================

    x = r_0
    h_0 = [th_0,dth_0]
    h = np.array(h_0)

    # ==============================
    # For loop using different polytropic indexes
    # ==============================

    # Solutions xi_0 and dxi_0
    xi_0 = []
    dtheta_0 = []

    x_a = []
    theta_a = []
    dtheta_a = []
    vecsize_a = []

    # Define initial conditions for Lane-Emden eq.
    x = r_0 #(scaled radius)
    h_0 = [th_0,dth_0] #(scaled "density" and its derivative)
    h = np.array(h_0) # variable to be updated ("density" and its derivative)

    # ==============================
    # Integrate Lane Emden eq for nn index
    # ==============================

    while h[0] > tol:

        hf = RK4(x,dx,h,n)
        x = x + dx
        h = hf

        x_a = np.append(x_a,x)
        theta_a = np.append(theta_a,h[0])
        dtheta_a = np.append(dtheta_a,h[1])


    xi_fin = x
    dtheta_fin = h[1]
    xi_0 = np.append(xi_0,x)
    dtheta_0 = np.append(dtheta_0,h[1])

    return x_a, theta_a, dtheta_a, xi_0, dtheta_0


def Get_Nested_Polytropes(Rstar,M_core,xi_core,n1=3.0,n2=1.5,Xe=0.7,Ye=0.3):
    """
    Solves twice the Lane-Emden equation to construct a nested polytrope
    of radius Rstar, and core mass M_core. Usefull for red giants. The star's final mass is determined by M_core, Rstar , xi_core

    Receives: (Rstar,M_core,xi_core,n1=3,n2=1.5,Xe=0.7,Ye=0.3)
    Rstar -> Star's radius (in CGS)
    M_core -> Core's mass (in CGS)
    xi_core -> Core's fraction to be used (0,1)
            (User needs to try different values until the right stellar mass is obtained)
    n1 -> core's polytropic index (3 by default)
    n2 -> envelope's polytropic index (1.5 by default)
    Xe -> H abundance
    Ye -> He abundance

    Returns:  (r1, rho1, M1, K1, r2, rho2, M2, K2)
    r1 -> polytropic core's radius (array).
          Note: Includes entire polytropic star.
    rho1 -> polytropic core's density (array)
    M1 -> polytropic core's mass (array)
    r2 -> polytropic envelope's radius (array).
          Note: Star.
    rho2 -> polytropic envelope's radius (array)
    M2 -> polytropic envelope's radius (array)

    """

    # Empty arrays
    x_b =[]
    theta_b=[]
    dtheta_b=[]
    vecsize_b=[]

    xi_b = []
    theta_b = []
    g_b = []

    # Solve Lane Emden for core with n=n1

    x_a, theta_a, dtheta_a, xi_0, dtheta_0 = Solve_LaneEmden(n1)
    ##Interpolation function for theta for n = 3.0
    theta_a_int_n_3 = interp1d(x_a,theta_a)
    ##Interpolation function for theta for n = 3.0
    dtheta_a_int_n_3 = interp1d(x_a,dtheta_a)


#     # Define stellar mass and radius desired (in CGS)
#     Mstar = 1* Mo
#     Rstar = 25.2 * Ro

#     # Define core mass in CGS
#     M_core = 0.3 * Mo

    # Select a core scaled radius xc which will determine the total mass

    xc = xi_0*xi_core
    xc_theta =theta_a_int_n_3(xc)
    xc_dtheta = dtheta_a_int_n_3(xc)

    #Define polytropic constant using H and He specific weights
    Ax = 1.
    Ay = 4.
    mu2 = (Xe/Ax + Ye/Ay)**(-1)
    mu1 = 8.*mu2

    ##Initial Conditions for the second polytrope

    x_0 = np.sqrt((n1+1)/(n2+1) * xc**2 * xc_theta**(n1-1)) * (mu2/mu1) # initial value for scaled radius
    theta2 = 1.  # Density
    dtheta2 = (n1+1)/(n2+1) * (xc * xc_dtheta / (x_0 * xc_theta)) * (mu2/mu1) # drho

    Init = [theta2,dtheta2,x_0]

    # Get polytropic envelope
    x_b, theta_b, dtheta_b, xi_fin, dtheta_fin = Solve_LaneEmden(n2,Ic=Init)


    # Obtain core radius and scaled variables
    Q = x_0/xi_fin
    R_core = Q * Rstar

    print 'R_core =',R_core[0]/Ro

    a1 = R_core[0] / xc
    rho_c1 = - xc/(3*xc_dtheta) * 3 * M_core / (4 * np.pi * R_core**3)
    K1 = a1**2 * rho_c1**(1. - 1./n1) * 4. * np.pi * G / (n1+1)

    a2 = Rstar / xi_fin
    rho_2 = - M_core / (4 * np.pi * a2**3 * x_0**2 * dtheta2)
    K2 = a2**2 * rho_2**(1 - 1/n2) * 4 * np.pi * G / (n2+1)


#     x_a = np.array(x_a)
#     xi_b = np.array(xi_b)
#     theta_a = np.array(theta_a)
#     theta_b = np.array(theta_b)

    # Get quantities with units
    r1 = a1 * x_a # core radius
    r2 = a2 * x_b # envelope radius

    rho1 = theta_a**(n1) * rho_c1 # core density
    rho2 = theta_b**(n2) * rho_2  # envelope density

    dm_int1 = interp1d(r1,4 * np.pi * r1**2 * rho1) # mass in shells for core
    dm_int2 = interp1d(r2,4 * np.pi * r2**2 * rho2) # mass in shells for envelope

    M1 = integ.quad(dm_int1,1e-5,R_core)
    M2 = integ.quad(dm_int2,R_core,Rstar)

    print 'M_core,   M_envelope,   M_total'
    print M1[0]/Mo,M2[0]/Mo, (M1[0]+M2[0])/Mo

    return r1, rho1, M1, K1, r2, rho2, M2, K2
#


def Get_Polytrope(Mstar,Rstar,nn=3.0):
    """
    Solves the Lane-Emden equation to construct a polytrope
    of radius Rstar, and mass Mstar. Usefull for WDs and He stars.

    Receives: (Mstar,R_star,nn=3)
    Mstar -> Star's radius (in CGS)
    Rstar -> Core's mass (in CGS)
    nn -> star's polytropic index (3 by default)

    Returns:  (r, rho, u, P, M)
    r -> polytropic star's radius (array).
    rho -> polytropic star's density (array)
    u -> polytropic star's speccific internal energy (array)
    P -> polytropic star's pressure (array)
    M -> polytropic star's mass (array)
    """

    # Solve Lane Emden for core with n
    x_a, theta_a, dtheta_a, xi_0, dtheta_0 = Solve_LaneEmden(nn)

    Dn = ((3./xi_0)*(dtheta_0))**(-1)
    Rn = xi_0
    Mn = -xi_0**2 * (dtheta_0)

    # Obtain central density
    rhoc_n = -Dn*(3*Mstar/(4*np.pi*Rstar**3))

    Gamma_n = (nn+1.)/nn

    Kn = (1./(nn+1.))*( (4 * np.pi * G) * (G * Mstar/Mn)**(nn-1.) * (Rstar/Rn)**(3.-nn) )**(1./nn)
    alpha = ( (nn+1)*Kn/(4 * np.pi * G * rhoc_n**((nn-1.)/nn)) )**(1./2.)

    #Assign units to r and rho
    r = x_a * alpha
    rho = rhoc_n*theta_a**nn
    P = Kn*rho**(Gamma_n)

    # We take u to be determined by Gamma=5/3 as used in Gadget
    u = P/(((5./3.) - 1.)*rho)
    M = 4*np.pi*integ.cumtrapz(rho*r**2,r)
    M = np.append(M,M[-1])

    print 'Mstar: ', M[-1]/Mo

    return r,rho,u,P,M

def joinPolytropes(r1, rho1, M1, K1,n1, r2, rho2, M2, K2,n2):

    R_core = r2[0]

    print r1[-1]/Ro
    print 'Core radius = ',R_core/Ro, ' [solar]'
    print M1[0]/Mo,M2[0]/Mo, (M1[0]+M2[0])/Mo
    boolr1 = r1 <= R_core
    boolr2 = r2 > R_core

    print 'Core radius = ',r2[0]/Ro, ' [solar]'

    ########
    #
    ########
    r1arr = np.array(r1[boolr1])
    r2arr = np.array(r2[boolr2])

    rho1arr = np.array(rho1[boolr1])
    rho2arr = np.array(rho2[boolr2])

    P1arr = K1 * rho1arr**(1+1/n1)
    P2arr = K2 * rho2arr**(1+1/n2)

    Gamma1 = (n1+1)/n1
    Gamma2 = (n2+1)/n2

    u1arr = P1arr/((Gamma1 - 1.)*rho1arr)
    u2arr = P2arr/((Gamma2 - 1.)*rho2arr)

    r = np.concatenate((r1arr,r2arr))
    rho = np.concatenate((rho1arr,rho2arr))
    P = np.concatenate((P1arr,P2arr))

    # This U is the one obtained for the original polytropic indexes
    u = np.concatenate((u1arr,u2arr))

    # Get U needed for Gadget (gamma = 5/3)
    u_g = P/((5./3-1)*rho)

    M = 4*np.pi*integ.cumtrapz(rho*r**2,r)
    M = np.append(M,M[-1])

    dM = M[1:] - M[:-1]

    print np.shape(r),np.shape(rho),np.shape(P),np.shape(u),np.shape(M)

    # ===================================================================
    # this Table is used later to create the SPH particle distrubution
    # ===================================================================
    # dat = Table([r,rho,u_g,P,M],names=('r', 'rho', 'u', 'P', 'M', 'u_original'))

    return r,rho,u_g,P,M
