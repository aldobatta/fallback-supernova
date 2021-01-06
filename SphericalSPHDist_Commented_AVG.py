# ====================================================#
#                                                     #
#   Routines and functions to create a spherically    #
#   symmetric distribution of SPH particles in        #
#   shells, using healpix and the method described    #
#   in Pakmor, 2012, (Stellar Gadget).                #
#                                                     #
#   It includes some variations such as rotating      #
#   shells randomly, and adding a gaussian            #
#   distribution to shell's radii to reduce the       #
#   inter-shell spacing.                               #
#                                                     #
# ====================================================#

import numpy as np
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.optimize import bisect
import matplotlib.pyplot as pl

from astropy.io import ascii
from astropy.table import Table, Column

import healpy as hp

class Constants():
    def __init__(self):
        self.msun = 1.989e33
        self.rsun = 6.955e10
        self.G  = 6.674e-8
        self.yr = 3.1536e7
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

        print ("Constants defined...")
        return None

class HPX():
    def __init__(self,r_low,P_mass,M_int):
        self.r_low = r_low
        self.P_mass = P_mass
        self.M_int = M_int

        print ("Initialized class to build healpix shells")
        return None


    # ============================================
    #   Use Healpix to create SPH particle distribution
    # ============================================

    # Obtain healpix index Nh (Np = 12*Nh^2) needed to have distance
    # between particles = shell width -> (close to the smoothing length)
    def n_hpx(self,r_up):
        # AVG - 24/04/2020 - This multiplies Nh = Np/12 times some delta radius... what are those?
        # AVG - What does n_hpx stand for? In human.
        return np.sqrt(np.pi/12)*(self.r_low + r_up)/(r_up - self.r_low)

    # Obtain healpix index Nh2 (Np = 12*Nh^2) needed to have M_shell = mp * Np = 12*Nh2^2
    def np_mass(self,r_up):
        # np_mass solves for Nh
        return np.sqrt(self.mshell(r_up)/(12.*self.P_mass))

    def find_n(self,r_up):
        # AVG - 24/04/2020 - This returns "something" minues the healpix index calculated in np_mass
        # AVG - does it has radial units?
        return self.n_hpx(r_up) - self.np_mass(r_up)

    def mshell(self,r_up):
        # Mass of the shell in that deltaVolume
        return self.M_int(r_up) - self.M_int(self.r_low)


    # Function that obtains a ring of particles sampling a shell form r_low to r_up
    def get_shell(self,r_low):
        """
        Obtains a spherical shell of npo particles isotropically distributed in a sphere
        using Healpix (Gorski 2005).
        Based on the method by Pakmor (Stellar Gadget).


        Receives:
        r_low -> location of the inner boundary of the shell that will contain the mass represented by the npo particles of mass mp.

        Returns: shellx, shelly, shellz, r_upf, npo, rshell, rhoshell, mshell(r_upf)

        shellx -> position in x of npo points
        shelly -> position in y of npo points
        shellz -> position in z of npo points
        r_upf -> position of upper boundary of shell containing the mass of npo particles
        npo -> number of particles used to represent the mass in the shell
        rshell -> position (radius) of the particles in the shell
        rhoshell -> average density in the shell
        mshell(r_upf) -> mass contained in the shell (and in the npo particles)
        """
        # print 'this is r_low... ',self.r_low/c.rsun
        # bisect.bisect(a, x, lo=0, hi=len(a))
        r_upf = bisect(self.find_n,self.r_low,20*(self.r_low))
        np_f = self.np_mass(r_upf)

        rshell = (self.r_low + r_upf)*0.5
        npo = 12*np_f**2

        # Do rounding
        np_f = pow(2,int(np.log2(np_f)+0.5))
        # AVG - Why is this done twice? No need to define npo 3 lines above?
        npo = 12*np_f**2

        #  healpy.query_disc(nsideint,vecfloat,radiusfloat)
        # Returns pixels whose centers lie within the disk defined by vec and radius (in radians) (if inclusive is False), or which overlap with this disk (if inclusive is True).
        # Parameters
        #     nsideint
        #         The nside of the Healpix map.
        #     vecfloat, sequence of 3 elements
        #         The coordinates of unit vector defining the disk center.
        #     radiusfloat
        #         The radius (in radians) of the disk
        #     Returns
        # ipixint, array
        #     The pixels which lie within the given disk.
        nlist = hp.query_disc(np_f,(0.,0.,0.),np.pi)

        shellx = []
        shelly = []
        shellz = []

        for i in nlist:
            # https://healpy.readthedocs.io/en/latest/generated/healpy.pixelfunc.pix2vec.html
            points = hp.pix2vec(np_f,i,nest=True)
        #     print points

            shellx.append(points[0])
            shelly.append(points[1])
            shellz.append(points[2])

        # Give units
        shellx = rshell*np.array(shellx)
        shelly = rshell*np.array(shelly)
        shellz = rshell*np.array(shellz)

        dp_x = shellx[0]-shellx[1]
        dp_y = shelly[0]-shelly[1]
        dp_z = shellz[0]-shellz[1]

        dist_points = np.sqrt(dp_x**2 + dp_y**2 + dp_z**2)
        # print ''
        # print '----------------------------------'
        # print 'Distance between points (original)'
        # print dist_points/Ro
        # print '----------------------------------'

        # Calculate density of the whole shell
        rhoshell = 3.*self.P_mass*npo/(4*np.pi*(r_upf**3 - self.r_low**3))

        return shellx, shelly, shellz, r_upf, npo, rshell, rhoshell, self.mshell(r_upf)


    ## Rotate shells

    def rotate_shell(self,x_pix,y_pix,z_pix):
        """
        Rotates points a random angle along a random axis.

        Receives:
        x_pix -> array with position X of all points to rotate
        y_pix -> array with position Y of all points to rotate
        z_pix -> array with position Z of all points to rotate

        Returns:

        x_rot -> array containing rotated position X of all points
        y_rot -> array containing rotated position Y of all points
        z_rot -> array containing rotated position Z of all points

        """

        # !==================================================
        # ! Perform a random rotation along a random axis
        # !==================================================

        # random orientation of vector in spherical coordinates
        # will be used as rotation axis
        phiran = np.random.uniform(0,2*np.pi)  # angle of vector projected in XY
        thetaran = np.arccos(2*np.random.uniform(0,1) - 1) # angle with Z axis

        #define rotation axis in cartesian coordinates
        x_axis = np.sin(thetaran)*np.cos(phiran)
        y_axis = np.sin(thetaran)*np.sin(phiran)
        z_axis = np.cos(thetaran)

        # Define random rotation around the rotation axis (from 0->2*pi)
        rangle = np.random.uniform(0,2*np.pi)

        x_rot = (np.cos(rangle)+(1.-np.cos(rangle))*x_axis**2)*x_pix + \
        (x_axis*y_axis*(1.-np.cos(rangle)) - z_axis*np.sin(rangle))*y_pix +\
        (x_axis*z_axis*(1.-np.cos(rangle)) + y_axis*np.sin(rangle))*z_pix

        y_rot = (x_axis*y_axis*(1.-np.cos(rangle)) + z_axis*np.sin(rangle))*x_pix +\
        (np.cos(rangle) + (1.-np.cos(rangle))*y_axis**2)*y_pix +\
        (y_axis*z_axis*(1.-np.cos(rangle))-x_axis*np.sin(rangle))*z_pix

        z_rot = (z_axis*x_axis*(1.-np.cos(rangle)) - y_axis*np.sin(rangle))*x_pix +\
        ( y_axis*z_axis*(1.-np.cos(rangle)) + x_axis*np.sin(rangle) )*y_pix +\
        (np.cos(rangle) + (1.-np.cos(rangle))*z_axis**2)*z_pix

        return x_rot, y_rot, z_rot

    # Give particles gaussian distribution on radius
    # Improved version of gaussian distribution (using arrays)
    def gaussDist2(self,Npoints,mu,sigma,Nsigma=3.0):
        """
        Obtains a gaussian distribution of Npoints around mu = 1.0 with a gaussian width of Nsigma.

        This distribution is used to scale the position of the points in a single shell with radius sigma, to obtain a gaussian distibution in radius around rshell. This in turn introduces noise into the particle distribution and reduces the space between particles in different shells.


        Receives:
        Npoints -> number of particles in the shell, used to obtain the gaussian distribution
        mu -> = 1 points are distributed around this value
        sigma -> location of 1 sigma (determined by the shells width and location (dr_shell/rshell)*dr_sigma )
        Nsigma -> gaussian distribution width desired

        Returns:
        rshift[:Npoints] -> array containing Npoints values in a gaussian distribution around mu



        """
        x_min = -Nsigma*sigma
        x_max = Nsigma*sigma


        xran = np.random.uniform(x_min,2*x_max + mu,100*Npoints)
        yran = np.random.uniform(0,1.0/(sigma*np.sqrt(2*np.pi)),100*Npoints)

        gauss = (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-(xran-mu)**2 / (2*sigma**2))

        bool_acc = yran<gauss

        rshift = xran[bool_acc]

        ratio_accepted = Npoints/float(len(rshift))

        # print len(rshift), Npoints, Npoints/float(len(rshift))

        if ratio_accepted > 1:

            for i in range(int(ratio_accepted) + 1):

                xran = np.random.uniform(x_min,2*x_max + mu,100*Npoints)
                yran = np.random.uniform(0,1.0/(sigma*np.sqrt(2*np.pi)),100*Npoints)
                gauss = (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-(xran-mu)**2 / (2*sigma**2))

                bool_acc = yran < gauss

                rshift = np.append(rshift,xran[bool_acc])

            # print len(rshift)
            # print len(rshift[:Npoints]), Npoints

        # else:
        #
        #     print len(rshift)
        #     print len(rshift[:Npoints]), Npoints

        return rshift[:Npoints]

        # Builds all the shells and put them on top of each other
def getSPHParticles(r_low,P_mass,M_int,rho_int,u_int,Rstar,rotshell,gaussRad,Nsigma,dr_sigma,debug=False):

    hpx = HPX(r_low,P_mass,M_int)

    # Were used for mass control
    Me_shell = 0
    Ms_fin = 0
    Mratio = 0

    count=0  # count number of shells

    while (hpx.r_low < Rstar): # Repeat as long as shells are inside the star

        shellx, shelly, shellz, r_upf, npo, rshell, rhoshell, dMsh = hpx.get_shell(hpx.r_low)

        if (rshell > Rstar):
            break

        # This quantities can be used to check the mass in shells
        M_sh = hpx.P_mass*npo
        Me_shell += M_sh
        dr_shell = r_upf - hpx.r_low

        # move to next shell
        hpx.r_low = r_upf

        # Used for tests (integrated mass)
        Ms_fin += dMsh
        Mratio = dMsh/M_sh # Used to scale particles' masses to match dM_shell

        # Rotate shell randomly
        if rotshell:
            shellx , shelly, shellz = hpx.rotate_shell(shellx,shelly,shellz)

        # Give gaussian distribution
        if gaussRad:
            mu = 1.0
            sigma = (dr_shell/rshell)*dr_sigma  # location of 1 sigma
            rg = hpx.gaussDist2(npo,mu,sigma,Nsigma)
        else:
            rg = 1

        shellx = rg*shellx
        shelly = rg*shelly
        shellz = rg*shellz


        # Store positions and particles' masses into arrays
        if count == 0:
            xpos = shellx
            ypos = shelly
            zpos = shellz

            # particle mass
            mp = np.ones(len(shellx))*(dMsh/npo)

            # This were used for comparing to original particle masses
            rsh = np.array([rshell])
            Msh = np.array([M_sh])
            rhosh = np.array([rhoshell])
            np_shell = np.array([npo])


        else:
            xpos = np.concatenate((xpos,shellx))
            ypos = np.concatenate((ypos,shelly))
            zpos = np.concatenate((zpos,shellz))

            # particle mass
            mp = np.concatenate((mp,np.ones(len(shellx))*(dMsh/npo)))

            # This were used for comparing to original particle masses
            rsh = np.concatenate((rsh,np.array([rshell])))
            Msh = np.concatenate((Msh,np.array([M_sh])))
            rhosh = np.concatenate((rhosh,np.array([rhoshell])))
            np_shell = np.concatenate((np_shell,np.array([npo])))


        count += 1

        # print 'Mfraction'
        # print'------------'
        # print 'using npo*mp ->',Me_shell/Mstar
        # print 'using dM ->',Ms_fin/Mstar


        # print ''

    print ''
    print '============================================='
    print 'Created N =',count, ' shells'
    print 'with a total of Np =',len(xpos),' particles'
    print '============================================='

    # AVG: add printing comparison with initial quantities when debug = True
    if debug:
        return xpos,ypos,zpos,mp,rsh,Msh,rhosh,np_shell
    else:
        return xpos,ypos,zpos,mp

def get_particle_properties(mp,pos,prel,vrel,Omega,SNe_pm,SNType,M_exp,mBH,rho_int,u_int,R_int):
    '''
    Get the SPH particle properties (v, rho, mp, u, h) from their position

    Receives:
    * mp -> Array with particle's masses
    * pos -> 3D array with all positions of SPH particles in a spherical distrubution
    * prel -> 3D vector to add position to all SPH particles with respect to origin
    * vrel -> 3D vector to add velocity to all SPH particles with respect to origin
    * Omega -> add angular velocity to SPH particles


    Returns:
    ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f -> arrays with all SPH particle properties
    '''
    # Number of sph particles
    nsph = len(mp)

    # Obtain extra quantities
    r_f = np.linalg.norm(pos,axis=1)
    r_xy = np.linalg.norm(pos[:,:2],axis=1)

    # tangential velocity (rotation)
    vtanx = Omega*(-pos[:,1])
    vtany = Omega*(pos[:,0])
    vtanz = 0  # Rotation axis along z

    # -------- Define SN energy -------- #
    V_rad = np.zeros((len(mp),3))  # radial velocity 3D (Piston)
    phi = np.arccos(pos[:,2]/r_f)  # angle between r and z axis
    u_SN = np.zeros(len(mp))       # Internal energy injected (Thermal)

    if SNType == 'Piston':
        vrad = np.sqrt(2*SNe_pm)  # radial velocity magnitude
        R_exp = R_int(M_exp + mBH) # obtain radius containing M_exp
        p_exp = r_f <= R_exp  # find particles within R_exp

        print 'Mass of particles within R_exp', sum(mp[p_exp])/c.msun, ' solar masses'
        V_rad[:,0][p_exp] = vrad*(pos[:,0][p_exp]/r_xy[p_exp])*np.sin(phi[p_exp])
        V_rad[:,1][p_exp] = vrad*(pos[:,1][p_exp]/r_xy[p_exp])*np.sin(phi[p_exp])
        V_rad[:,2][p_exp] = vrad*np.cos(phi[p_exp])


    if SNType =='Thermal':
        R_exp = R_int(M_exp + mBH) # obtain radius containing M_exp
        p_exp = r_f <= R_exp  # find particles within R_exp
        u_SN[p_exp] += SNe_pm  # internal energy added to M_exp


    #------- Get final quatities --------#
    x_f = pos[:,0] + np.ones(nsph)*prel[0]
    y_f = pos[:,1] + np.ones(nsph)*prel[1]
    z_f = pos[:,2] + np.ones(nsph)*prel[2]

    vx_f = np.zeros(nsph) + np.ones(nsph)*vrel[0] + vtanx + V_rad[:,0]
    vy_f = np.zeros(nsph) + np.ones(nsph)*vrel[1] + vtany + V_rad[:,1]
    vz_f = np.zeros(nsph) + np.ones(nsph)*vrel[2] + vtanz + V_rad[:,2]

    u_f = u_int(r_f) + u_SN         # internal energy
    print("Printing u_f:")
    print(u_f)
    m_f = mp                        # particle's mass

    N_ngb = 50
    # eta = 2.0*(N_ngb*(3./(4*np.pi)))**(1./3.)
    eta=1.4
    rho_f = rho_int(r_f)            # density (from profile)

    h_f = eta*(m_f/rho_f)**(1./3.)  # smoothing length
    id_f = np.arange(1,nsph+1)      # particle id
    ptype = [0]*len(h_f)            # particle type

    return ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f


def remove_gasparticles(xpos,ypos,zpos,mp,R_core):
    """
    Removes SPH particles in the core and returns the properties of a
    point mass that accounts for the mass removed.

    Note: The outermost shell from the core remains, so it is fixed as
    a boundary between the point mass and the outer shells.

    Receives:
    * (xpos, ypos, zpos)  -> Array with (x,y,z) SPH particles' positions
    * mp -> array with particle's mass
    * R_core -> Core radius and maximus radius for the core shell.

    Returns:
    * xpos,ypos,zpos -> arrays with all SPH particle positions
    * R_core_shell -> radius of the innermost shell
    * N_core_shell -> number of particles in the boundary shell
    * M_core -> mass inside the boundary shell.
    """

    # Get particle's radius
    rp = np.sqrt(xpos**2 + ypos**2 + zpos**2)

    # Get SPH particles outside the core
    p_env = rp > R_core

    #Get SPH particles inside the core
    p_core  = rp < R_core

    # Get core's outermost shell radius
    r_bdry = max(rp[p_core])

    # Get particles at that radius (+ - 1% in case of round off errors)
    p_bdry = np.logical_and(p_core,rp >= 0.98*r_bdry)

    # Particles to be removed
    p_coref = rp < 0.99*r_bdry

    # Remaining particles
    p_envf = rp > 0.99*r_bdry

    #Number of particles check

    print 'Radius of innermost remaining particles, ',round(min(rp[p_env])/c.rsun,3),'solar'

    Ni = len(xpos)
    N_core = len(xpos[p_coref])
    N_env = len(xpos[p_envf])
    N_bdry = len(xpos[p_core]) - len(xpos[p_coref])

    print '--------------------------------------'
    print 'NSPH_initial  |    N_env    |   Ncore '
    print Ni,'           ',N_env ,'     ',N_core
    print ''

    # mass of particles removed
    Mc = sum(mp[p_coref])
    Menv = sum(mp[p_envf])

    print '--------------------------------------'
    print 'Mass_initial  |    Menv    |  Mcore (point mass) '
    print round(sum(mp)/c.msun,4),'           ',round((Menv)/c.msun,4),'         ',round(Mc/c.msun,4)
    print ''

#     print len(xpos[p_coref]), len(xpos[p_core]),\
#     len(xpos[p_core]) - len(xpos[p_coref]),Mc/Mo, sum(mp[p_core])/Mo, (sum(mp[p_core])-Mc)/Mo

    return Mc, N_bdry, xpos[p_envf], ypos[p_envf], zpos[p_envf], mp[p_envf]

def add_Particle(partype,posBH,velBH,mBH,ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f):
    '''
    Appends the properties of a sink particle (BH) to the arrays containing all
    SPH particles' properties

    Receives:
    posBH -> list (or array) with (x,y,z) BH's position
    velBH -> list (or array) with (vx,vy,vz) BH's velocity
    mBH -> BH mass
    ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f -> arrays containing all SPH particles' properties.


    Returns:
    ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f -> array containing all particle properties.


    Note: The BH is given u = h = rho = 0 in order to have same size arrays.
    These values are not used when creating the hdf5 file.
    '''
    x_f = np.append(x_f,posBH[0])
    y_f = np.append(y_f,posBH[1])
    z_f = np.append(z_f,posBH[2])

    vx_f = np.append(vx_f,velBH[0])
    vy_f = np.append(vy_f,velBH[1])
    vz_f = np.append(vz_f,velBH[2])

    m_f = np.append(m_f,mBH)
    ptype.append(partype)
    id_f = np.append(id_f,len(id_f)+1)
    u_f = np.append(u_f,0)
    h_f = np.append(h_f,0)
    rho_f = np.append(rho_f,0)


    return ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f



c = Constants()
import h5py as h5py

def make_hdf5_from_table(data,filename_out,
                      scale_to_units=False,DistUnit=c.rsun,
                      MassUnit=c.msun):
    """
    Creates an hdf5 file suitable for use in Gadget (or GIZMO) from a table with all particle properties.

    Input data is assumed to have units in CGS and can be scaled to other desired units.

    Receives:
    data -> Table containing all particle properties:
            (ptype,id_f,m_f,x_f,y_f,z_f,vx_f,vy_f,vz_f,u_f,h_f,rho_f)

    filename_out -> String used for naming the filename (should include the extension .hdf5)

    scale_to_units -> Boolean used to choose if data will be scaled to a particular unit` (False by default)

    DistUnit -> Length unit to be used (c.rsun by default)

    MassUnit -> Mass unit to be used (c.msun by default)

    Based on a modification by Morgan MacLeod from a file by Phil Hopkins (GIZMO stuff)
    """

    print ("--------------------------------")
    print ("Saving initial conditions in:")
    print (filename_out)
    print ("--------------------------------")



    # CHECK IF WE'RE Doing a units conversion cgs->solar
    if scale_to_units:
        print ("Converting from CGS to other Units...")
        mscale = MassUnit
        lscale = DistUnit
        tscale = np.sqrt(lscale**3 / (c.G * mscale))
        vscale = lscale/tscale
        escale = mscale * vscale**2

        # mass
        data['m'] = data['m']/mscale

        # position
        data['x'] = data['x']/lscale
        data['y'] = data['y']/lscale
        data['z'] = data['z']/lscale

        # velocity
        data['vx'] = data['vx']/vscale
        data['vy'] = data['vy']/vscale
        data['vz'] = data['vz']/vscale

        # energies
        data['u'] = data['u']/(escale/mscale)

        # densities
        data['rho'] = data['rho']/(mscale/lscale**3)

        # smoothing lengths
        data['hsml'] = data['hsml']/lscale

        print ("--------------------------------")
        print ("Units conversion complete:")
        print ("  Mass Scale   = ",mscale)
        print ("  Length Scale = ",lscale)
        print ("  Time Scale   = ",tscale)
        print ("  Vel Scale    = ",vscale)
        print ("  Energy Scale = ",escale)
        print ("  Density Scale = ",mscale/lscale**3)
        print  (" ... in cgs units")
        print ("--------------------------------")
    else:
        print ("No units conversion requested...")

    # now we get ready to actually write this out
    #  first - open the hdf5 ics file, with the desired filename
    file = h5py.File(filename_out,'w')
    print ("HDF5 file created ... ")

    # set particle number of each type into the 'npart' vector
    #  NOTE: this MUST MATCH the actual particle numbers assigned to each type, i.e.
    #   npart = np.array([number_of_PartType0_particles,number_of_PartType1_particles,number_of_PartType2_particles,
    #                     number_of_PartType3_particles,number_of_PartType4_particles,number_of_PartType5_particles])
    #   or else the code simply cannot read the IC file correctly!
    #

    # MM: Count number of different particle types and
    # Fill in an array, npart
    data0 = data[data['type']==0].copy()
    data1 = data[data['type']==1].copy()
    data2 = data[data['type']==2].copy()
    data3 = data[data['type']==3].copy()
    data4 = data[data['type']==4].copy()
    data5 = data[data['type']==5].copy()

    N0 = len(data0)
    N1 = len(data1)
    N2 = len(data2)
    N3 = len(data3)
    N4 = len(data4)
    N5 = len(data5)

    npart = np.array([N0,N1,N2,N3,N4,N5])
    print ("Particle numbers [N0,N1,N2,N3,N4,N5]: ",npart)


    # now we make the Header - the formatting here is peculiar, for historical (GADGET-compatibility) reasons
    h = file.create_group("Header");

    # here we set all the basic numbers that go into the header
    # (most of these will be written over anyways if it's an IC file; the only thing we actually *need* to be 'correct' is "npart")
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
                                         #  if we make a multi-part IC file. with this simple script, we aren't equipped to do that.
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
                                        # we set masses explicitly by-particle this should be zero. that is more flexible anyways, as it
                                        # allows for physics which can change particle masses
    ## all of the parameters below will be overwritten by whatever is set in the run-time parameterfile if
    ##   this file is read in as an IC file, so their values are irrelevant. they are only important if you treat this as a snapshot
    ##   for restarting. Which you shouldn't - it requires many more fields be set.
    ##   But we still need to set some values for the code to read

    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 1.0; # box size
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
    h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
    h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
    h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
    h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
    h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
    h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
    h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs
    ## ok, that ends the block of 'useless' parameters


    # Now, the actual data!
    #   These blocks should all be written in the order of their particle type (0,1,2,3,4,5)
    #   If there are no particles of a given type, nothing is needed (no block at all)
    #   PartType0 is 'special' as gas. All other PartTypes take the same, more limited set of information in their ICs
    print ("Initializing particles of type 0...")

    # start with particle type zero. first (assuming we have any gas particles) create the group
    p = file.create_group("PartType0")

    # POSITIONS

    # now combine the xyz positions into a matrix with the correct format
    q=np.zeros((N0,3))
    q[:,0]=np.array(data0['x'])
    q[:,1]=np.array(data0['y'])
    q[:,2]=np.array(data0['z'])

    # write it to the 'Coordinates' blocke
    p.create_dataset("Coordinates",data=q)

    # VELOCITIES

    # similarly, combine the xyz velocities into a matrix with the correct format
    q=np.zeros((N0,3))
    q[:,0]=np.array(data0['vx'])
    q[:,1]=np.array(data0['vy'])
    q[:,2]=np.array(data0['vz'])
    # write it to the 'Velocities' block
    p.create_dataset("Velocities",data=q)

    # write particle ids to the ParticleIDs block
    p.create_dataset("ParticleIDs",data=np.array(data0['id']) )

    # write particle masses to the Masses block
    p.create_dataset("Masses",data=np.array(data0['m']) )

    # write internal energies to the InternalEnergy block
    p.create_dataset("InternalEnergy",data=np.array(data0['u']) )

    # write densities to the Density block
    p.create_dataset("Density",data=np.array(data0['rho']) )

    # write smoothing lengths to the SmoothingLength block
    p.create_dataset("SmoothingLength",data=np.array(data0['hsml']) )

    # combine the xyz magnetic fields into a matrix with the correct format
    # MM: note these are currently floored to zero.
#     q=np.zeros((N0,3));
#     q[:,0]=np.zeros(N0)
#     q[:,1]=np.zeros(N0)
#     q[:,2]=np.zeros(N0)
    # write magnetic fields to the MagneticField block. note that this is unnecessary if the code is compiled with
    #   MAGNETIC off. however, it is not a problem to have the field there, even if MAGNETIC is off, so you can
    #   always include it with some dummy values and then use the IC for either case
#     p.create_dataset("MagneticField",data=q)

    # PartType1
    if N1 > 0:
        print ("Initializing particles of type 1...")
        p1 = file.create_group("PartType1")
        # POSITIONS
        # now combine the xyz positions into a matrix with the correct format
        q=np.zeros((N1,3))
        q[:,0]=np.array(data1['x'])
        q[:,1]=np.array(data1['y'])
        q[:,2]=np.array(data1['z'])
        # write it to the 'Coordinates' block
        p1.create_dataset("Coordinates",data=q)
        # VELOCITIES
        # similarly, combine the xyz velocities into a matrix with the correct format
        q=np.zeros((N1,3))
        q[:,0]=np.array(data1['vx'])
        q[:,1]=np.array(data1['vy'])
        q[:,2]=np.array(data1['vz'])
        # write it to the 'Velocities' block
        p1.create_dataset("Velocities",data=q)

        p1.create_dataset("ParticleIDs",data=np.array(data1['id']))
        p1.create_dataset("Masses",data=np.array(data1['m']))


    # PartType2
    if N2 > 0:
        print ("Initializing particles of type 2...")
        p2 = file.create_group("PartType2")
        # POSITIONS
        # now combine the xyz positions into a matrix with the correct format
        q=np.zeros((N2,3))
        q[:,0]=np.array(data2['x'])
        q[:,1]=np.array(data2['y'])
        q[:,2]=np.array(data2['z'])
        # write it to the 'Coordinates' block
        p2.create_dataset("Coordinates",data=q)
        # VELOCITIES
        # similarly, combine the xyz velocities into a matrix with the correct format
        q=np.zeros((N2,3))
        q[:,0]=np.array(data2['vx'])
        q[:,1]=np.array(data2['vy'])
        q[:,2]=np.array(data2['vz'])
        # write it to the 'Velocities' block
        p2.create_dataset("Velocities",data=q)

        p2.create_dataset("ParticleIDs",data=np.array(data2['id']))
        p2.create_dataset("Masses",data=np.array(data2['m']))

    # PartType3
    if N3 > 0:
        print ("Initializing particles of type 3...")
        p3 = file.create_group("PartType3")
        # POSITIONS
        # now combine the xyz positions into a matrix with the correct format
        q=np.zeros((N3,3))
        q[:,0]=np.array(data3['x'])
        q[:,1]=np.array(data3['y'])
        q[:,2]=np.array(data3['z'])
        # write it to the 'Coordinates' block
        p3.create_dataset("Coordinates",data=q)
        # VELOCITIES
        # similarly, combine the xyz velocities into a matrix with the correct format
        q=np.zeros((N3,3))
        q[:,0]=np.array(data3['vx'])
        q[:,1]=np.array(data3['vy'])
        q[:,2]=np.array(data3['vz'])
        # write it to the 'Velocities' block
        p3.create_dataset("Velocities",data=q)

        p3.create_dataset("ParticleIDs",data=np.array(data3['id']))
        p3.create_dataset("Masses",data=np.array(data3['m']))


    # PartType4
    if N4 > 0:
        print ("Initializing particles of type 4...")
        p4 = file.create_group("PartType4")
        # POSITIONS
        # now combine the xyz positions into a matrix with the correct format
        q=np.zeros((N4,3))
        q[:,0]=np.array(data4['x'])
        q[:,1]=np.array(data4['y'])
        q[:,2]=np.array(data4['z'])
        # write it to the 'Coordinates' block
        p4.create_dataset("Coordinates",data=q)
        # VELOCITIES
        # similarly, combine the xyz velocities into a matrix with the correct format
        q=np.zeros((N4,3))
        q[:,0]=np.array(data4['vx'])
        q[:,1]=np.array(data4['vy'])
        q[:,2]=np.array(data4['vz'])
        # write it to the 'Velocities' block
        p4.create_dataset("Velocities",data=q)

        p4.create_dataset("ParticleIDs",data=np.array(data4['id']))
        p4.create_dataset("Masses",data=np.array(data4['m']))


    # PartType5
    if N5 > 0:
        print ("Initializing particles of type 5...")
        p5 = file.create_group("PartType5")
        # POSITIONS
        # now combine the xyz positions into a matrix with the correct format
        q=np.zeros((N5,3))
        q[:,0]=np.array(data5['x'])
        q[:,1]=np.array(data5['y'])
        q[:,2]=np.array(data5['z'])
        # write it to the 'Coordinates' block
        p5.create_dataset("Coordinates",data=q)

        # VELOCITIES
        # similarly, combine the xyz velocities into a matrix with the correct format
        q=np.zeros((N5,3))
        q[:,0]=np.array(data5['vx'])
        q[:,1]=np.array(data5['vy'])
        q[:,2]=np.array(data5['vz'])

        # write it to the 'Velocities' block
        p5.create_dataset("Velocities",data=q)
        p5.create_dataset("ParticleIDs",data=np.array(data5['id']))
        p5.create_dataset("Masses",data=np.array(data5['m']))



    # close the HDF5 file, which saves these outputs
    file.close()
    print ("... all done!")


def readfile(filename,Profiletype = 'Heger',Rotating=True):
    """Function used to read stellar profiles from MESA or Heger models (Kepler).

    Receives:
    filename -> string with exact filename
    Profiletype -> 'Heger' by default indicates the star was produced by Alex Heger's code (WH 2006). Accepts Profiletypes: MESA, ChrisIC, ChrisSN, Heger
                MESA -> False by default, to indicate it is a MESA profile
                Heger -> True by default, to indicate the
                ChrisSN -> False by default, to indicate if this is one of Chris' SN profiles
                ChrisIC -> False by default, to indicate if this is one of Chris' IC profiles (similar to Heger's)
    Rotating -> True by default, to indicate that the star contins information of Omega

    Returns:
    m -> array with mass profile
    r -> array with radius
    v -> array with radial velocity
    rho -> array with density profile
    Omega -> array with angular velocity (filled with zeros if Rotating=False)
    jprofile -> array with specific angular momentum profile
    T -> array with Temperature profile
    p -> array with pressure profile
    e -> array with specific internal energy profile


    """

    if Profiletype == 'MESA':

        data = ascii.read(filename,header_start=4,data_start=5)

        # print data.colnames
        m = data['mass'][::-1]*c.msun # cell outer total mass
        r = c.rsun*(data['radius'][::-1]) # cell outer radius
        v = data['mass'][::-1]*0 # cell outer velocity
        rho = 10**(data['logRho'][::-1]) # cell density
        if Rotating == True:
            Omega = data['omega'][::-1] #5*s26_data[:,9] # cell specific angular momentum
            jprofile = data['j_rot'][::-1]
        else:
            Omega = np.zeros(len(m))
            jprofile = np.ones(len(m))

        T = data['temperature'][::-1] # cell temperature
        p = data['pressure'][::-1] # cell pressure
        e = data['energy'][::-1] # cell specific internal energy
        S = data['entropy'][::-1] # cell specific entropy
        
    if Profiletype == 'Heger':

        data = np.genfromtxt(filename)
        m = data[:,1] # cell outer total mass
        r = data[:,2] # cell outer radius
        v = data[:,3] # cell outer velocity
        rho = data[:,4] # cell density
        Omega = data[:,9] #5*s26_data[:,9] # cell specific angular momentum
        jprofile = (2./3.)*Omega*r**2

        T = data[:,5] # cell temperature
        p = data[:,6] # cell pressure
        e = data[:,7] # cell specific internal energy
        S = data[:,8] # cell specific entropy

    if Profiletype == 'ChrisSN':
        data = np.genfromtxt(filename)
        m = data[:,0]*c.msun  # cell outer total mass
        r = data[:,1] # cell outer radius
        v = data[:,5] # cell outer velocity
        rho = data[:,2] # cell density
        Omega = np.ones(len(m)) # This star is not rotating
        jprofile =  np.ones(len(m)) # array with ones

        T = data[:,6] # cell temperature
        p = data[:,4] # cell pressure
        e = data[:,3] # cell specific energy
        # S = data[:,8] # cell specific entropy

    if Profiletype == 'ChrisIC':
        data = np.genfromtxt(filename)
        m = data[:,2]  # cell outer total mass
        r = data[:,3] # cell outer radius
        v = data[:,4] # cell outer velocity
        rho = data[:,5] # cell density
        Omega = data[:,10] # This star is not rotating
        jprofile =  np.copy(Omega) # array with ones

        T = data[:,6] # cell temperature
        p = data[:,7] # cell pressure
        e = data[:,8] # cell specific energy
        S = data[:,9] # cell specific entropy

    return m, r ,v ,rho, Omega,jprofile, T, p, e


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
