import numpy as np

# ====================================================#
# Define physical constants

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

c = Constants()

class ICs():
    def __init__(self,m1,m2,m3):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        print ("Initial masses defined...")
        return None



    global Ro, Mo, G, AU, day, hrs
    Ro=6.96e10
    Mo=1.99e33
    G=6.6726e-08
    AU = 1.49597870e+13
    day = 60*60*24
    hrs = 60*60


    def get_IC(self,m1,m2,a,e,m3,r,rperi,e3):

        print '  Masses (solar units)'
        print '    m1     |     m2     |     m3    '
        print '------------------------------------'
        print '  ',round(m1/Mo,3),'      ',round(m2/Mo,3),'       ',round(m3/Mo,3)

        # a = 0.01*AU
        # e = 0.3
        # Get Binary orbit
        global x1,y1,vx1,vy1,x2,y2,vx2,vy2,P

        x1,y1,vx1,vy1,x2,y2,vx2,vy2,P = self.getBinary(m1,m2,a,e)

        # global m1,m2


        print '\nBinary properties (m1, m2)'
        print'----------------------'
        print '\nOrbital separation:',a/AU, ' AU (', a/Ro, 'solar radius)'
        print 'Eccentricity:', e


        print 'Orbital period:',round(P/day,4), ' days (',round(P/hrs,4), ' hours)'
        print 'v1 =', round(np.sqrt(vx1**2 + vy1**2)/1e5,4), ', v2 =', round(np.sqrt(vx2**2 + vy2**2)/1e5),4, ' km/s'


        # 3rd body orbit

        # r = 500 * Ro # distance to binary's CM
        # rperi = 100 * Ro # desired perihelium distance
        # e3 = 1 # eccentricity
        Mbin = m1+m2 # binary's mass
        Mstar = m3 # star's mass

        v, k, f, rdot, fdot = self.Orbit_3rd(e3,r,rperi,Mbin,Mstar)

        # global m3

        print '\n3rd body orbital properties (m3)'
        print'----------------------'
        print '\nInitial distance to binary:', r/Ro, ' solar'
        print 'Initial velocity magnitude |v|:', v/(1.0e5), 'km/s'
        print 'Orbital angular momentum k:', k, ' cm^2 / s'
        print 'Initial true anomaly f:', f, ' radians'
        print 'Initial radial velocity (magnitude) dr/dt:', rdot/1e5, 'km/s'
        print 'Initial tangential velocity (magnitude) r * df/dt:', r* fdot/1e5, 'km/s'

        x3,y3,vx3,vy3 = self.xyCoord(r,f,-rdot,-fdot)


        # Put all initial conditions to be evolved into an array
        initialvalues = np.array([x1,y1,vx1,vy1,x2,y2,vx2,vy2,x3,y3,vx3,vy3])

        return initialvalues,P



    def threebody_derivs(self,rv):

        # Positions and velocities for m1
        x1 = rv[0]
        y1 = rv[1]
        vx1 = rv[2]
        vy1 = rv[3]
        # Positions and velocities for m2
        x2 = rv[4]
        y2 = rv[5]
        vx2 = rv[6]
        vy2 = rv[7]
        # Positions and velocities for m3
        x3 = rv[8]
        y3 = rv[9]
        vx3 = rv[10]
        vy3 = rv[11]


        #get quantities from positions and velocities
        r1_2 = np.sqrt((x1-x2)**2 + (y1-y2)**2)  # distance 1 , 2
        r1_3 = np.sqrt((x1-x3)**2 + (y1-y3)**2)  # distance 1 , 3
        r2_3 = np.sqrt((x2-x3)**2 + (y2-y3)**2)  # distance 3 , 2


        # Derivatives for the Earth (interacts with the red dwarf and the sun)
        dxdt_1 = vx1
        dydt_1 = vy1

        dvxdt_1 = - (G*self.m2 / r1_2**3) * (x1-x2) - (G*self.m3 / r1_3**3) * (x1-x3)
        dvydt_1 = - (G*self.m2 / r1_2**3) * (y1-y2) - (G*self.m3 / r1_3**3) * (y1-y3)

        # Derivatives for the Sun (only interacts with the red dwarf)
        dxdt_2 = vx2
        dydt_2 = vy2

        dvxdt_2 = - (G*self.m1 / r1_2**3) * (x2-x1) - (G*self.m3 / r2_3**3) * (x2-x3)
        dvydt_2 = - (G*self.m1 / r1_2**3) * (y2-y1) - (G*self.m3 / r2_3**3) * (y2-y3)

        # Derivatives for the Sun (only interacts with the sun)
        dxdt_3 = vx3
        dydt_3 = vy3

        dvxdt_3 = - (G*self.m1 / r1_3**3) * (x3-x1) - (G*self.m2 / r2_3**3) * (x3-x2)
        dvydt_3 = - (G*self.m1 / r1_3**3) * (y3-y1) - (G*self.m2 / r2_3**3) * (y3-y2)

        # pack the derivatives up, and ship them out
        derivarray = np.array([dxdt_1, dydt_1, dvxdt_1, dvydt_1, dxdt_2, dydt_2, dvxdt_2, dvydt_2, dxdt_3, dydt_3, dvxdt_3, dvydt_3])

        return derivarray

    def threebody_derivs_odeint(self,rv,times):

        # Positions and velocities for m1
        x1 = rv[0]
        y1 = rv[1]
        vx1 = rv[2]
        vy1 = rv[3]
        # Positions and velocities for m2
        x2 = rv[4]
        y2 = rv[5]
        vx2 = rv[6]
        vy2 = rv[7]
        # Positions and velocities for m3
        x3 = rv[8]
        y3 = rv[9]
        vx3 = rv[10]
        vy3 = rv[11]


        #get quantities from positions and velocities
        r1_2 = np.sqrt((x1-x2)**2 + (y1-y2)**2)  # distance 1 , 2
        r1_3 = np.sqrt((x1-x3)**2 + (y1-y3)**2)  # distance 1 , 3
        r2_3 = np.sqrt((x2-x3)**2 + (y2-y3)**2)  # distance 3 , 2


        # Derivatives for the Earth (interacts with the red dwarf and the sun)
        dxdt_1 = vx1
        dydt_1 = vy1

        dvxdt_1 = - (G*self.m2 / r1_2**3) * (x1-x2) - (G*self.m3 / r1_3**3) * (x1-x3)
        dvydt_1 = - (G*self.m2 / r1_2**3) * (y1-y2) - (G*self.m3 / r1_3**3) * (y1-y3)

        # Derivatives for the Sun (only interacts with the red dwarf)
        dxdt_2 = vx2
        dydt_2 = vy2

        dvxdt_2 = - (G*self.m1 / r1_2**3) * (x2-x1) - (G*self.m3 / r2_3**3) * (x2-x3)
        dvydt_2 = - (G*self.m1 / r1_2**3) * (y2-y1) - (G*self.m3 / r2_3**3) * (y2-y3)

        # Derivatives for the Sun (only interacts with the sun)
        dxdt_3 = vx3
        dydt_3 = vy3

        dvxdt_3 = - (G*self.m1 / r1_3**3) * (x3-x1) - (G*self.m2 / r2_3**3) * (x3-x2)
        dvydt_3 = - (G*self.m1 / r1_3**3) * (y3-y1) - (G*self.m2 / r2_3**3) * (y3-y2)

        # pack the derivatives up, and ship them out
        derivarray = np.array([dxdt_1, dydt_1, dvxdt_1, dvydt_1, dxdt_2, dydt_2, dvxdt_2, dvydt_2, dxdt_3, dydt_3, dvxdt_3, dvydt_3])

        return derivarray


    def getBinary(self,m1,m2,a,e):

        mt = m1 + m2
        mu = m1*m2/mt
        SemimayorAx = a/(1.-e)
        P = np.sqrt((2*np.pi)**2 * SemimayorAx**3 / (G*mt))
        Jspec = mu*np.sqrt(G*mt*SemimayorAx*(1-e**2))

        # Primary's distance to perihelium f primary
        a1 = (1 - e)*SemimayorAx/(1+m1/m2)

        # positions
        x1 = a1
        y1 = 0.

        x2 = -(m1/m2)*x1
        y2 = -(m1/m2)*y1

        #velocities
        vx1 = 0.0
        vy1 = Jspec/(a1*m1*(1+ m1/m2))

        vx2 = -(m1/m2)*vx1
        vy2 = -(m1/m2)*vy1

        return np.array([x1,y1,vx1,vy1,x2,y2,vx2,vy2,P])

    def Orbit_3rd(self,e,r,rperi,Mbin,Mstar):
        mu = G * (Mbin + Mstar)
        k = (mu * rperi * (1 + e))**(1./2.)
        f = np.arccos((k**2 / (mu * r) - 1)/e)
        v = ((mu**2 * (e**2 - 1))/ k**2 + 2*mu/r)**(1./2.)

        fdot = k / r**2
        rdot = np.sqrt(v**2 - (fdot*r)**2)

        return v, k, f, rdot, fdot

    def xyCoord(self,r,f,rdot,fdot):
        x = r*np.cos(f)
        y = r*np.sin(f)

        vx = rdot*np.cos(f) - r*fdot*np.sin(f)
        vy = rdot*np.sin(f) + r*fdot*np.cos(f)

        return np.array([x,y,vx,vy])


    def RK4(self,times,pos_vel_array,derivs):

        dt = times[1]- times[0]

        # Create array to storeresults
        result = np.zeros( (len(times),len(pos_vel_array)) )

        for i, t in enumerate(times):

            result[i] = pos_vel_array

            # then advance the solution using Runge Kutta method
            k1 = derivs(pos_vel_array)

            k2 = dt*derivs(pos_vel_array + k1/2.0)

            k3 = dt*derivs(pos_vel_array + k2/2.0)

            k4 = dt*derivs(pos_vel_array + k3)



            pos_vel_array += k1/6. + k2/3. + k3/3. + k4/6.

        return result
