
import numpy as np
import math
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Concorde Data
rho0 = 1.225
g = 9.81

m = 120000.0    #kg
Sref = 360.0    #m²
Lref = 27.5     #m
Iyy = 14.5e6    #kg.m²
xcg = 0.50      #en%deLref

Cza = 2.3
Czq = 0
Czdm = 0.84
alpha0 = 0

Cm0 = 0.0
Cma25 = -0.71
Cmq = -0.12
Cmdm = -0.25

Cma = Cma25 + (xcg - 0.25)*Cza

Cx0 = 1.2/100
ki = 0.33

# Model de Poussee :     F = F0*((rho/rho0)**Mf)*((V0/V)**Lf)*dx
F0 = 690.4e3
V0 = 1
Lf = 0  
Mf = 1

#functions
def frho(alt):
    if alt < 0:
        alt = 0
    if alt < 11000:
        T = 288.15 - 0.0065*alt 
        rho = rho0*(1 - 2.2558e-5*alt)**4.26
        rho_h = -(g/287.04 - 0.0065)/T
    else: 
        T = 216.65
        rho = 0.36*np.exp(-0.158*(alt - 11000)/1000)
        rho_h = -(g/287.04)/T
    return rho, rho_h

def FM(state):
    [V, gamma, alpha, q, alt, dx, dm] = state
    Cz = Cza*(alpha - alpha0) + Czdm*dm
    Cm = Cm0 + Cma*(alpha - alpha0) + Cmq*q*Lref/V + Cmdm*dm
    Cx = Cx0 + ki*Cz**2
    rho, rho_h = frho(alt)
    Pdyn = 0.5*rho*V*V
    X = Pdyn*Sref*Cx
    Z = Pdyn*Sref*Cz
    M = Pdyn*Sref*Lref*Cm
    F = F0*((rho/rho0)**Mf)*((V0/V)**Lf)*dx
    return [F, X, Z, M]

def deriv_longi(state):
    [V, gamma, alpha, q, alt, dx, dm] = state
    [F, X, Z, M] = FM(state)
    dv = (F - X)/m - g*np.sin(gamma)
    dg = (Z - m*g)/(m*V)
    da = q - dg
    dq = M/Iyy
    dh = V*np.sin(gamma)
    return [dv, dg, da, dq, dh]

def trim_longi(V, alt):
    def Eq(X):
        [gamma, alpha, q, dx, dm] = X
        state = [V, gamma, alpha, q, alt, dx, dm]
        return deriv_longi(state)
    Xinit = [0,0,0,0,0]
    return fsolve(Eq, Xinit)

def sys_longi(V, alt):
    [gam, alpha, q, dx, dm] = trim_longi(V, alt)
    state = [V, gam, alpha, q, alt] + [dx, dm]
    dEq = []
    for i in range(7):
        h = 0.01
        st = state.copy()
        st[i] = state[i] + h
        dEqp = deriv_longi(st)
        st = state.copy()
        st[i] = state[i] - h
        dEqm = deriv_longi(st)
        dEq.append([(dEqp[j]-dEqm[j])/(2*h) for j in range(5)])
        sys = np.array(dEq).T
        Asys = sys[:,0:5]
        Bsys = sys[:,5:7]
    return [Asys, Bsys]

def lin_longi(V, alt):
    [gam, alpha, q, dx, dm] = trim_longi(V, alt)
    Cz = Cza*(alpha - alpha0) + Czdm*dm
    Cx = Cx0 + ki*Cz**2
    finesse = Cz/Cx
    rho, rho_h = frho(alt)
    
    A = np.zeros((5,5))
    A[0,0] = (Lf-2)*g/(V*finesse)
    A[0,1] = -g
    A[0,2] = -2*g*ki*Cza
    A[0,3] = -2*g*ki*Lref*Czq/V
    A[0,4] = (Mf-1)*g*rho_h/finesse
    
    A[1,0] = 2*g/(V**2)
    A[1,1] = 0
    A[1,2] = rho*V*Sref*Cza/(2*m)
    A[1,3] = rho*Sref*Lref*Czq/(2*m)
    A[1,4] = rho_h*g/V
    
    A[2,0] = -A[1,0]
    A[2,1] = 0
    A[2,2] = -A[1,2]
    A[2,3] = 1 - A[1,3]
    A[2,4] = -A[1,4]
    
    A[3,0] = 0
    A[3,1] = 0
    A[3,2] = rho*(V**2)*Sref*Lref*Cma/(2*Iyy)
    A[3,3] = rho*V*Sref*(Lref**2)*Cmq/(2*Iyy)
    A[3,4] = 0

    A[4,0] = 0
    A[4,1] = V
    A[4,2] = 0
    A[4,3] = 0
    A[4,4] = 0

    B = np.zeros((5,2))
    B[0,0] = F0*((rho/rho0)**Mf)*((V0/V)**Lf)/m
    B[0,1] = -2*g*ki*Czdm

    B[1,0] = 0
    B[1,1] = rho*V*Sref*Czdm/(2*m)
    
    B[2,0] = 0
    B[2,1] = -B[1,1]
    
    B[3,0] = 0
    B[3,1] = rho*(V**2)*Sref*Lref*Cmdm/(2*Iyy)

    B[4,0] = 0
    B[4,1] = 0
    
    return [A, B]

def sim_longi(V, h, ddu, KBF, Tf):
    trim_sim = trim_longi(V, h)
    
    g0 = trim_sim[0]
    a0 = trim_sim[1]
    q0 = trim_sim[2]
    dx0 = trim_sim[3]
    dm0 = trim_sim[4]
    
    def sd(t,y):
        dx = dx0 + ddu[0] + np.dot(Kdx,y)
        dm = dm0 + ddu[1] + np.dot(Kdm,y)
        state = list(y) + [dx, dm]
        return deriv_longi(state)
    
    y0 = [V, g0, a0, q0, h]
    return solve_ivp(sd, [0, Tf], y0)

def sim_syslin(A, B, du, K, Tf):
    def sd(t,x):
        dx = np.dot(A,x) + np.dot(B, du + np.dot(K,x))
        return dx
    x0 = [0,0,0,0,0]
    return solve_ivp(sd, [0, Tf], x0)

# Trim de l'avion
alt = 1524.0
V = 100.0

trim1 = trim_longi(V, alt)

rho, rho_h = frho(alt)

print('alpha = ', trim1[1]*57.3, 'deg')
print('Thrust =', trim1[3]*F0*rho/rho0/1000, 'kN')
print('dx =', trim1[3]*100, '%')
print('dm =', trim1[4]*57.3, 'deg')


#Simulation longi 
ddx = 0
ddm = -2/57.3 #rad
du = [ddx,ddm]

Kdx = np.zeros(5)
Kdm = np.zeros(5)

# retour en q sur la profondeur
Kdm[3] = 1
KBF = [Kdx, Kdm]

Tf = 20 #secondes

#simulation non linéaire
sim = sim_longi(V, alt, du, KBF, Tf)

# Affichage simulation
fig, axs = plt.subplots(5, 1)
axs[0].plot(sim.t, sim.y[0])
axs[0].set_ylabel('spd')
axs[0].grid(True)

axs[1].plot(sim.t, sim.y[1])
axs[1].set_ylabel('gamma')
axs[1].grid(True)

axs[2].plot(sim.t, sim.y[2])
axs[2].set_ylabel('alpha')
axs[2].grid(True)

axs[3].plot(sim.t, sim.y[3])
axs[3].set_ylabel('q')
axs[3].grid(True)

axs[4].plot(sim.t, sim.y[4])
axs[4].set_ylabel('alt')
axs[4].grid(True)
axs[4].set_xlabel('time')

plt.show()


# Linéarisation et modes

def print_modes_longi(Asys):
    A = np.linalg.eig(Asys)[0]
    print(A)
    
    w0 = math.sqrt(abs(A[0]*A[1]))
    amort = abs(A[0]+A[1])/2
    amort1 = amort/w0
    wp = w0*math.sqrt(1-amort1**2)
    T = 2*math.pi/wp
    print('SPO', 'period (s) =', T, 'amort =', amort1)
    
    w0 = math.sqrt(abs(A[2]*A[3]))
    amort = abs(A[2]+A[3])/2
    amort1 = amort/w0
    wp = w0*math.sqrt(1-amort1**2)
    T = 2*math.pi/wp
    print('PHU', 'period (s) =', T, 'amort =', amort1)
    
    T3 = 1/abs(A[4])
    print('RAP', 'period (s) =', T3)

    return []

[Asys, Bsys] = sys_longi(V, alt)
print_modes_longi(Asys)

AsysBF = Asys + np.dot(Bsys,KBF)
print_modes_longi(AsysBF)

[Alin, Blin] = lin_longi(V, alt)
print_modes_longi(Alin)

AlinBF = Alin + np.dot(Blin,KBF)
print_modes_longi(AlinBF)



# Simulation linéaire
sim = sim_syslin(Asys, Bsys, du, KBF, Tf)

# Affichage simulation
fig, axs = plt.subplots(5, 1)
axs[0].plot(sim.t, sim.y[0])
axs[0].set_ylabel('spd')
axs[0].grid(True)

axs[1].plot(sim.t, sim.y[1])
axs[1].set_ylabel('gamma')
axs[1].grid(True)

axs[2].plot(sim.t, sim.y[2])
axs[2].set_ylabel('alpha')
axs[2].grid(True)

axs[3].plot(sim.t, sim.y[3])
axs[3].set_ylabel('q')
axs[3].grid(True)

axs[4].plot(sim.t, sim.y[4])
axs[4].set_ylabel('alt')
axs[4].grid(True)
axs[4].set_xlabel('time')

plt.show()

