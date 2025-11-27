"""
This is the script used to compute the effectiveness of pPML for different values of A 
when Dz=0.025, corresponding to Fig. 7(a)
"""


import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

i = int(sys.argv[1])

# Parameters
Nx, Nz = 4, 128
k = 1
Lx, Lz = 2*np.pi, 1
N2 = 0.3

om_list = np.linspace(np.log10(0.05), np.log10(0.7), num=1001, endpoint=True)
om_list = 10**(om_list)*np.sqrt(N2)
om = om_list[i]

gamma=5/3             # Heat capacity ratio
m2 = 3                # polytrope
k2 = 0.5404           # The gradient of temperature

Dz=0.05
A = 0.5/(np.sqrt(0.3)) # Choice of A
# A = 3/(np.sqrt(0.3))
# A = 10/(np.sqrt(0.3))

timestep = 2*np.pi/np.sqrt(N2)*0.01
stop_iteration = np.inf
timestepper = d3.SBDF2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz))

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
Sc = dist.Field(name='Sc', bases=(xbasis, zbasis))
w = dist.Field(name='w', bases=(xbasis, zbasis))
phi1 = dist.Field(name='phi1', bases=(xbasis, zbasis))
phi2 = dist.Field(name='phi2', bases=(xbasis, zbasis))

sig = dist.Field(name='sig', bases=zbasis)
tau_u1 = dist.Field(name='tau_u1', bases=xbasis)
tau_u2 = dist.Field(name='tau_u2', bases=xbasis)

dSc0 = dist.Field(name='dSc0', bases=zbasis)
dlog = dist.Field(name='dlog',bases=zbasis)
rho0 = dist.Field(name='rho0',bases=zbasis)

tau_w = dist.Field(name='tau_w')
t = dist.Field(name='t')

# Setup of Background Profile
def dSc0_profile(z):
    # Define the entropy profile
    dSc_upper = (1/gamma)*k2/(1-k2*(z-1))
    return dSc_upper

def log_profile(z):
    # Define the log density profile
    log_upper = -k2*m2/(1 - k2 * (z - 1))
    return log_upper

def rho0_profile(z):
    # Define the density profile
    rho_upper = (1 - k2 * (z - 1))**m2
    return rho_upper

# Setup of pseudo-PML region
def sig_profile(z, Dz, A):
    S =  A*(1 - np.tanh((z-4*Dz)/Dz))
    return S

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

coskx = dist.Field(bases=xbasis)
coskx['g'] = np.cos(k*x)
coskx = d3.Grid(coskx).evaluate()

sinkx = dist.Field(bases=xbasis)
sinkx['g'] = np.sin(k*x)
sinkx = d3.Grid(sinkx).evaluate()

sig['g'] = sig_profile(z, 0.025, A)
dlog['g'] = log_profile(z, Dz)
dSc0['g'] = dSc0_profile(z, Dz)
rho0['g'] = rho0_profile(z, Dz)

grad_w = ez@d3.grad(w)


# Governing equations
problem = d3.IVP([w, u, Sc, phi1, phi2, tau_u1, tau_u2, tau_w], time=t, namespace=locals())
problem.add_equation("dt(u) +grad(w) + ez*lift(tau_u1,-1) - Sc*ez + ez*phi1 = 0")
problem.add_equation("dt(Sc) + dSc0*(ez@u) = 0")
problem.add_equation("div(u) + (ez@u)*dlog+ tau_w + phi2 + lift(tau_u2, -1) = 0")
problem.add_equation("sig*dt(phi1 + ez@grad(w)) + phi1 = 0")
problem.add_equation("sig*dt(phi2 + ez@grad(ez@u)) + phi2 = 0")
problem.add_equation("(ez@u)(z=0) = 0")
problem.add_equation("(ez@u)(z=Lz) = sinkx*np.cos(om*t) - coskx*np.sin(om*t)")
problem.add_equation("integ(w) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_iteration = stop_iteration

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')

TE_op = d3.integ(rho0*u@u+rho0*(Sc**2)/dSc0)
KE_op = d3.integ(rho0*(ez@u)*(ez@u))

TE = 0

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            TE_new = TE_op['g'][0,0]
            KE_new = KE_op['g'][0,0]
            logger.info("Iteration=%i, Time=%e, TE=%e, KE=%e" %(solver.iteration, solver.sim_time, TE_new, KE_new))
            
        if (solver.iteration) % 400 == 0:
            TE_new = TE_op['g'][0,0]
            if np.abs(TE - TE_new)/TE_new < 1e-4:
                print(TE)
                break
            TE = TE_new

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
    KEz_op = d3.integ(rho0*(ez@u)*(ez@u))/Lx/Lz
    KEz = KEz_op['g'][0,0]

    if os.path.exists('KEz.dat'):
        data = np.loadtxt('KEz.dat')
        data = np.vstack((data, [om, KEz, A]))
        np.savetxt('KEz.dat', data)
    else:
        data = np.array([om, KEz, A])
        np.savetxt('KEz.dat', data)

