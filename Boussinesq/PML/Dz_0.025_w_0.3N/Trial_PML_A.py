"""
This is the script used to compute how the effectiveness of PML varies with A,
for a fixed forcing frequency of 0.3N and Dz=0.025, corresponding to Fig. 3(a)
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
N2 = 1
om = 0.3

A_list = np.linspace(0.5, 21, num=61, endpoint=True)
A = A_list[i]

Dz=0.025

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
b = dist.Field(name='b', bases=(xbasis, zbasis))
p = dist.Field(name='p', bases=(xbasis, zbasis))
phi1 = dist.Field(name='phi1', bases=(xbasis, zbasis))
phi2 = dist.Field(name='phi2', bases=(xbasis, zbasis))

sig = dist.Field(name='sig', bases=zbasis)
tau_u1 = dist.Field(name='tau_u1', bases=xbasis)
tau_u2 = dist.Field(name='tau_u2', bases=xbasis)
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)

tau_p = dist.Field(name='tau_p')
t = dist.Field(name='t')

# Setup of PML region
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

sig['g'] = sig_profile(z, Dz, A)

grad_b = d3.grad(b) + ez*lift(tau_b1, -1)
grad_p = ez@d3.grad(p)


# Governing equations
problem = d3.IVP([u, b, p, phi1, phi2, tau_u1, tau_u2, tau_p], time=t, namespace=locals())
problem.add_equation("dt(u) + grad(p) + phi1*ez - b*ez + ez*lift(tau_u1, -1) = 0")
problem.add_equation("sig*dt(phi1 + ez@grad(p)) + phi1 = 0")
problem.add_equation("dt(b) + N2*ez@u = 0")
problem.add_equation("div(u) + tau_p + phi2 + lift(tau_u2, -1) = 0")
problem.add_equation("sig*dt(phi2 + ez@grad(ez@u)) + phi2 = 0")
problem.add_equation("(ez@u)(z=0) = 0")
problem.add_equation("(ez@u)(z=Lz) = sinkx*np.cos(om*t) - coskx*np.sin(om*t)")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_iteration = stop_iteration

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')

TE_op = d3.integ(u@u + b*b)  #Total energy
KE_op = d3.integ((ez@u)*(ez@u))  #Vertical kinetic energy

TE = 0

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            TE_new = TE_op['g'][0,0]
            KE_new = KE_op['g'][0,0]
            logger.info("Iteration=%i, Time=%e, TE=%e,\
                         KE=%e" %(solver.iteration, solver.sim_time, TE_new, KE_new))
            
        if (solver.iteration) % 1000 == 0:         
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
    KEz_op = d3.integ((ez@u)*(ez@u))/Lx/Lz
    KEz = KEz_op['g'][0,0]  # Averaged vertical velocity square

    if os.path.exists('KEz.dat'):
        data = np.loadtxt('KEz.dat')
        data = np.vstack((data, [om, KEz, A]))
        np.savetxt('KEz.dat', data)
    else:
        data = np.array([om, KEz, A])
        np.savetxt('KEz.dat', data)

