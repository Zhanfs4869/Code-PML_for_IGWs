"""
This is the script used to compute, for different values of A, 
the minimum time step that allows stable iterations in the pPML method.
"""


import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import sys
import os
from pathlib import Path

DT_INIT   = 2*np.pi/np.sqrt(0.3)*0.01    # initial timestep
DT_FLOOR  = 1e-7    # lower bound (if everything is stable down to this value, return this)
MAX_DECADES = 12    # search at most 10^12 smaller to find the bracket
N_PERIODS_TEST = 1  # number of periods tested for each dt
U_BLOWUP = 1e6      # velocity blowup threshold
E_BLOWUP = 1e12     # energy blowup threshold
REL_TE_TOL = 1e-4   # TE relative-change tolerance (consistent with original logic)

# bisection stopping criteria
DT_ABS_TOL = 1e-5           # absolute tolerance
DT_REL_TOL = 1e-1           # relative tolerance (relative to dt_hi)
MAX_BISECT_ITERS = 40       # max bisection iterations


def run_one_dt(timestep, record_ke=False, ):
    """
    Returns: (is_stable, KEz or None)

    - Advance for N_PERIODS_TEST * T.
    - Any exception/NaN/Inf/blowup => unstable.
    - If TE relative change < REL_TE_TOL, treat as stable early exit.
    - During search phase KEz is not written; if record_ke=True, write KEz.dat at the end.
    """
    import numpy as np
    import dedalus.public as d3
    import logging
    logger = logging.getLogger(__name__)
    import sys
    import os
    from pathlib import Path
    
    timestepper = d3.SBDF2
    dtype = np.float64
    Lz=1
    
    i = int(sys.argv[1])

    # Parameters
    Nx, Nz = 4, 128
    k = 1
    Lx, Lz = 2*np.pi, 1
    N2 = 0.3

    A_list = np.linspace(np.log10(0.1), np.log10(10), num=101, endpoint=True)
    A_list = (10**A_list)/np.sqrt(N2)
    A = A_list[i]
    
    om=0.3*np.sqrt(N2)

    gamma=5/3             # heat capacity ratio
    m2 = 3                # polytrope index
    k2 = 0.5404           # temperature gradient

    Dz=0.05
    T = 2*np.pi/om

    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz))

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

    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A_local, n: d3.Lift(A_local, lift_basis, n)

    coskx = dist.Field(bases=xbasis); coskx['g'] = np.cos(k*x); coskx = d3.Grid(coskx).evaluate()
    sinkx = dist.Field(bases=xbasis); sinkx['g'] = np.sin(k*x); sinkx = d3.Grid(sinkx).evaluate()

    # Setup of background profile
    def dSc0_profile(z):
        dSc_upper = (1/gamma)*k2/(1-k2*(z-1))
        return dSc_upper

    def log_profile(z):
        log_upper = -k2*m2/(1 - k2 * (z - 1))
        return log_upper

    def rho0_profile(z):
        rho_upper = (1 - k2 * (z - 1))**m2
        return rho_upper

    # Setup of pseudo-PML region
    def sig_profile(z, Dz, A_local):
        S =  A_local*(1 - np.tanh((z-4*Dz)/Dz)) # PML
        return S
        
    sig['g']  = sig_profile(z, 0.025, A)
    dlog['g'] = log_profile(z)
    dSc0['g'] = dSc0_profile(z)
    rho0['g'] = rho0_profile(z)

    # Governing equations
    problem = d3.IVP([w, u, Sc, phi1, phi2, tau_u1, tau_u2, tau_w], time=t, namespace=locals())
    problem.add_equation("dt(u) + grad(w) + ez*lift(tau_u1,-1) - Sc*ez + ez*phi1 = 0")
    problem.add_equation("dt(Sc) + dSc0*(ez@u) = 0")
    problem.add_equation("div(u) + (ez@u)*dlog + tau_w + phi2 + lift(tau_u2, -1) = 0")
    problem.add_equation("sig*dt(phi1 + ez@grad(w)) + phi1 = 0")
    problem.add_equation("sig*dt(phi2 + ez@grad(ez@u)) + phi2 = 0")
    problem.add_equation("(ez@u)(z=0) = 0")
    problem.add_equation("(ez@u)(z=Lz) = sinkx*np.cos(om*t) - coskx*np.sin(om*t)")
    problem.add_equation("integ(w) = 0")

    #Solver
    solver = problem.build_solver(timestepper)
    if record_ke:
        solver.stop_sim_time = np.inf
    else:
        solver.stop_sim_time = N_PERIODS_TEST * T
    solver.stop_iteration = 10000

    # Field properties
    TE_op = d3.integ(u@u + (Sc**2)/dSc0)
    KE_op = d3.integ((ez@u)*(ez@u))

    TE_prev = 0.0
    set=1

    try:
        logger.info('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            
            if (solver.iteration % 400) == 0:
                TE_new = TE_op['g'][0,0]
                KE_new = KE_op['g'][0,0]
                
                logger.info("Iteration=%i, Time=%e, dt=%e,\
                TE=%e, KE=%e" %(solver.iteration, solver.sim_time, \
                    timestep, TE_new, KE_new))

                if (not np.isfinite(TE_new)) or (not np.isfinite(KE_new)):
                    set=0
                    break
                if  (abs(TE_new) > E_BLOWUP) or (abs(KE_new) > E_BLOWUP):
                    set=0
                    break

                if np.abs(TE_prev - TE_new)/TE_new < 1e-4:
                    print(TE_prev)
                    print(TE_new)
                    set=1
                    break
                TE_prev = TE_new

    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    
    finally:
        if set==0:
            return False, None, A
        elif set==1:
            return True, None, A


def find_bracket():
    """
    Return (dt_lo_stable, dt_hi_unstable) as the bracketing interval for bisection.

    Strategy:
      - Test DT_INIT:
         * If stable: keep dividing by 10 until the first unstable, giving (stable, unstable).
         * If unstable: keep dividing by 10 until the first stable, giving (stable, unstable).
      - If always stable down to DT_FLOOR: return (DT_FLOOR, None), meaning no upper unstable bound.
      - If always unstable down to DT_FLOOR: return (None, DT_FLOOR), meaning no lower stable bound.
    """
    print("1")
    dt = DT_INIT
    ok, _, _ = run_one_dt(dt)
    print(ok)
    if ok:
        # from stable, search downward for unstable bound
        dt_lo = dt
        for _ in range(MAX_DECADES):
            dt_next = dt/10.0
            if dt_next < DT_FLOOR:
                return (DT_FLOOR, None)  # always stable
            ok2, _, _ = run_one_dt(dt_next)
            print(ok2)
            if ok2:
                dt = dt_next
                dt_lo = dt_next
            else:
                dt_hi = dt_next
                return (dt_lo, dt_hi)
        return (DT_FLOOR, None)
    else:
        # from unstable, search downward for first stable
        dt_hi = dt
        for _ in range(MAX_DECADES):
            dt_next = dt/10.0
            if dt_next < DT_FLOOR:
                return (None, DT_FLOOR)  # always unstable
            ok2, _, _ = run_one_dt(dt_next)
            if ok2:
                dt_lo = dt_next
                return (dt_lo, dt_hi)
            else:
                dt = dt_next
                dt_hi = dt_next
        return (None, DT_FLOOR)

def bisect_min_stable(dt_lo, dt_hi):
    """
    Given dt_lo stable and dt_hi unstable, bisect within [dt_lo, dt_hi],
    and return the approximate minimum stable dt (we return the last stable value dt_lo).
    """
    lo = dt_lo
    hi = dt_hi

    for _ in range(MAX_BISECT_ITERS):
        mid = 0.5*(lo + hi)

        # stopping rule
        if (lo - hi) <= DT_REL_TOL*hi:
            break

        ok, _, _ = run_one_dt(mid)
        if ok:
            lo = mid   # mid is stable, push upward toward the unstable boundary
        else:
            hi = mid   # mid is unstable, shrink upper bound

    return lo

# ---------------- Main script ----------------
if __name__ == "__main__":
    dt_lo, dt_hi = find_bracket()
    print(dt_lo, dt_hi)

    if dt_lo is not None and dt_hi is not None:
        dt_min = bisect_min_stable(dt_lo, dt_hi)
        _ok, _, om = run_one_dt(dt_min, record_ke=False)

    elif dt_lo is not None and dt_hi is None:
        dt_min = DT_FLOOR
        _ok, _, om = run_one_dt(dt_min, record_ke=False)

    elif dt_lo is None and dt_hi is not None:
        dt_min = DT_FLOOR
    else:
        dt_min = DT_FLOOR

    print(f"omega={om:.6e}, min_stable_dt={dt_min:.3e}")

    # record dt_min
    if os.path.exists('dt_min.dat'):
        arr = np.loadtxt('dt_min.dat')
        newrow = np.array([[om, dt_min]])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = np.vstack([arr, newrow])
        np.savetxt('dt_min.dat', out)
    else:
        np.savetxt('dt_min.dat', np.array([[om, dt_min]]))
