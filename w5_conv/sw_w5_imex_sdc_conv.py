"""
The Williamson 6 test case, Rossby-Haurwitz wave

This uses an icosahedral mesh of the sphere, and runs a series of resolutions to find convergence.
"""

from re import L
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, errornorm, norm, cos, sin,
                       acos, grad, curl, div, conditional)
import sys
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
# setup resolution and timestepping parameters for convergence test
dts = [ 600.,400.,300.]
dts = [ 600., 400.,300., 200., 100. ]
dts = [ 1440.,1200.,1080.,900. ]
tmax = 1.*day
ndumps = 1
#dts = [ 2160., 1800., 1440.,1200.,900. ]

dt_true = 10.

# setup shallow water parameters
a = 6371220.
H = 5960.
parameters = ShallowWaterParameters(H=H)
kvals_Mvals={8:4, 6:3, 4:2, 2:1}
kvals = [8, 6, 4, 2]

kvals_Mvals={ 3:3, 2:2 }

scheme_index= [0,1,2,3]

cols=['b','g','r','c']

ref_level = 3
degree = 1
#ref_level = 2
#degree = 1
mesh = IcosahedralSphereMesh(radius=a,
                             refinement_level=ref_level, degree=2)

x = SpatialCoordinate(mesh)

# Domain
domain = Domain(mesh, dt_true, 'BDM', degree)

# setup shallow water parameters
R = 6371220.
H = 5960.


# Equations
Omega = parameters.Omega
fexpr = 2*Omega*x[2]/R
theta, lamda, _ = lonlatr_from_xyz(x[0], x[1], x[2])
R0 = pi/9.
R0sq = R0**2
lamda_c = -pi/2.
lsq = (lamda - lamda_c)**2
theta_c = pi/6.
thsq = (theta - theta_c)**2
rsq = min_value(R0sq, lsq+thsq)
r = sqrt(rsq)
bexpr = 2000 * (1 - r/R0)
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)
# Split continuity term
eqns = split_continuity_form(eqns)
# Label terms are implicit and explicit
eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
eqns.label_terms(lambda t: t.has_label(transport), explicit)


# I/O
dirname = "williamson_5_imex_sdc_ref%s_dt%s_k%s_deg%s" % (ref_level, dt_true, 1, degree)
dumpfreq = int(tmax / (ndumps*dt_true))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        checkpoint_method = 'dumbcheckpoint')
io = IO(domain, output)
# transported_fields = [RK4(domain,"u"),
#                       RK4(domain,"D")]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

scheme =  ARK2(domain)
# Time stepper
#stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)
stepper = Timestepper(eqns, scheme, io, transport_methods)
# Time stepper
#stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)

 # ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

u0 = stepper.fields('u')
D0 = stepper.fields('D')
u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
g = parameters.g
Rsq = R**2
Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

u0.project(uexpr)
D0.interpolate(Dexpr)

Dbar = Function(D0.function_space()).assign(H)
stepper.set_reference_profiles([('D', Dbar)])

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)

u = stepper.fields('u')
D = stepper.fields('D')

utrue_data = u.dat.data[:]
Dtrue_data = D.dat.data[:]

for dt in dts:

    # ------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #
    mesh = IcosahedralSphereMesh(radius=a,
                                  refinement_level=ref_level, degree=2)

    x = SpatialCoordinate(mesh)
    # Domain
    domain = Domain(mesh, dt, 'BDM', degree)

    # Equation
    Omega = parameters.Omega
    fexpr = 2*Omega*x[2]/R
    theta, lamda, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    R0 = pi/9.
    R0sq = R0**2
    lamda_c = -pi/2.
    lsq = (lamda - lamda_c)**2
    theta_c = pi/6.
    thsq = (theta - theta_c)**2
    rsq = min_value(R0sq, lsq+thsq)
    r = sqrt(rsq)
    bexpr = 2000 * (1 - r/R0)
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, bexpr=bexpr)
    # Split continuity term
    eqns = split_continuity_form(eqns)
    # Label terms are implicit and explicit
    eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
    eqns.label_terms(lambda t: t.has_label(transport), explicit)

    for s in scheme_index:

        # I/O
        dirname = "williamson_5_ref%s_imex_sdc_gif_dt%s_k%s_deg%s" % (ref_level, dt, s, degree)
        dumpfreq = int(tmax / (ndumps*dt))
        output = OutputParameters(dirname=dirname,
                                dumpfreq=dumpfreq,
                                checkpoint_method = 'dumbcheckpoint')
        io = IO(domain, output)


        if (s==0):
          scheme=BackwardEuler(domain)
        elif (s==1):
          M=2
          k=2
          base_scheme=IMEX_Euler(domain)
          scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-radau",final_update=False)
        elif(s==2):
           M=3
           k=4
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-radau",final_update=False)
        elif(s==3):
           M=3
           k=4
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-legendre",final_update=True)
        elif(s==4):
           M=2
           k=1
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-radau",final_update=True)
        elif(s==5):
           M=3
           k=3
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-radau",final_update=True)
        elif(s==6):
           M=2
           k=2
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-radau",final_update=False)
        elif(s==7):
           M=3
           k=4
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-radau",final_update=False)
        elif(s==8):
           M=2
           k=1
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-lobatto",final_update=True)
        elif(s==9):
           M=3
           k=3
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-lobatto",final_update=True)
        elif(s==10):
           M=2
           k=1
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-lobatto",final_update=False)
        elif(s==11):
           M=3
           k=3
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-lobatto",final_update=False)
        transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

        # Time stepper
        stepper = Timestepper(eqns, scheme, io, transport_methods)
        # ------------------------------------------------------------------------ #
        # Initial conditions
        # ------------------------------------------------------------------------ #

        u0 = stepper.fields('u')
        D0 = stepper.fields('D')
        u_max = 20.   # Maximum amplitude of the zonal wind (m/s)
        uexpr = as_vector([-u_max*x[1]/R, u_max*x[0]/R, 0.0])
        g = parameters.g
        Rsq = R**2
        Dexpr = H - ((R * Omega * u_max + 0.5*u_max**2)*x[2]**2/Rsq)/g - bexpr

        u0.project(uexpr)
        D0.interpolate(Dexpr)

        Dbar = Function(D0.function_space()).assign(H)
        stepper.set_reference_profiles([('D', Dbar)])

        # ------------------------------------------------------------------------ #
        # Run
        # ------------------------------------------------------------------------ #

        stepper.run(t=0, tmax=tmax)

        u = stepper.fields('u')
        D = stepper.fields('D')

        usol = Function(u.function_space())
        Dsol = Function(D.function_space())

        usol.dat.data[:] = utrue_data
        Dsol.dat.data[:] = Dtrue_data

        error_norm_D = errornorm(Dsol, stepper.fields("D"), mesh=mesh)
        norm_D = norm(Dsol, mesh=mesh)

        error_norm_u = errornorm(usol, stepper.fields("u"), mesh=mesh)
        norm_u = norm(usol, mesh=mesh)

        print(dt,',', s,',', error_norm_D,',', norm_D,',', error_norm_u,',', norm_u)
