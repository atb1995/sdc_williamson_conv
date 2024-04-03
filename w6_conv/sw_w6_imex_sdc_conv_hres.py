"""
The Williamson 6 test case, Rossby-Haurwitz wave

This uses an icosahedral mesh of the sphere, and runs a series of resolutions to find convergence.
"""

from re import L
from gusto import *
from firedrake import (IcosahedralSphereMesh, SpatialCoordinate,
                       as_vector, pi, sqrt, min_value, errornorm, norm, cos, sin,
                       acos, atan, grad, curl, div, conditional)
import sys
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #

day = 24.*60.*60.
# setup resolution and timestepping parameters for convergence test
dts = [ 1080.,900., 720., 450., 360. ]
dts = [540., 450.,360., 270.,225.,180.]
tmax = 5*day
ndumps = 1
ncell_1d = 48

dt_true = 10.

# setup shallow water parameters
a = 6371220.
H = 5960.
parameters = ShallowWaterParameters(H=H)
scheme_index= [0,1,2,3,4,5,6,7]

cols=['b','g','r','c']

# ref_level = 3
degree = 1
mesh = GeneralCubedSphereMesh(a, ncell_1d, degree=2)
x = SpatialCoordinate(mesh)
global_normal = x
mesh.init_cell_orientations(x)
domain = Domain(mesh, dt_true, 'RTCF', 1)

# Equations
lon, lat, _ = lonlatr_from_xyz(x[0], x[1], x[2])
Omega = parameters.Omega
fexpr = 2*Omega * x[2] / a
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_advection_form')
# Split continuity term
eqns = split_continuity_form(eqns)
# Label terms are implicit and explicit
eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
eqns.label_terms(lambda t: t.has_label(transport), explicit)
# I/O
dirname = "williamson_6_imex_sdc_C%s_dt%s_k%s_deg%s" % (ncell_1d, dt_true, 1, degree)
dumpfreq = int(tmax / (ndumps*dt_true))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        checkpoint_method = 'dumbcheckpoint')
io = IO(domain, output)
transported_fields = [RK4(domain,"u"),
                      RK4(domain,"D")]
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]

scheme =  SSP3(domain)
# Time stepper
#stepper = SemiImplicitQuasiNewton(eqns, io, transported_fields, transport_methods)
stepper = Timestepper(eqns, scheme, io, transport_methods)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #
u0 = stepper.fields('u')
D0 = stepper.fields('D')

Vu = domain.spaces("HDiv")
K = Constant(7.847e-6)
w = K
R = 4.
h0 = 8000.
g = parameters.g

# Intilising the velocity field
CG2 = FunctionSpace(mesh, 'CG', 2)
psi = Function(CG2)
psiexpr = -a**2 * w * sin(lat) + a**2 * K * cos(lat)**R * sin(lat) * cos(R*lon)
psi.interpolate(psiexpr)
uexpr = domain.perp(grad(psi))

# Initilising the depth field
A = (w / 2) * (2 * Omega + w) * cos(lat)**2 + 0.25 * K**2 * cos(lat)**(2 * R) * ((R + 1) * cos(lat)**2 + (2 * R**2 - R - 2) - 2 * R**2 * cos(lat)**(-2))
B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
B = B_frac * cos(lat)**R * ((R**2 + 2 * R + 2) - (R + 1)**2 * cos(lat)**2)
C = (1 / 4) * K**2 * cos(lat)**(2 * R) * ((R + 1)*cos(lat)**2 - (R + 2))
Dexpr = h0 * g + a**2 * (A + B*cos(lon*R) + C * cos(2 * R * lon))

# Finalizing fields and setting reference profiles
u0.project(uexpr)
D0.interpolate(Dexpr / g)
# Dbar is a background field for diagnostics
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
   #  cfl = 0.1

   #  dx = (2*pi*R/(12*day))*dt/(cfl)

   #  print(dx)

   #  ref_level = int(np.log2(2*pi*R*cos(atan(0.5))/(5.*dx))  )
   #  print(ref_level)
    #------------------------------------------------------------------------ #
    # Set up model objects
    mesh = GeneralCubedSphereMesh(a, ncell_1d, degree=2)
    x = SpatialCoordinate(mesh)
    global_normal = x
    mesh.init_cell_orientations(x)
    domain = Domain(mesh, dt, 'RTCF', 1)

    # Equations
    lon, lat, _ = lonlatr_from_xyz(x[0], x[1], x[2])
    Omega = parameters.Omega
    fexpr = 2*Omega * x[2] / a
    eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_advection_form')
    # Split continuity term
    eqns = split_continuity_form(eqns)
    # Label terms are implicit and explicit
    eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
    eqns.label_terms(lambda t: t.has_label(transport), explicit)
    for s in scheme_index:

        # I/O
        dirname = "williamson_6_imex_sdc_C%s_dt%s_k%s_deg%s" % (ncell_1d, dt, s, degree)
        dumpfreq = int(tmax / (ndumps*dt))
        output = OutputParameters(dirname=dirname,
                                dumpfreq=dumpfreq,
                                checkpoint_method = 'dumbcheckpoint')
        io = IO(domain, output)

        # Time stepper
        if (s==0):
          scheme=BackwardEuler(domain)
        elif (s==1):
          M=1
          k=1
          base_scheme=IMEX_Euler(domain)
          scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-legendre",final_update=True)
        elif(s==2):
           M=2
           k=2
           base_scheme=IMEX_Euler(domain)
           scheme=IMEX_SDC(base_scheme,domain, M, k, quadrature="gauss-legendre",final_update=True)
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

        Vu = domain.spaces("HDiv")
        K = Constant(7.847e-6)
        w = K
        R = 4.
        h0 = 8000.
        g = parameters.g

        # Intilising the velocity field
        CG2 = FunctionSpace(mesh, 'CG', 2)
        psi = Function(CG2)
        psiexpr = -a**2 * w * sin(lat) + a**2 * K * cos(lat)**R * sin(lat) * cos(R*lon)
        psi.interpolate(psiexpr)
        uexpr = domain.perp(grad(psi))

        # Initilising the depth field
        A = (w / 2) * (2 * Omega + w) * cos(lat)**2 + 0.25 * K**2 * cos(lat)**(2 * R) * ((R + 1) * cos(lat)**2 + (2 * R**2 - R - 2) - 2 * R**2 * cos(lat)**(-2))
        B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
        B = B_frac * cos(lat)**R * ((R**2 + 2 * R + 2) - (R + 1)**2 * cos(lat)**2)
        C = (1 / 4) * K**2 * cos(lat)**(2 * R) * ((R + 1)*cos(lat)**2 - (R + 2))
        Dexpr = h0 * g + a**2 * (A + B*cos(lon*R) + C * cos(2 * R * lon))

        # Finalizing fields and setting reference profiles
        u0.project(uexpr)
        D0.interpolate(Dexpr / g)
        # Dbar is a background field for diagnostics
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
