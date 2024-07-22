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
dts = [ 1440.,1200., 1080., 900., 450.]
dt_true = 10.
tmax = 1*day
ndumps = 1

# setup shallow water parameters
a = 6371220.
H = 5960.
parameters = ShallowWaterParameters(H=H)
scheme_index= [0,1,2,3]

ref_level = 3
degree = 1

#------------------------------------------------------------------------ #
# Set up model objects
# ------------------------------------------------------------------------ #
mesh = IcosahedralSphereMesh(radius=a,
                              refinement_level=ref_level, degree=2)

x = SpatialCoordinate(mesh)
# Domain
domain = Domain(mesh, dt_true, 'BDM', degree)

# Equations
lon, lat, _ = lonlatr_from_xyz(x[0], x[1], x[2])
Omega = parameters.Omega
fexpr = 2*Omega * x[2] / a
eqns = ShallowWaterEquations(domain, parameters, fexpr=fexpr, u_transport_option='vector_advection_form')
# # Split continuity term
# eqns = split_continuity_form(eqns)
# # Label terms are implicit and explicit
# eqns.label_terms(lambda t: not any(t.has_label(time_derivative, transport)), implicit)
# eqns.label_terms(lambda t: t.has_label(transport), explicit)

scheme=ImplicitMidpoint(domain)
transport_methods = [DGUpwind(eqns, "u"), DGUpwind(eqns, "D")]


# I/O
dirname = "williamson_6_true_imex_comp_ref%s_dt%s_deg%s" % (ref_level, dt_true, degree)
dumpfreq = int(tmax / (ndumps*dt_true))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        checkpoint=True,
                        dump_nc=True,
                        dump_vtus=False,
                        checkpoint_method="checkpointfile",
                        chkptfreq=dumpfreq,
                        dumplist_latlon=['D','u'])
io = IO(domain, output)


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

for dt in dts:
    #------------------------------------------------------------------------ #
    # Set up model objects
    # ------------------------------------------------------------------------ #

    x = SpatialCoordinate(mesh)
    # Domain
    domain = Domain(mesh, dt, 'BDM', degree)

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
        dirname = "williamson_6_IMEX_SDC_comp_ref%s_dt%s_k%s_deg%s" % (ref_level, dt, s, degree)
        dumpfreq = int(tmax / (ndumps*dt))
        output = OutputParameters(dirname=dirname,
                                dumpfreq=dumpfreq,
                                checkpoint=True,
                                dump_nc=True,
                                dump_vtus=False,
                                checkpoint_method="checkpointfile",
                                chkptfreq=dumpfreq,
                                dumplist_latlon=['D','u'])
        io = IO(domain, output)
        node_dist = "LEGENDRE"
        qdelta_imp="BE"
        qdelta_exp="FE"
        solver_parameters = {'snes_type': 'newtonls',
                                                'ksp_type': 'gmres',
                                                'pc_type': 'bjacobi',
                                                'sub_pc_type': 'ilu'}

        # Time stepper
        if (s==0):
                node_type="GAUSS"
                M = 2
                k = 3
                base_scheme=IMEX_Euler(domain,solver_parameters=solver_parameters)
                scheme = IMEX_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="zero-to-node", final_update=True, initial_guess="base")
        elif(s==1):
                node_type="GAUSS"
                M=3
                k=5
                base_scheme=IMEX_Euler(domain,solver_parameters=solver_parameters)
                scheme = IMEX_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="zero-to-node", final_update=True  , initial_guess="base")
        elif (s==2):
                node_type="GAUSS"
                M = 2
                k = 3
                base_scheme=IMEX_Euler(domain,solver_parameters=solver_parameters)
                scheme = IMEX_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="node-to-node", final_update=True, initial_guess="base")
        elif(s==3):
                node_type="GAUSS"
                M=3
                k=5
                base_scheme=IMEX_Euler(domain,solver_parameters=solver_parameters)
                scheme = IMEX_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="node-to-node", final_update=True, initial_guess="base")
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