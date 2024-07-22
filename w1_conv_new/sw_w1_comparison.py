"""
The Williamson 1 test case (advection of gaussian hill), solved with a
discretisation of the non-linear advection equations.

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
dt = 900.
tmax = 1*day
ndumps = 1
# setup shallow water parameters
R = 6371220.
H = 5960.

ref_level= 4
degree = 1

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=2)

x = SpatialCoordinate(mesh)

# Domain
domain = Domain(mesh, dt, 'BDM', degree)
# Equation
V = domain.spaces('DG')
eqns = AdvectionEquation(domain, V, "D")

# I/O
dirname = "williamson_1_comp1_ref%s_dt%s_deg%s" % (ref_level, dt, degree)
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        checkpoint=True,
                        dump_nc=True,
                        dump_vtus=False,
                        checkpoint_method="checkpointfile",
                        chkptfreq=dumpfreq,
                        dumplist_latlon=['D'])
io = IO(domain, output)
solver_parameters = {'snes_type': 'ksponly',
                                       'ksp_type': 'cg',
                                       'pc_type': 'bjacobi',
                                       'sub_pc_type': 'ilu'}
node_dist = "LEGENDRE"
qdelta_imp="BE"
qdelta_exp="FE"
solver_parameters = {'snes_type': 'ksponly',
                        'ksp_type': 'cg',
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'}


# Time stepper

node_type="GAUSS"
M = 2
k = 2
base_scheme=ForwardEuler(domain,solver_parameters=solver_parameters)
scheme = FE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="node-to-node", final_update=True, initial_guess="base")
transport_methods = [ DGUpwind(eqns, "D")]
stepper = PrescribedTransport(eqns, scheme, io, transport_methods)

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #
u0 = stepper.fields('u')
D0 = stepper.fields('D')

u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
D_max = 1000.
#theta, lamda, _ = lonlatr_from_xyz(x[0], x[1], x[2])
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
lamda_c=3.*pi/2.
theta_c=0.
alpha=0.

# Intilising the velocity field
CG2 = FunctionSpace(mesh, 'CG', degree+1)
psi = Function(CG2)
psiexpr = -R*u_max*(sin(theta)*cos(alpha)-cos(alpha)*cos(theta)*sin(alpha))
psi.interpolate(psiexpr)
uexpr = domain.perp(grad(psi))
c_dist=R*acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c))

Dexpr = conditional(c_dist < R/3., 0.5*D_max*(1.+cos(3.*pi*c_dist/R)), 0.0)

u0.project(uexpr)
D0.interpolate(Dexpr)
# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)


D_imp = Function(D0.function_space())
D_imp.dat.data[:] = D0.dat.data[:]     

#     ------------------------------------------------------------------------ #
#     Set up model objects
#     ------------------------------------------------------------------------ #

#     Domain
x = SpatialCoordinate(mesh)
domain = Domain(mesh, dt, 'BDM', degree)

# Equation
V = domain.spaces('DG')
eqns = AdvectionEquation(domain, V, "D")
ik = 0
# I/O
dirname = "williamson_1_comp2_ref%s_dt%s_deg%s" % (ref_level, dt, degree)
dumpfreq = int(tmax / (ndumps*dt))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        checkpoint=True,
                        dump_nc=True,
                        dump_vtus=False,
                        checkpoint_method="checkpointfile",
                        chkptfreq=dumpfreq,
                        dumplist_latlon=['D'])
io = IO(domain, output)
node_dist = "LEGENDRE"
qdelta_imp="BE"
qdelta_exp="FE"
solver_parameters = {'snes_type': 'ksponly',
                        'ksp_type': 'cg',
                        'pc_type': 'bjacobi',
                        'sub_pc_type': 'ilu'}


# Time stepper

node_type="GAUSS"
M = 2
k = 2
base_scheme=ForwardEuler(domain,solver_parameters=solver_parameters)
scheme = FE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="zero-to-node", final_update=True, initial_guess="base")

transport_methods = [ DGUpwind(eqns, "D")]

stepper = PrescribedTransport(eqns, scheme, io, transport_methods)


u0 = stepper.fields("u")
D0 = stepper.fields("D")

# ------------------------------------------------------------------------ #
# Initial conditions
# ------------------------------------------------------------------------ #

x = SpatialCoordinate(mesh)
u_max = 2*pi*R/(12*day)  # Maximum amplitude of the zonal wind (m/s)
D_max = 1000.
lamda, theta, _ = lonlatr_from_xyz(x[0], x[1], x[2])
lamda_c=3.*pi/2.
theta_c=0.
alpha=0.

# Intilising the velocity field
CG2 = FunctionSpace(mesh, 'CG', degree+1)
psi = Function(CG2)
psiexpr = -R*u_max*(sin(theta)*cos(alpha)-cos(alpha)*cos(theta)*sin(alpha))
psi.interpolate(psiexpr)
uexpr = domain.perp(grad(psi))
c_dist=R*acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c))

Dexpr = conditional(c_dist < R/3., 0.5*D_max*(1.+cos(3.*pi*c_dist/R)), 0.0)

u0.project(uexpr)
D0.interpolate(Dexpr)

# ------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------ #

stepper.run(t=0, tmax=tmax)

u = stepper.fields('u')
D = stepper.fields('D')

print(errornorm(D_imp, D0, mesh=mesh)/ norm(D_imp, mesh=mesh))