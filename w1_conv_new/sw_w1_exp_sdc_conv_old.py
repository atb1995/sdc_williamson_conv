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
dts = [ 900.,600., 400., 300.]
tmax = 1*day
ndumps = 1
# setup shallow water parameters
R = 6371220.
H = 5960.
ref_level= 4
degree = 1

dt_true = 50.

mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=2)

x = SpatialCoordinate(mesh)

# Domain
domain = Domain(mesh, dt_true, 'BDM', degree)
# Equation
V = domain.spaces('DG')
eqns = AdvectionEquation(domain, V, "D")

# I/O
dirname = "williamson_1_true_old4_ref%s_dt%s_deg%s" % (ref_level, dt_true, degree)
dumpfreq = int(tmax / (ndumps*dt_true))
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
scheme = SSPRK3(domain, solver_parameters=solver_parameters)
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

u = stepper.fields('u')
D = stepper.fields('D')
D_true = Function(D.function_space())
D_true.dat.data[:] = D.dat.data[:]

scheme_index= [1,2]
for dt in dts:

        #     ------------------------------------------------------------------------ #
        #     Set up model objects
        #     ------------------------------------------------------------------------ #

        #     Domain
        mesh = IcosahedralSphereMesh(radius=R,
                                refinement_level=ref_level, degree=2)

        x = SpatialCoordinate(mesh)
        domain = Domain(mesh, dt, 'BDM', degree)

        # Equation
        V = domain.spaces('DG')
        eqns = AdvectionEquation(domain, V, "D")
        ik = 0
        for s in scheme_index:

                # I/O
                dirname = "williamson_1_EX_SDC_n2n_old4_ref%s_dt%s_k%s_deg%s" % (ref_level, dt, s, degree)
                dumpfreq = int(tmax / (ndumps*dt))
                print(dumpfreq)
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
                if (s==0):
                        scheme=ForwardEuler(domain, solver_parameters=solver_parameters)
                elif (s==1):
                        node_type="RADAU-RIGHT"
                        M = 2
                        k = 2
                        base_scheme=ForwardEuler(domain,solver_parameters=solver_parameters)
                        scheme = FE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="node-to-node", final_update=False, initial_guess="base")
                elif(s==2):
                        node_type="RADAU-RIGHT"
                        M=3
                        k=4
                        base_scheme=ForwardEuler(domain,solver_parameters=solver_parameters)
                        scheme = FE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="node-to-node", final_update=False, initial_guess="base")
                elif(s==3):
                        node_type="GAUSS"
                        M=3
                        k=4
                        base_scheme=ForwardEuler(domain,solver_parameters=solver_parameters)
                        scheme = FE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, formulation="node-to-node", final_update=False, initial_guess="base")

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

                D_sol = Function(D.function_space())
                D_sol.dat.data[:] = D_true.dat.data[:]

                error_D = errornorm(D_sol, D, mesh=mesh)/ norm(D_sol, mesh=mesh)

                print("dt, error, k:",dt, error_D, k)