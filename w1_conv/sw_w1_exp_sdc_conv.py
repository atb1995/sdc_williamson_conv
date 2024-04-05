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
dts = [ 900., 400., 300.]
tmax = 1*day
ndumps = 1
# setup shallow water parameters
R = 6371220.
H = 5960.
dt_true = 50.

cols=['b','g','r','c']


ref_level= 3
degree = 1

scheme_index= [0,2]
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=2)

x = SpatialCoordinate(mesh)

# Domain
domain = Domain(mesh, dt_true, 'BDM', degree)
# Equation
V = domain.spaces('DG')
eqns = AdvectionEquation(domain, V, "D")

# I/O
dirname = "will1_sol_1_ref%s_dt%s_k%s_deg%s" % (ref_level, dt_true, 1, degree)
dumpfreq = int(tmax / (ndumps*dt_true))
output = OutputParameters(dirname=dirname,
                        dumpfreq=dumpfreq,
                        checkpoint_method = 'dumbcheckpoint')
io = IO(domain, output)
scheme = RK4(domain)
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

utrue_data = u.dat.data[:]
Dtrue_data = D.dat.data[:]

# print('dt,k, errornorm, norm')
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
                dirname = "will1_sol_1_ref%s_dt%s_k%s_deg%s" % (ref_level, dt, s, degree)
                dumpfreq = int(tmax / (ndumps*dt))
                output = OutputParameters(dirname=dirname,
                                        dumpfreq=dumpfreq,
                                        checkpoint_method = 'dumbcheckpoint')
                io = IO(domain, output)

                # Time stepper
                node_type = "RADAU-RIGHT"
                node_dist = "LEGENDRE"
                qdelta_imp="BE"
                qdelta_exp="FE"
                if (s==0):
                        butcher_exp =  np.array([[1./3., 0., 0.,0.,0.,0.,0.], [1./3., 2./3., 0.,0.,0.,0.,0.],
                                                  [0., 5./12., -1./12.,0.,0.,0.,0.],[0., 1./12., 0.25,2./3.,0.,0.,0.],[0., 0., 0.,5./12.,-1./12.,0.,0.],
                                                  [0., 0., 0.,1./12.,0.25,2./3.,0.],[0., 0., 0.,1./12.,0.25,2./3.,0.]]) 
                        # M = 2
                        # k = 2
                        scheme = ExplicitMultistage(domain,butcher_exp)
                        #scheme = Euler_SDC_QD("explicit", base_scheme,domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True, initial_guess="copy")
                        #scheme = TrapeziumRule(domain)
                elif (s==1):
                        M = 2
                        k = 2
                        base_scheme = BackwardEuler(domain)
                        scheme = Euler_SDC_QD("implicit", base_scheme,domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True, initial_guess="copy")
                elif (s==2):
                        M = 2
                        k = 2
                        base_scheme = ForwardEuler(domain)
                        scheme = FE_SDC(base_scheme,domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True, initial_guess="copy")
                elif (s==3):
                        M = 2
                        k = 2
                        base_scheme = BackwardEuler(domain)
                        scheme = BE_SDC(base_scheme,domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True, initial_guess="copy")
                elif(s==4):
                        #scheme= SSPRK3(domain)
                        M=3
                        k=4
                        base_scheme = ForwardEuler(domain)
                        scheme = FE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp)
                elif(s==5):
                        #scheme = RK4(domain)
                        M = 3
                        k = 4
                        base_scheme = BackwardEuler(domain)
                        scheme = BE_SDC(base_scheme,domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp, final_update=True)
                elif(s==6):
                        #scheme= SSPRK3(domain)
                        M=3
                        k =3
                        base_scheme = ForwardEuler(domain)
                        scheme = FE_SDC(base_scheme, domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp)

                elif(s==7):
                        #scheme = RK4(domain)
                        M=3
                        k =3
                        base_scheme = BackwardEuler(domain)
                        scheme = BE_SDC(base_scheme,domain, M, k, node_type, node_dist, qdelta_imp, qdelta_exp)

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
                lamda_t=3.*pi/2. + 2.*pi/12.
                theta_c=0.
                alpha=0.

                # Intilising the velocity field
                CG2 = FunctionSpace(mesh, 'CG', degree+1)
                psi = Function(CG2)
                psiexpr = -R*u_max*(sin(theta)*cos(alpha)-cos(alpha)*cos(theta)*sin(alpha))
                psi.interpolate(psiexpr)
                uexpr = domain.perp(grad(psi))
                c_dist=R*acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_c))
                c_dist_t=R*acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lamda-lamda_t))


                Dexpr = conditional(c_dist < R/3., 0.5*D_max*(1.+cos(3.*pi*c_dist/R)), 0.0)

                Dexpr_sol = conditional(c_dist_t < R/3., 0.5*D_max*(1.+cos(3.*pi*c_dist_t/R)), 0.0)

                u0.project(uexpr)
                D0.interpolate(Dexpr)

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
                #usol.project(uexpr)
                #Dsol.interpolate(Dexpr_sol)


                error_norm_D = errornorm(Dsol, stepper.fields("D"), mesh=mesh)
                norm_D = norm(Dsol, mesh=mesh)
                error_D=error_norm_D/norm_D

                print(dt,',',s,',',error_norm_D,',',norm_D)
