"""
Min failing example when calculating an error in script. It hang when transfering data from one time stepper to another.
"""

from firedrake import (IcosahedralSphereMesh, errornorm,
                       FunctionSpace, Function)

# ---------------------------------------------------------------------------- #
# Test case parameters
# ---------------------------------------------------------------------------- #
R = 1.
ref_level= 4
degree = 1

# ---------------------------------------------------------------------------- #
# Set up stuff for first time stepper
# ---------------------------------------------------------------------------- #
mesh = IcosahedralSphereMesh(radius=R,
                             refinement_level=ref_level, degree=2)
V = FunctionSpace(mesh, "DG", degree)
D0 = Function(V)

# Zero initial condiitons
Dexpr = 0.0
D0.interpolate(Dexpr)

# Store these initial conditions
D_true = Function(D0.function_space())
D_true.dat.data[:] = D0.dat.data[:]

# ---------------------------------------------------------------------------- #
# Set up stuff for second time stepper
# ---------------------------------------------------------------------------- #
mesh = IcosahedralSphereMesh(radius=R,
                        refinement_level=ref_level, degree=2)
V = FunctionSpace(mesh, "DG", degree)
D0 = Function(V)

# Zero initial condiitons
Dexpr = 0.0
D0.interpolate(Dexpr)

D_sol = Function(D0.function_space())
print("IT HANGS HERE")
D_sol.dat.data[:] = D_true.dat.data[:]
error_D = errornorm(D_sol, D0, mesh=mesh)
print(error_D)