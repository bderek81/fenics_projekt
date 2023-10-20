import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, mesh, plot
from mpi4py import MPI
from petsc4py import PETSc

t = 3.0
num_steps = 100
plot_steps = True

# Domena i funkcijski prostor
#---------------------------------------------------------
domena = mesh.create_interval(points = (0, 3), comm = MPI.COMM_WORLD, nx = 1000)
x = ufl.SpatialCoordinate(domena)
V = fem.FunctionSpace(domena, ('CG', 1))

# Diskretizacija vremena
#---------------------------------------------------------
dt = fem.Constant(domena, PETSc.ScalarType(t / num_steps))

# Rub domene i rubni uvjeti
#---------------------------------------------------------
dim_domena = domena.topology.dim
dim_rub = dim_domena - 1
domena.topology.create_connectivity(dim_rub, dim_domena)
rub_indeks = mesh.exterior_facet_indices(domena.topology)
rub_dofs = fem.locate_dofs_topological(V, dim_rub, rub_indeks)

# Lijevi rubni uvjet
#---------------------------------------------------------
dofs_L = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
T_L = fem.Constant(domena, PETSc.ScalarType(1))
ru_L = fem.dirichletbc(T_L, dofs_L, V)

# Desni rubni uvjet
#---------------------------------------------------------
g = fem.Constant(domena, PETSc.ScalarType(1))

# Materijal
#---------------------------------------------------------
kappa = fem.Function(V)

cells_0 = mesh.locate_entities(domena, domena.topology.dim, lambda x: x[0] < 1)
kappa.x.array[cells_0] = np.full_like(cells_0, 8, dtype=PETSc.ScalarType)

cells_1 = mesh.locate_entities(domena, domena.topology.dim, lambda x: x[0] >= 1)
kappa.x.array[cells_1] = np.full_like(cells_1, 4, dtype=PETSc.ScalarType)

# Pocetni uvjet
#---------------------------------------------------------
T_n = fem.Function(V)
T_n.interpolate(lambda x: 1 - x[0] + 1/3 * x[0]**2)

# Slaba formulacija
#---------------------------------------------------------
T, v = ufl.TrialFunction(V), ufl.TestFunction(V)

f = fem.Constant(domena, PETSc.ScalarType(0))

a = T * v * ufl.dx + dt * ufl.inner(kappa * ufl.grad(T), ufl.grad(v)) * ufl.dx
b = T_n * v * ufl.dx + dt * f * v * ufl.dx + dt * kappa * g * v * ufl.ds
#b = T_n * v * ufl.dx + dt * kappa * g * v * ufl.ds

problem = fem.petsc.LinearProblem(a, b, bcs=[ru_L])

# Plotter, pocetno stanje
#---------------------------------------------------------
cells, types, x = plot.create_vtk_mesh(V)
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(10, 8))

plt.plot(x[:,0], T_n.x.array.real, linewidth = 3, color = 'red')

# Solver
#---------------------------------------------------------
for n in range(num_steps):
   T_n1 = problem.solve()

   T_n.interpolate(T_n1)

   if plot_steps and n + 1 < num_steps:
      plt.plot(x[:,0], T_n.x.array.real, color = 'purple')

plt.plot(x[:,0], T_n.x.array.real, linewidth = 3, color = 'navy')

plt.xlim([0, 3])
plt.ylim([0, 1.5])
ax.set_xlabel('x')
ax.set_ylabel('T')
ax.set_title(f't = {t}')
plt.grid()
plt.savefig('./rezultat/rezultat.png', bbox_inches='tight')
plt.show()
plt.close()