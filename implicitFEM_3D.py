import taichi as ti
import math

ti.init(arch=ti.cpu)

# initialize settings
init_x, init_y, init_z = 0.3, 0.3, 0.3
N_x = 10
N_y = 10
N_z = 10
N = N_x * N_y * N_z
# axis-x + axis-y + axis-z + diagonal_xy + diagonal_xz + diagonal_yz
N_edges = (N_x - 1) * N_y * N_z + (N_y - 1) * N_x * N_z + (N_z - 1) * N_x * N_y \
    + (N_x - 1) * (N_y - 1) * N_z + (N_x - 1) * (N_z - 1) * N_y + (N_y - 1) * (N_z - 1) * N_x
N_tetrahedron = 5 * (N_x - 1) * (N_y - 1) * (N_z - 1)
dx = 0.5 / N_x

# physical quantities
m = 1
g = 0.5
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# sub-step
N_substeps = 100
# time-step size (for time integration)
dh = h / N_substeps

# simulation components
scalar = lambda: ti.field(dtype=ti.f32)
vec = lambda: ti.Vector.field(3, dtype=ti.f32)
mac3x3 = lambda: ti.Matrix.field(3, 3, dytpe=ti.f32)

# vectors and matrixs
x = ti.Vector.field(3, ti.f32, N)
v = ti.Vector.field(3, ti.f32, N)
elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, N_tetrahedron)
# elements_V0 = ti.field(ti.f32, N_tetrahedron)
elements_V0 = ti.field(ti.f32, 5)
f = ti.Vector.field(3, ti.f32, N)

# geometric components
tetrahedrons = ti.Vector.field(4, ti.i32, N_tetrahedron)
edges = ti.Vector.field(2, ti.i32, N_edges)

# derivatives
dD = mac3x3()
dF = mac3x3()
dP = mac3x3()
dH = mac3x3()

# allocate_tensors
# dD[i][j]: the difference caused by the change of j-th component in the i-th node
ti.root.dense(ti.i, 4).dense(ti.j, 3).place(dD)
# dD[k][i][j]: the difference of the k-th tetrahedron's F in a cube
# caused by the change of j-th component in the i-th node
ti.root.dense(ti.i, 5).dense(ti.i, 4).dense(ti.j, 3).place(dF)
# dP[e][i][j]: the difference of the e-th tetrahedron's P
# cause by the change of the j-th component in the i-th node
ti.root.dense(ti.i, N_tetrahedron).dense(ti.i, 4).dense(ti.j, 3).place(dP, dH)

# df/dx
K_builder = ti.linalg.SparseMatrixBuilder(3*N, 3*N, max_num_triplets=9*N*N)

# for solving system of linear equations
A_builder = ti.linalg.SparseMatrixBuilder(3*N, 3*N, max_num_triplets=9*N*N)
b = ti.field(ti.f32, shape=3*N)
x = ti.field(ti.f32, shape=3*N)

