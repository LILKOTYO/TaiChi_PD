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
mac12x12 = lambda: ti.Matrix.field(12, 12, dtype=ti.f32)
# vectors and matrixs
x = ti.Vector.field(3, ti.f32, N)
v = ti.Vector.field(3, ti.f32, N)
elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, N_tetrahedron)
elements_V0 = ti.field(ti.f32, N_tetrahedron)
f = ti.Vector.field(3, ti.f32, N)
# tensors
delta_f = vec()
K = mac12x12()

# geometric components
tetrahedrons = ti.Vector.field(4, ti.i32, N_tetrahedron)
edges = ti.Vector.field(2, ti.i32, N_edges)

# allocate_tensors
# delta_f[i][j][k]: in the i-th tetrahedron,
# the delta force acting on the k-th mass point, which caused by
# the j-th component in the position vector
ti.root.dense(ti.i, N_tetrahedron).dense(ti.i, 12).dense(ti.j, 4).place(delta_f)
