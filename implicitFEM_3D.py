import taichi as ti
import numpy as np
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
mass = 1
gravity = 0.5
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
# elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, N_tetrahedron)
# elements_V0 = ti.field(ti.f32, N_tetrahedron)
elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, 5)
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
# dF[k][i][j]: the difference of the k-th tetrahedron's F in a cube
# caused by the change of j-th component in the i-th node
ti.root.dense(ti.i, 5).dense(ti.i, 4).dense(ti.j, 3).place(dF)
# dP[e][i][j]: the difference of the e-th tetrahedron's P
# cause by the change of the j-th component in the i-th node
ti.root.dense(ti.i, N_tetrahedron).dense(ti.i, 4).dense(ti.j, 3).place(dP, dH)

# for solving system of linear equations

KA_builder = ti.linalg.SparseMatrixBuilder(3*N, 3*N, max_num_triplets=9*N*N)
b = ti.field(ti.f32, shape=3*N)
x = ti.field(ti.f32, shape=3*N)


@ti.func
def ijk_2_index(i, j, k): return k * N_x * N_y + j * N_x + i

# -----------------------meshing and init----------------------------
@ti.kernel
def meshing():
    # setting up tetrahedrons
    for i, j, k in ti.ndrange(N_x - 1, N_y - 1, N_z - 1):
        # tetrahedron id
        tid = 5 * (k * (N_x - 1) * (N_y - 1) + j * (N_x - 1) + i)
        tetrahedrons[tid][0] = ijk_2_index(i, j, k + 1)
        tetrahedrons[tid][1] = ijk_2_index(i + 1, j, k + 1)
        tetrahedrons[tid][2] = ijk_2_index(i + 1, j, k)
        tetrahedrons[tid][3] = ijk_2_index(i + 1, j + 1, k + 1)

        tid += 1
        tetrahedrons[tid][0] = ijk_2_index(i, j, k + 1)
        tetrahedrons[tid][1] = ijk_2_index(i, j, k)
        tetrahedrons[tid][2] = ijk_2_index(i + 1, j, k)
        tetrahedrons[tid][3] = ijk_2_index(i, j + 1, k)

        tid += 1
        tetrahedrons[tid][0] = ijk_2_index(i, j, k + 1)
        tetrahedrons[tid][1] = ijk_2_index(i, j + 1, k + 1)
        tetrahedrons[tid][2] = ijk_2_index(i, j + 1, k)
        tetrahedrons[tid][3] = ijk_2_index(i + 1, j + 1, k + 1)

        tid += 1
        tetrahedrons[tid][0] = ijk_2_index(i + 1, j, k)
        tetrahedrons[tid][1] = ijk_2_index(i + 1, j + 1, k)
        tetrahedrons[tid][2] = ijk_2_index(i + 1, j + 1, k + 1)
        tetrahedrons[tid][3] = ijk_2_index(i, j + 1, k)

        tid += 1
        tetrahedrons[tid][0] = ijk_2_index(i, j, k + 1)
        tetrahedrons[tid][1] = ijk_2_index(i + 1, j, k)
        tetrahedrons[tid][2] = ijk_2_index(i + 1, j + 1, k + 1)
        tetrahedrons[tid][3] = ijk_2_index(i, j + 1, k)

    # setting up edges
    # edge id
    eid_base = 0

    # axis-x edges
    for i in range(N_x-1):
        for j, k in ti.ndrange(N_y, N_z):
            eid = eid_base + k * (N_x - 1) * N_y + j * (N_x - 1) + i
            edges[eid] = [ijk_2_index(i, j, k), ijk_2_index(i + 1, j, k)]

    eid_base += (N_x - 1) * N_y * N_z
    # axis-y edges
    for j in range(N_y-1):
        for i, k in ti.ndrange(N_x, N_z):
            eid = eid_base + k * (N_y - 1) * N_x + i * (N_y - 1) + j
            edges[eid] = [ijk_2_index(i, j, k), ijk_2_index(i, j + 1, k)]

    eid_base += N_x * (N_y - 1) * N_z
    # axis-z edges
    for k in range(N_z-1):
        for i, j in ti.ndrange(N_x, N_y):
            eid = eid_base + i * (N_z - 1) * N_y + j * (N_z - 1) + k
            edges[eid] = [ijk_2_index(i, j, k), ijk_2_index(i, j, k + 1)]

    eid_base += N_x * N_y * (N_z - 1)
    # diagonal_xy
    for k in range(N_z):
        for i, j in ti.ndrange(N_x-1, N_y-1):
            eid = eid_base + k * (N_x - 1) * (N_y - 1) + j * (N_x - 1) + i
            edges[eid] = [ijk_2_index(i, j, k), ijk_2_index(i + 1, j + 1, k)]

    eid_base += (N_x - 1) * (N_y - 1) * N_z
    # diagonal_xz
    for j in range(N_y):
        for i, k in ti.ndrange(N_x-1, N_z-1):
            eid = eid_base + j * (N_x - 1) * (N_z - 1) + k * (N_x - 1) + i
            edges[eid] = [ijk_2_index(i, j, k), ijk_2_index(i + 1, j, k + 1)]

    eid_base += (N_x - 1) * N_y * (N_z - 1)
    # diagonal_yz
    for i in range(N_x):
        for j, k in ti.ndrange(N_y-1, N_z-1):
            eid = eid_base + i * (N_y - 1) * (N_z - 1) + j * (N_z - 1) + k
            edges[eid] = [ijk_2_index(i, j, k), ijk_2_index(i, j + 1, k + 1)]

@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))

@ti.kernel
def initialize():
    YoungsModulus[None] = 5e5
    # init position and velocity
    for i, j, k in ti.ndrange(N_x, N_y, N_z):
        index = ijk_2_index(i, j, k)
        x[index] = ti.Vector([init_x + i * dx, init_y + j * dx, init_z + k * dx])
        v[index] = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def compute_D(i):
    q = tetrahedrons[i][0]
    w = tetrahedrons[i][1]
    e = tetrahedrons[i][2]
    r = tetrahedrons[i][3]
    return ti.Matrix.cols([x[q] - x[r], x[w] - x[r], x[e] - x[r]])


@ti.kernel
def initialize_elements():
    for i in range(5):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/6
    # initialize dD
    for i, j in ti.ndrange(3, 3):
        for n in range(3):
            for m in range(3):
                dD[i, j][n, m] = 0
        dD[i, j][j, i] = 1
    for dim in range(3):
        dD[3, dim] = -(dD[0, dim] + dD[1, dim] + dD[2, dim])
    # initialize dF
    for k in range(5):
        for i in range(4):
            for j in range(3):
                dF[k, i, j] = dD[i, j] @ elements_Dm_inv[k]


# ----------------------core-----------------------------
@ti.func
def compute_F(i):
    return compute_D(i) @ elements_Dm_inv[i % 5]


@ti.func
def compute_P(i):
    F = compute_F(i)
    F_T = F.inverse().transpose()
    J = max(F.determinant(), 0.01)
    return LameMu[None] * (F - F_T) + LameLa[None] * ti.log(J) * F_T


@ti.func
def compute_Psi(i):
    F = compute_F(i)
    J = max(F.determinant(), 0.01)
    return LameMu[None] / 2 * ((F @ F.transpose()).trace() - 3) - LameMu[None] * ti.log(J) + LameLa[None] / 2 * ti.log(J)**2


@ti.kernel
def compute_force():
    for i in range(N):
        f[i] = ti.Vector([0, -gravity * mass, 0])

    for i in range(N_tetrahedron):
        loc_id = i % 5
        P = compute_P(i)
        H = - elements_V0[loc_id] * (P @ elements_Dm_inv[loc_id].transpose())

        h1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
        h2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
        h3 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])

        q = tetrahedrons[i][0]
        w = tetrahedrons[i][1]
        e = tetrahedrons[i][2]
        r = tetrahedrons[i][3]

        f[q] += h1
        f[w] += h2
        f[e] += h3
        f[r] += -(h1 + h2 + h3)


@ti.func
def compute_K(K_tri: ti.linalg.sparse_matrix_builder()):
    for k in range(N_tetrahedron):
        loc_id = k % 5
        # clear dP
        for i in range(4):
            for j in range(3):
                for n in ti.static(range(3)):
                    for m in ti.static(range(3)):
                        dP[k, i, j][n, m] = 0

        F = compute_F(k)
        F_1 = F.inverse()
        F_1_T = F_1.transpose()
        J = max(F.determinant(), 0.01)

        for i in range(4):
            for j in range(3):
                for n in ti.static(range(3)):
                    for m in ti.static(range(3)):
                        # dF/dF_{ij}
                        dFdFij = ti.Matrix([[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]])
                        dFdFij[n, m] = 1

                        # dF^T/dF_{ij}
                        dF_TdFij = dFdFij.transpose()

                        # Tr( F^{-1} dF/dF_{ij} )
                        dTr = F_1_T[n, m]

                        dP_dFij = LameMu[None] * dFdFij \
                                  + (LameMu[None] - LameLa[None] * ti.log(J)) * F_1_T @ dF_TdFij @ F_1_T \
                                  + LameLa[None] * dTr * F_1_T
                        dFij_ndim = dF[k, i, j][n, m]

                        dP[k, i, j] += dP_dFij * dFij_ndim

        for i in range(4):
            for j in range(3):
                dH[k, i, j] = -elements_V0[loc_id] * dP[k, i, j] @ elements_Dm_inv[loc_id].transpose()

        for i in ti.static(range(4)):
            n_idx = tetrahedrons[k][i]
            for j in ti.static(range(3)):
                # c_idx: the index of the specific component of n_idx-th node in all node nodes
                # or we can say: the index of the specific component of i-th node in this tetrahedron
                c_idx = n_idx * 3 + j
                for n in ti.static(range(3)):
                    # df_{nx}/dx_{ij}
                    K_tri[tetrahedrons[k][n] * 3 + 0, c_idx] += dH[k, i, j][0, n]
                    # df_{ny}/dx_{ij}
                    K_tri[tetrahedrons[k][n] * 3 + 1, c_idx] += dH[k, i, j][1, n]
                    # df_{nz}/dx_{ij}
                    K_tri[tetrahedrons[k][n] * 3 + 2, c_idx] += dH[k, i, j][2, n]

                # df_{3x}/dx_{ij}
                K_tri[tetrahedrons[k][3] * 3 + 0, c_idx] += -(dH[k, i, j][0, 0] + dH[k, i, j][0, 1] + dH[k, i, j][0, 2])
                # df_{3y}/dx_{ij}
                K_tri[tetrahedrons[k][3] * 3 + 1, c_idx] += -(dH[k, i, j][1, 0] + dH[k, i, j][1, 1] + dH[k, i, j][1, 2])
                # df_{3y}/dx_{ij}
                K_tri[tetrahedrons[k][3] * 3 + 2, c_idx] += -(dH[k, i, j][2, 0] + dH[k, i, j][2, 1] + dH[k, i, j][2, 2])


@ti.kernel
def solve_linear_system(A_tri: ti.linalg.sparse_matrix_builder()):
    dh_inv = 1 / dh
    # 用numpy构建动态二维数组，记录triplets的位置
