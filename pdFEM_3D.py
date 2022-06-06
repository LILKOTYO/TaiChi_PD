import taichi as ti
from scipy.sparse import dia_matrix, csr_matrix, linalg
import numpy as np
import math

ti.init(arch=ti.gpu)

# simulation components
scalar = lambda: ti.field(dtype=ti.f32)
vec = lambda: ti.Vector.field(3, dtype=ti.f32)
mac3x3 = lambda: ti.Matrix.field(3, 3, dtype=ti.f32)

@ti.data_oriented
class Object:
    def __init__(self):
        # initialize settings
        self.init_x = 0.3
        self.init_y = 0.3
        self.init_z = 0.3
        self.N_x = 3
        self.N_y = 3
        self.N_z = 3
        self.N = self.N_x * self.N_y * self.N_z
        # axis-x + axis-y + axis-z + diagonal_xy + diagonal_xz + diagonal_yz
        self.N_edges = (self.N_x - 1) * self.N_y * self.N_z + (self.N_y - 1) * self.N_x * self.N_z + (
                    self.N_z - 1) * self.N_x * self.N_y \
                       + (self.N_x - 1) * (self.N_y - 1) * self.N_z + (self.N_x - 1) * (self.N_z - 1) * self.N_y + (
                                   self.N_y - 1) * (self.N_z - 1) * self.N_x
        self.N_tetrahedron = 5 * (self.N_x - 1) * (self.N_y - 1) * (self.N_z - 1)
        self.N_faces = 4 * (self.N_x - 1) * (self.N_y - 1) \
                       + 4 * (self.N_x - 1) * (self.N_z - 1) \
                       + 4 * (self.N_y - 1) * (self.N_z - 1)
        self.dx = 0.5 / self.N_x

        # physical quantities
        self.mass = 1
        self.gravity = 9.8
        self.YoungsModulus = ti.field(ti.f32, ())
        self.PoissonsRatio = ti.field(ti.f32, ())
        self.LameMu = ti.field(ti.f32, ())
        self.LameLa = ti.field(ti.f32, ())
        self.jacobi_iter = 50
        self.jacobi_alpha = 0.1

        # time-step size (for simulation, 16.7ms)
        self.h = 0.2
        # sub-step
        self.N_substeps = 100
        # time-step size (for time integration)
        self.dh = self.h / self.N_substeps
        self.dh_inv = 1 / self.dh
        self.num_of_iterate = 50

        # vectors and matrices
        self.x = ti.Vector.field(3, ti.f32, self.N)
        self.x_proj = ti.Vector.field(3, ti.f32, self.N)
        self.x_iter = ti.Vector.field(3, ti.f32, self.N)
        self.count = ti.field(ti.i32, self.N)
        self.v = ti.Vector.field(3, ti.f32, self.N)
        self.elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, 5)
        self.elements_Dm = ti.Matrix.field(3, 3, ti.f32, 5)
        self.elements_V0 = ti.field(ti.f32, 5)
        self.f_ext = ti.Vector.field(3, ti.f32, self.N)

        # derivatives
        self.dD = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(4, 3))
        self.dF = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(5, 4, 3))
        self.dP = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.N_tetrahedron, 4, 3))
        self.dH = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.N_tetrahedron, 4, 3))

        # geometric components
        self.tetrahedrons = ti.Vector.field(4, ti.i32, self.N_tetrahedron)
        self.faces = ti.field(ti.i32, self.N_faces * 3)

        # for linear system
        self.M = self.initialize_M()
        self.sig = ti.Vector.field(3, ti.f32, self.N_tetrahedron)

        self.meshing()
        self.initialize()
        self.updateLameCoeff()
        self.initialize_elements()

    @ti.func
    def ijk_2_index(self, i, j, k):
        return k * self.N_x * self.N_y + j * self.N_x + i

    # -----------------------meshing and init----------------------------
    @ti.kernel
    def meshing(self):
        # setting up tetrahedrons
        for i, j, k in ti.ndrange(self.N_x - 1, self.N_y - 1, self.N_z - 1):
            # tetrahedron id
            tid = 5 * (k * (self.N_x - 1) * (self.N_y - 1) + j * (self.N_x - 1) + i)
            self.tetrahedrons[tid][0] = self.ijk_2_index(i, j, k + 1)
            self.tetrahedrons[tid][1] = self.ijk_2_index(i + 1, j, k + 1)
            self.tetrahedrons[tid][2] = self.ijk_2_index(i + 1, j, k)
            self.tetrahedrons[tid][3] = self.ijk_2_index(i + 1, j + 1, k + 1)

            tid += 1
            self.tetrahedrons[tid][0] = self.ijk_2_index(i, j, k + 1)
            self.tetrahedrons[tid][1] = self.ijk_2_index(i, j, k)
            self.tetrahedrons[tid][2] = self.ijk_2_index(i + 1, j, k)
            self.tetrahedrons[tid][3] = self.ijk_2_index(i, j + 1, k)

            tid += 1
            self.tetrahedrons[tid][0] = self.ijk_2_index(i, j, k + 1)
            self.tetrahedrons[tid][1] = self.ijk_2_index(i, j + 1, k + 1)
            self.tetrahedrons[tid][2] = self.ijk_2_index(i, j + 1, k)
            self.tetrahedrons[tid][3] = self.ijk_2_index(i + 1, j + 1, k + 1)

            tid += 1
            self.tetrahedrons[tid][0] = self.ijk_2_index(i + 1, j, k)
            self.tetrahedrons[tid][1] = self.ijk_2_index(i + 1, j + 1, k)
            self.tetrahedrons[tid][2] = self.ijk_2_index(i + 1, j + 1, k + 1)
            self.tetrahedrons[tid][3] = self.ijk_2_index(i, j + 1, k)

            tid += 1
            self.tetrahedrons[tid][0] = self.ijk_2_index(i, j, k + 1)
            self.tetrahedrons[tid][1] = self.ijk_2_index(i + 1, j, k)
            self.tetrahedrons[tid][2] = self.ijk_2_index(i + 1, j + 1, k + 1)
            self.tetrahedrons[tid][3] = self.ijk_2_index(i, j + 1, k)

        # init faces
        fid = 0
        for i, j in ti.ndrange(self.N_x - 1, self.N_y - 1):
            self.faces[fid + 0] = self.ijk_2_index(i, j, 0)
            self.faces[fid + 1] = self.ijk_2_index(i + 1, j, 0)
            self.faces[fid + 2] = self.ijk_2_index(i + 1, j + 1, 0)
            self.faces[fid + 3] = self.ijk_2_index(i, j, 0)
            self.faces[fid + 4] = self.ijk_2_index(i + 1, j + 1, 0)
            self.faces[fid + 5] = self.ijk_2_index(i, j + 1, 0)

            self.faces[fid + 6] = self.ijk_2_index(i, j, self.N_z - 1)
            self.faces[fid + 7] = self.ijk_2_index(i + 1, j, self.N_z - 1)
            self.faces[fid + 8] = self.ijk_2_index(i + 1, j + 1, self.N_z - 1)
            self.faces[fid + 9] = self.ijk_2_index(i, j, self.N_z - 1)
            self.faces[fid + 10] = self.ijk_2_index(i + 1, j + 1, self.N_z - 1)
            self.faces[fid + 11] = self.ijk_2_index(i, j + 1, self.N_z - 1)
            fid += 12

        for i, k in ti.ndrange(self.N_x - 1, self.N_z - 1):
            self.faces[fid + 0] = self.ijk_2_index(i, 0, k)
            self.faces[fid + 1] = self.ijk_2_index(i + 1, 0, k)
            self.faces[fid + 2] = self.ijk_2_index(i, 0, k + 1)
            self.faces[fid + 3] = self.ijk_2_index(i, 0, k + 1)
            self.faces[fid + 4] = self.ijk_2_index(i + 1, 0, k)
            self.faces[fid + 5] = self.ijk_2_index(i + 1, 0, k + 1)

            self.faces[fid + 6] = self.ijk_2_index(i, self.N_y - 1, k)
            self.faces[fid + 7] = self.ijk_2_index(i + 1, self.N_y - 1, k)
            self.faces[fid + 8] = self.ijk_2_index(i, self.N_y - 1, k + 1)
            self.faces[fid + 9] = self.ijk_2_index(i, self.N_y - 1, k + 1)
            self.faces[fid + 10] = self.ijk_2_index(i + 1, self.N_y - 1, k)
            self.faces[fid + 11] = self.ijk_2_index(i + 1, self.N_y - 1, k + 1)
            fid += 12

        for j, k in ti.ndrange(self.N_y - 1, self.N_z - 1):
            self.faces[fid + 0] = self.ijk_2_index(0, j, k)
            self.faces[fid + 1] = self.ijk_2_index(0, j, k + 1)
            self.faces[fid + 2] = self.ijk_2_index(0, j + 1, k)
            self.faces[fid + 3] = self.ijk_2_index(0, j + 1, k)
            self.faces[fid + 4] = self.ijk_2_index(0, j, k + 1)
            self.faces[fid + 5] = self.ijk_2_index(0, j + 1, k + 1)

            self.faces[fid + 6] = self.ijk_2_index(self.N_x - 1, j, k)
            self.faces[fid + 7] = self.ijk_2_index(self.N_x - 1, j, k + 1)
            self.faces[fid + 8] = self.ijk_2_index(self.N_x - 1, j + 1, k)
            self.faces[fid + 9] = self.ijk_2_index(self.N_x - 1, j + 1, k)
            self.faces[fid + 10] = self.ijk_2_index(self.N_x - 1, j, k + 1)
            self.faces[fid + 11] = self.ijk_2_index(self.N_x - 1, j + 1, k + 1)
            fid += 12

    @ti.kernel
    def updateLameCoeff(self):
        E = self.YoungsModulus[None]
        nu = self.PoissonsRatio[None]
        self.LameLa[None] = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.LameMu[None] = E / (2 * (1 + nu))

    @ti.kernel
    def initialize(self):
        self.YoungsModulus[None] = 1e4
        self.PoissonsRatio[None] = 0
        # init position and velocity
        for i, j, k in ti.ndrange(self.N_x, self.N_y, self.N_z):
            index = self.ijk_2_index(i, j, k)
            self.x[index] = ti.Vector([self.init_x + i * self.dx, self.init_y + j * self.dx, self.init_z + k * self.dx])
            self.x_proj[index] = ti.Vector([0.0, 0.0, 0.0])
            self.v[index] = ti.Vector([0.0, 0.0, 0.0])
            self.f_ext[index] = ti.Vector([0.0, -self.gravity * self.mass, 0.0])

    @ti.kernel
    def initialize_elements(self):
        for i in range(5):
            Dm = self.compute_D(i)
            self.elements_Dm[i] = Dm
            self.elements_Dm_inv[i] = Dm.inverse()
            self.elements_V0[i] = ti.abs(Dm.determinant()) / 6
        # initialize dD
        for i, j in ti.ndrange(4, 3):
            for n in ti.static(range(3)):
                for m in ti.static(range(3)):
                    self.dD[i, j][n, m] = 0

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                self.dD[i, j][j, i] = 1

        for dim in ti.static(range(3)):
            self.dD[3, dim] = -(self.dD[0, dim] + self.dD[1, dim] + self.dD[2, dim])
        # initialize dF
        for k in ti.static(range(5)):
            for i in ti.static(range(4)):
                for j in ti.static(range(3)):
                    self.dF[k, i, j] = self.dD[i, j] @ self.elements_Dm_inv[k]

    def initialize_M(self):
        data = np.ones(3*self.N)
        offset = np.array([0])
        return dia_matrix((data, offset), shape=(3*self.N, 3*self.N))

    @ti.func
    def compute_D(self, i):
        q = self.tetrahedrons[i][0]
        w = self.tetrahedrons[i][1]
        e = self.tetrahedrons[i][2]
        r = self.tetrahedrons[i][3]
        return ti.Matrix.cols(
            [self.x[q] - self.x[r], self.x[w] - self.x[r], self.x[e] - self.x[r]])

    @ti.func
    def compute_D_in_local(self, i):
        q = self.tetrahedrons[i][0]
        w = self.tetrahedrons[i][1]
        e = self.tetrahedrons[i][2]
        r = self.tetrahedrons[i][3]
        return ti.Matrix.cols(
            [self.x_proj[q] - self.x_proj[r], self.x_proj[w] - self.x_proj[r], self.x_proj[e] - self.x_proj[r]])

    @ti.func
    def compute_F(self, i):
        return self.compute_D_in_local(i) @ self.elements_Dm_inv[i % 5]

    @ti.func
    def initialize_iter_vector(self):
        for i in range(self.N):
            self.x_proj[i] = self.x
            self.x_iter[i] = ti.Vector([0.0, 0.0, 0.0])
            self.count[i] = 0

    @ti.func
    def clear_iter_vector(self):
        for i in range(self.N):
            self.x_iter[i] = ti.Vector([0.0, 0.0, 0.0])
            self.count[i] = 0

    @ti.func
    def jacobi(self):
        for it in range(self.N_tetrahedron):
            F = self.compute_F(it)
            U, S, V = ti.svd(F)
            S[0, 0] = min(max(0.95, S[0, 0]), 1.05)
            S[1, 1] = min(max(0.95, S[1, 1]), 1.05)
            S[2, 2] = min(max(0.95, S[2, 2]), 1.05)
            F_star = U @ S @ V.transpose()
            D_star = F_star @ self.elements_Dm[it%5]

            q = self.tetrahedrons[it][0]
            w = self.tetrahedrons[it][1]
            e = self.tetrahedrons[it][2]
            r = self.tetrahedrons[it][3]

            # find the center of gravity
            center = (self.x_proj[q] + self.x_proj[w] + self.x_proj[e] + self.x_proj[r]) / 4

            # find the projected vector
            for n in range(3):
                x3_new = center[n] - (D_star[n, 0] + D_star[n, 1] + D_star[n, 2]) / 4
                self.x_iter[r][n] += x3_new
                self.x_iter[q][n] += x3_new + D_star[n, 0]
                self.x_iter[w][n] += x3_new + D_star[n, 1]
                self.x_iter[e][n] += x3_new + D_star[n, 2]

            self.count[q] += 1
            self.count[w] += 1
            self.count[e] += 1
            self.count[r] += 1

        for i in range(self.N):
            self.x_proj[i] = (self.x_iter[i] + self.jacobi_alpha * self.x_proj[i]) / (self.count[i] + self.jacobi_alpha)

    @ti.kernel
    def local_step(self):
        self.initialize_iter_vector()
        # Jacobi Solver
        for k in self.jacobi_iter:
            self.clear_iter_vector()
            self.jacobi()

    @ti.kernel
    def global_step(self):


    def update(self):
        self.local_step()
