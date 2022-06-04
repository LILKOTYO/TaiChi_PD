import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

# simulation components
scalar = lambda: ti.field(dtype=ti.f32)
vec = lambda: ti.Vector.field(3, dtype=ti.f32)
mac3x3 = lambda: ti.Matrix.field(3, 3, dytpe=ti.f32)

@ti.data_oriented
class Object:
    def __init__(self):
        # initialize settings
        self.init_x = 0.3
        self.init_y = 0.3
        self.init_z = 0.3
        self.N_x = 10
        self.N_y = 10
        self.N_z = 10
        self.N = self.N_x * self.N_y * self.N_z
        # axis-x + axis-y + axis-z + diagonal_xy + diagonal_xz + diagonal_yz
        self.N_edges = (self.N_x - 1) * self.N_y * self.N_z + (self.N_y - 1) * self.N_x * self.N_z + (self.N_z - 1) * self.N_x * self.N_y \
                  + (self.N_x - 1) * (self.N_y - 1) * self.N_z + (self.N_x - 1) * (self.N_z - 1) * self.N_y + (self.N_y - 1) * (self.N_z - 1) * self.N_x
        self.N_tetrahedron = 5 * (self.N_x - 1) * (self.N_y - 1) * (self.N_z - 1)
        self.dx = 0.5 / self.N_x

        # physical quantities
        self.mass = 1
        self.gravity = 0.5
        self.YoungsModulus = ti.field(ti.f32, ())
        self.PoissonsRatio = ti.field(ti.f32, ())
        self.LameMu = ti.field(ti.f32, ())
        self.LameLa = ti.field(ti.f32, ())

        # time-step size (for simulation, 16.7ms)
        self.h = 16.7e-3
        # sub-step
        self.N_substeps = 100
        # time-step size (for time integration)
        self.dh = self.h / self.N_substeps

        # vectors and matrixs
        self.x = ti.Vector.field(3, ti.f32, self.N)
        self.v = ti.Vector.field(3, ti.f32, self.N)
        self.v_new = ti.ti.Vector.field(3, ti.f32, self.N)
        # elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, N_tetrahedron)
        # elements_V0 = ti.field(ti.f32, N_tetrahedron)
        self.elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, 5)
        self.elements_V0 = ti.field(ti.f32, 5)
        self.f = ti.Vector.field(3, ti.f32, self.N)

        # geometric components
        self.tetrahedrons = ti.Vector.field(4, ti.i32, self.N_tetrahedron)
        self.edges = ti.Vector.field(2, ti.i32, self.N_edges)

        # derivatives
        self.dD = mac3x3()
        self.dF = mac3x3()
        self.dP = mac3x3()
        self.dH = mac3x3()

        # allocate_tensors
        # dD[i][j]: the difference caused by the change of j-th component in the i-th node
        ti.root.dense(ti.i, 4).dense(ti.j, 3).place(self.dD)
        # dF[k][i][j]: the difference of the k-th tetrahedron's F in a cube
        # caused by the change of j-th component in the i-th node
        ti.root.dense(ti.i, 5).dense(ti.i, 4).dense(ti.j, 3).place(self.dF)
        # dP[e][i][j]: the difference of the e-th tetrahedron's P
        # cause by the change of the j-th component in the i-th node
        ti.root.dense(ti.i, self.N_tetrahedron).dense(ti.i, 4).dense(ti.j, 3).place(self.dP, self.dH)

        # for solving system of linear equations
        # self.activate_coor = []
        self.K_builder = ti.linalg.SparseMatrixBuilder(3 * self.N, 3 * self.N, max_num_triplets=9 * self.N * self.N)
        self.A_builder = ti.linalg.SparseMatrixBuilder(3 * self.N, 3 * self.N, max_num_triplets=9 * self.N * self.N)
        b = ti.field(ti.f32, shape=3 * self.N)
        x = ti.field(ti.f32, shape=3 * self.N)

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

        # setting up edges
        # edge id
        eid_base = 0

        # axis-x edges
        for i in range(self.N_x - 1):
            for j, k in ti.ndrange(self.N_y, self.N_z):
                eid = eid_base + k * (self.N_x - 1) * self.N_y + j * (self.N_x - 1) + i
                self.edges[eid] = [self.ijk_2_index(i, j, k), self.ijk_2_index(i + 1, j, k)]

        eid_base += (self.N_x - 1) * self.N_y * self.N_z
        # axis-y edges
        for j in range(self.N_y - 1):
            for i, k in ti.ndrange(self.N_x, self.N_z):
                eid = eid_base + k * (self.N_y - 1) * self.N_x + i * (self.N_y - 1) + j
                self.edges[eid] = [self.ijk_2_index(i, j, k), self.ijk_2_index(i, j + 1, k)]

        eid_base += self.N_x * (self.N_y - 1) * self.N_z
        # axis-z edges
        for k in range(self.N_z - 1):
            for i, j in ti.ndrange(self.N_x, self.N_y):
                eid = eid_base + i * (self.N_z - 1) * self.N_y + j * (self.N_z - 1) + k
                self.edges[eid] = [self.ijk_2_index(i, j, k), self.ijk_2_index(i, j, k + 1)]

        eid_base += self.N_x * self.N_y * (self.N_z - 1)
        # diagonal_xy
        for k in range(self.N_z):
            for i, j in ti.ndrange(self.N_x - 1, self.N_y - 1):
                eid = eid_base + k * (self.N_x - 1) * (self.N_y - 1) + j * (self.N_x - 1) + i
                self.edges[eid] = [self.ijk_2_index(i, j, k), self.ijk_2_index(i + 1, j + 1, k)]

        eid_base += (self.N_x - 1) * (self.N_y - 1) * self.N_z
        # diagonal_xz
        for j in range(self.N_y):
            for i, k in ti.ndrange(self.N_x - 1, self.N_z - 1):
                eid = eid_base + j * (self.N_x - 1) * (self.N_z - 1) + k * (self.N_x - 1) + i
                self.edges[eid] = [self.ijk_2_index(i, j, k), self.ijk_2_index(i + 1, j, k + 1)]

        eid_base += (self.N_x - 1) * self.N_y * (self.N_z - 1)
        # diagonal_yz
        for i in range(self.N_x):
            for j, k in ti.ndrange(self.N_y - 1, self.N_z - 1):
                eid = eid_base + i * (self.N_y - 1) * (self.N_z - 1) + j * (self.N_z - 1) + k
                self.edges[eid] = [self.ijk_2_index(i, j, k), self.ijk_2_index(i, j + 1, k + 1)]

    @ti.kernel
    def updateLameCoeff(self):
        E = self.YoungsModulus[None]
        nu = self.PoissonsRatio[None]
        self.LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
        self.LameMu[None] = E / (2*(1+nu))

    @ti.kernel
    def initialize(self):
        self.YoungsModulus[None] = 5e5
        # init position and velocity
        for i, j, k in ti.ndrange(self.N_x, self.N_y, self.N_z):
            index = self.ijk_2_index(i, j, k)
            self.x[index] = ti.Vector([self.init_x + i * self.dx, self.init_y + j * self.dx, self.init_z + k * self.dx])
            self.v[index] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def compute_D(self, i):
        q = self.tetrahedrons[i][0]
        w = self.tetrahedrons[i][1]
        e = self.tetrahedrons[i][2]
        r = self.tetrahedrons[i][3]
        return ti.Matrix.cols([self.x[q] - self.x[r], self.x[w] - self.x[r], self.x[e] - self.x[r]])


    @ti.kernel
    def initialize_elements(self):
        for i in range(5):
            Dm = self.compute_D(i)
            self.elements_Dm_inv[i] = Dm.inverse()
            self.elements_V0[i] = ti.abs(Dm.determinant())/6
        # initialize dD
        for i, j in ti.ndrange(3, 3):
            for n in range(3):
                for m in range(3):
                    self.dD[i, j][n, m] = 0
            self.dD[i, j][j, i] = 1
        for dim in range(3):
            self.dD[3, dim] = -(self.dD[0, dim] + self.dD[1, dim] + self.dD[2, dim])
        # initialize dF
        for k in range(5):
            for i in range(4):
                for j in range(3):
                    self.dF[k, i, j] = self.dD[i, j] @ self.elements_Dm_inv[k]


# ----------------------core-----------------------------
    @ti.func
    def compute_F(self, i):
        return self.compute_D(i) @ self.elements_Dm_inv[i % 5]


    @ti.func
    def compute_P(self, i):
        F = self.compute_F(i)
        F_T = F.inverse().transpose()
        J = max(F.determinant(), 0.01)
        return self.LameMu[None] * (F - F_T) + self.LameLa[None] * ti.log(J) * F_T


    @ti.func
    def compute_Psi(self, i):
        F = self.compute_F(i)
        J = max(F.determinant(), 0.01)
        return self.LameMu[None] / 2 * ((F @ F.transpose()).trace() - 3) \
               - self.LameMu[None] * ti.log(J) \
               + self.LameLa[None] / 2 * ti.log(J)**2


    @ti.kernel
    def compute_elastic_force(self):
        for i in range(self.N):
            self.f[i] = ti.Vector([0, -self.gravity * self.mass, 0])

        for i in range(self.N_tetrahedron):
            loc_id = i % 5
            P = self.compute_P(i)
            H = - self.elements_V0[loc_id] * (P @ self.elements_Dm_inv[loc_id].transpose())

            h1 = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
            h2 = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
            h3 = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])

            q = self.tetrahedrons[i][0]
            w = self.tetrahedrons[i][1]
            e = self.tetrahedrons[i][2]
            r = self.tetrahedrons[i][3]

            self.f[q] += h1
            self.f[w] += h2
            self.f[e] += h3
            self.f[r] += -(h1 + h2 + h3)


    @ti.func
    def compute_K(self, K_tri: ti.linalg.sparse_matrix_builder(), A_tri: ti.linalg.sparse_matrix_builder()):
        param_A = 1 + self.LameLa[None] / self.dh
        for k in range(self.N_tetrahedron):
            loc_id = k % 5
            # clear dP
            for i in range(4):
                for j in range(3):
                    for n in ti.static(range(3)):
                        for m in ti.static(range(3)):
                            self.dP[k, i, j][n, m] = 0

            F = self.compute_F(k)
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

                            dP_dFij = self.LameMu[None] * dFdFij \
                                      + (self.LameMu[None] - self.LameLa[None] * ti.log(J)) * F_1_T @ dF_TdFij @ F_1_T \
                                      + self.LameLa[None] * dTr * F_1_T
                            dFij_ndim = self.dF[k, i, j][n, m]

                            self.dP[k, i, j] += dP_dFij * dFij_ndim

            for i in range(4):
                for j in range(3):
                    self.dH[k, i, j] = -self.elements_V0[loc_id] * self.dP[k, i, j] @ self.elements_Dm_inv[loc_id].transpose()

            for i in ti.static(range(4)):
                n_idx = self.tetrahedrons[k][i]
                for j in ti.static(range(3)):
                    # c_idx: the index of the specific component of n_idx-th node in all node nodes
                    # or we can say: the index of the specific component of i-th node in this tetrahedron
                    c_idx = n_idx * 3 + j
                    for n in ti.static(range(3)):
                        # df_{nx}/dx_{ij}
                        K_tri[self.tetrahedrons[k][n] * 3 + 0, c_idx] += self.dH[k, i, j][0, n]
                        A_tri[self.tetrahedrons[k][n] * 3 + 0, c_idx] += self.dH[k, i, j][0, n] * param_A
                        # df_{ny}/dx_{ij}
                        K_tri[self.tetrahedrons[k][n] * 3 + 1, c_idx] += self.dH[k, i, j][1, n]
                        A_tri[self.tetrahedrons[k][n] * 3 + 1, c_idx] += self.dH[k, i, j][1, n] * param_A
                        # df_{nz}/dx_{ij}
                        K_tri[self.tetrahedrons[k][n] * 3 + 2, c_idx] += self.dH[k, i, j][2, n]
                        A_tri[self.tetrahedrons[k][n] * 3 + 2, c_idx] += self.dH[k, i, j][2, n] * param_A

                    # df_{3x}/dx_{ij}
                    K_tri[self.tetrahedrons[k][3] * 3 + 0, c_idx] += -(self.dH[k, i, j][0, 0]
                                                                       + self.dH[k, i, j][0, 1]
                                                                       + self.dH[k, i, j][0, 2])
                    A_tri[self.tetrahedrons[k][3] * 3 + 0, c_idx] += -(self.dH[k, i, j][0, 0]
                                                                       + self.dH[k, i, j][0, 1]
                                                                       + self.dH[k, i, j][0, 2]) * param_A
                    # df_{3y}/dx_{ij}
                    K_tri[self.tetrahedrons[k][3] * 3 + 1, c_idx] += -(self.dH[k, i, j][1, 0]
                                                                       + self.dH[k, i, j][1, 1]
                                                                       + self.dH[k, i, j][1, 2])
                    A_tri[self.tetrahedrons[k][3] * 3 + 1, c_idx] += -(self.dH[k, i, j][1, 0]
                                                                       + self.dH[k, i, j][1, 1]
                                                                       + self.dH[k, i, j][1, 2]) * param_A
                    # df_{3y}/dx_{ij}
                    K_tri[self.tetrahedrons[k][3] * 3 + 2, c_idx] += -(self.dH[k, i, j][2, 0]
                                                                       + self.dH[k, i, j][2, 1]
                                                                       + self.dH[k, i, j][2, 2])
                    A_tri[self.tetrahedrons[k][3] * 3 + 2, c_idx] += -(self.dH[k, i, j][2, 0]
                                                                       + self.dH[k, i, j][2, 1]
                                                                       + self.dH[k, i, j][2, 2]) * param_A

    @ti.kernel
    def assembly_linear_system(self, A_tri: ti.linalg.sparse_matrix_builder()):
        dh_inv = 1 / self.dh
        dh2_inv = dh_inv**2
        # assemble the system matrix A
        for i in range(3*self.N):
            A_tri[i, i] += dh2_inv * self.mass

        # assemble the unknown vector b

