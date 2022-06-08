import taichi as ti
from scipy.sparse import linalg
import numpy as np
import math

ti.init(arch=ti.cpu)

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
        self.N_edges = (self.N_x - 1) * self.N_y * self.N_z + (self.N_y - 1) * self.N_x * self.N_z + (self.N_z - 1) * self.N_x * self.N_y \
                  + (self.N_x - 1) * (self.N_y - 1) * self.N_z + (self.N_x - 1) * (self.N_z - 1) * self.N_y + (self.N_y - 1) * (self.N_z - 1) * self.N_x
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

        # time-step size (for simulation, 16.7ms)
        self.h = 0.2
        # sub-step
        self.N_substeps = 100
        # time-step size (for time integration)
        self.dh = self.h / self.N_substeps
        self.dh_inv = 1 / self.dh
        self.num_of_iterate = 1

        # vectors and matrixs
        self.x = ti.Vector.field(3, ti.f32, self.N)
        self.x_new = ti.Vector.field(3, ti.f32, self.N)
        self.v = ti.Vector.field(3, ti.f32, self.N)
        self.v_new = ti.field(ti.f32, shape=3*self.N)
        # elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, N_tetrahedron)
        # elements_V0 = ti.field(ti.f32, N_tetrahedron)
        self.elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, 5)
        self.elements_V0 = ti.field(ti.f32, 5)
        self.f = ti.Vector.field(3, ti.f32, self.N)

        # geometric components
        self.tetrahedrons = ti.Vector.field(4, ti.i32, self.N_tetrahedron)
        # self.edges = ti.Vector.field(2, ti.i32, self.N_edges)
        self.faces = ti.field(ti.i32, self.N_faces * 3)

        # derivatives
        self.dD = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(4, 3))
        self.dF = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(5, 4, 3))
        self.dP = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.N_tetrahedron, 4, 3))
        self.dH = ti.Matrix.field(3, 3, dtype=ti.f32, shape=(self.N_tetrahedron, 4, 3))

        # for solving system of linear equations
        # self.activate_coor = []
        self.K_builder = ti.linalg.SparseMatrixBuilder(3 * self.N, 3 * self.N, max_num_triplets=9 * self.N * self.N)
        # self.A_builder = ti.linalg.SparseMatrixBuilder(3 * self.N, 3 * self.N, max_num_triplets=9 * self.N * self.N)
        self.M_builder = ti.linalg.SparseMatrixBuilder(3 * self.N, 3 * self.N, max_num_triplets=9 * self.N * self.N)
        self.initialize_M(self.M_builder)
        self.M = self.M_builder.build()
        b = ti.field(ti.f32, shape=3 * self.N)
        x = ti.field(ti.f32, shape=3 * self.N)

        self.meshing()
        print(self.faces)
        exit()
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

        for i in range(1):
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
        self.LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
        self.LameMu[None] = E / (2*(1+nu))

    @ti.kernel
    def initialize(self):
        self.YoungsModulus[None] = 1e4
        self.PoissonsRatio[None] = 0
        # init position and velocity
        for i, j, k in ti.ndrange(self.N_x, self.N_y, self.N_z):
            index = self.ijk_2_index(i, j, k)
            self.x[index] = ti.Vector([self.init_x + i * self.dx, self.init_y + j * self.dx, self.init_z + k * self.dx])
            self.x_new[index] = ti.Vector([self.init_x + i * self.dx, self.init_y + j * self.dx, self.init_z + k * self.dx])
            self.v[index] = ti.Vector([0.0, -3.0, 0.0])
            self.v_new[index + 0] = 0.0
            self.v_new[index + 1] = 0.0
            self.v_new[index + 2] = 0.0

    @ti.func
    def compute_D(self, i):
        q = self.tetrahedrons[i][0]
        w = self.tetrahedrons[i][1]
        e = self.tetrahedrons[i][2]
        r = self.tetrahedrons[i][3]
        return ti.Matrix.cols([self.x_new[q] - self.x_new[r], self.x_new[w] - self.x_new[r], self.x_new[e] - self.x_new[r]])

    @ti.kernel
    def initialize_elements(self):
        for i in range(5):
            Dm = self.compute_D(i)
            self.elements_Dm_inv[i] = Dm.inverse()
            self.elements_V0[i] = ti.abs(Dm.determinant())/6
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

    @ti.kernel
    def initialize_M(self, M_tri: ti.linalg.sparse_matrix_builder()):
        for i in range(3*self.N):
            M_tri[i, i] += self.mass

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
        return self.LameMu[None] / 2 * ((F.transpose() @ F).trace() - 3) \
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

    @ti.kernel
    def compute_K(self, K_tri: ti.linalg.sparse_matrix_builder()):
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
                    dF = self.dF[loc_id, i, j]
                    self.dP[k, i, j] = self.LameMu[None] * dF \
                                       + (self.LameMu[None] - self.LameLa[None] * ti.log(J)) * F_1_T @ dF.transpose() @ F_1_T \
                                       + self.LameLa[None] * (F_1_T @ dF).trace() * F_1_T

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
                        # df_{ny}/dx_{ij}
                        K_tri[self.tetrahedrons[k][n] * 3 + 1, c_idx] += self.dH[k, i, j][1, n]
                        # df_{nz}/dx_{ij}
                        K_tri[self.tetrahedrons[k][n] * 3 + 2, c_idx] += self.dH[k, i, j][2, n]

                    # df_{3x}/dx_{ij}
                    K_tri[self.tetrahedrons[k][3] * 3 + 0, c_idx] += -(self.dH[k, i, j][0, 0]
                                                                       + self.dH[k, i, j][0, 1]
                                                                       + self.dH[k, i, j][0, 2])
                    # df_{3y}/dx_{ij}
                    K_tri[self.tetrahedrons[k][3] * 3 + 1, c_idx] += -(self.dH[k, i, j][1, 0]
                                                                       + self.dH[k, i, j][1, 1]
                                                                       + self.dH[k, i, j][1, 2])
                    # df_{3y}/dx_{ij}
                    K_tri[self.tetrahedrons[k][3] * 3 + 2, c_idx] += -(self.dH[k, i, j][2, 0]
                                                                       + self.dH[k, i, j][2, 1]
                                                                       + self.dH[k, i, j][2, 2])

    @ti.kernel
    def reset_iter_vec(self):
        for i in range(self.N):
            self.x_new[i] = self.x[i]
            self.v_new[3*i + 0] = self.v[i][0]
            self.v_new[3*i + 1] = self.v[i][1]
            self.v_new[3*i + 2] = self.v[i][2]

    @ti.kernel
    def update_iter_vec(self, dx: ti.types.ndarray()):
        for i in range(self.N):
            self.x_new[i] += ti.Vector([dx[3*i], dx[3*i+1], dx[3*i+2]])
            self.v_new[3 * i + 0] += dx[3 * i + 0] * self.dh_inv
            self.v_new[3 * i + 1] += dx[3 * i + 1] * self.dh_inv
            self.v_new[3 * i + 2] += dx[3 * i + 2] * self.dh_inv
            if self.x_new[i][1] < 0.1:
                self.x_new[i][1] = 0.1
                if self.v_new[3 * i + 1] < 0.0:
                    self.v_new[3 * i + 1] = 0.0

    @ti.kernel
    def updatePosVel(self):
        for i in range(self.N):
            self.x[i] = self.x_new[i]
            self.v[i] = ti.Vector([self.v_new[3*i+0], self.v_new[3*i+1], self.v_new[3*i+2]])
            # if self.x[i][1] < 0.1:
            #     self.x[i][1] = 0.1
            #     self.v[i][1] = -self.v[i][1] * 0.8

    def update(self):
        dh2_inv = self.dh_inv ** 2
        self.reset_iter_vec()
        velocity = self.v.to_numpy().reshape(3*self.N)
        for it in range(self.num_of_iterate):
            # build K
            self.compute_K(self.K_builder)
            K = self.K_builder.build()
            # assemble A
            A = (1 + self.LameLa[None] * self.dh_inv) * K + dh2_inv * self.M
            # assemble b
            velocity_new = self.v_new.to_numpy()
            f_d = -self.LameLa[None] * K @ velocity_new
            self.compute_elastic_force()
            f_e = self.f.to_numpy().reshape(3*self.N)
            b = self.dh_inv * self.M @ (velocity - velocity_new) + f_e + f_d

            # solve the linear system
            solver = ti.linalg.SparseSolver(solver_type="LDLT")
            solver.analyze_pattern(A)
            solver.factorize(A)
            # Solve the linear system
            dx = solver.solve(b)
            self.update_iter_vec(dx)

        self.updatePosVel()


cube = Object()

window = ti.ui.Window("FEM Simulation", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
canvas.set_background_color((0.2, 0.2, 0.3))

wait = input("PRESS ENTER TO CONTINUE.")
while window.running:
    for frame in range(30):
        cube.update()

    camera.position(0.5, 0.5, 2)
    camera.lookat(0.5, 0.5, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.particles(cube.x, radius=0.002, color=(0.8, 0.8, 0.8))
    scene.mesh(cube.x, cube.faces, color=(0.5, 0.5, 0.5))
    canvas.scene(scene)
    window.show()
