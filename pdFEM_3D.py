import taichi as ti
from scipy.sparse import dia_matrix, csc_matrix, linalg
import numpy as np

ti.init(arch=ti.gpu)


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
        self.N_tetrahedron = 5 * (self.N_x - 1) * (self.N_y - 1) * (self.N_z - 1)
        self.N_faces = 4 * (self.N_x - 1) * (self.N_y - 1) \
                       + 4 * (self.N_x - 1) * (self.N_z - 1) \
                       + 4 * (self.N_y - 1) * (self.N_z - 1)
        self.dx = 0.5 / self.N_x

        # physical quantities
        self.mass = 1
        self.gravity = 9.8
        self.jacobi_iter = 50
        self.jacobi_alpha = 0.1
        self.stiffness = 5000

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
        self.f_ext = ti.Vector.field(3, ti.f32, self.N)

        # geometric components
        self.tetrahedrons = ti.Vector.field(4, ti.i32, self.N_tetrahedron)
        self.faces = ti.field(ti.i32, self.N_faces * 3)

        # for linear system
        self.M = self.initialize_M()
        self.sig = ti.Vector.field(3, ti.f32, self.N_tetrahedron)
        data = np.ones(3 * self.N)
        offset = np.array([0])
        self.I = dia_matrix((data, offset), shape=(3 * self.N, 3 * self.N))

        self.meshing()
        self.initialize()
        self.GcT, self.sum_GcTGc = self.initialize_Gc()
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
    def initialize(self):
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

    def initialize_M(self):
        data = np.ones(3*self.N)
        offset = np.array([0])
        return dia_matrix((data, offset), shape=(3*self.N, 3*self.N))

    def initialize_Gc(self):
        dim = 3 * self.N
        data = np.ones(12)
        GcT = []
        sum_GcTGc = csc_matrix((dim, dim))
        for i in range(self.N_tetrahedron):
            row = np.arange(12)
            col = []
            for n in range(4):
                ind = self.tetrahedrons[i][n]
                col.append(3 * ind + 0)
                col.append(3 * ind + 1)
                col.append(3 * ind + 2)
            col_nd = np.array(col)
            Gc_i = csc_matrix((data, (row, col_nd)), shape=(12, dim))
            GcT_i = Gc_i.transpose()
            GcTGc_i = GcT_i @ Gc_i
            GcT.append(GcT_i)
            sum_GcTGc = sum_GcTGc + GcTGc_i
        return GcT, sum_GcTGc

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
            self.x_proj[i] = self.x[i]
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
            for n in ti.static(range(3)):
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
    def updatePosVel(self, x_star: ti.types.ndarray()):
        for i in range(self.N):
            x_new = ti.Vector([x_star[3*i+0], x_star[3*i+1], x_star[3*i+2]])
            self.v[i] = self.dh_inv * (x_new - self.x[i])
            self.x[i] = x_new
            if x_new[1] < 0.1:
                self.x[i][1] = 0.1
                if self.v[i][1] < 0:
                    self.v[i][1] = 0.0

    @ti.kernel
    def local_step(self):
        self.initialize_iter_vector()
        # Jacobi Solver
        for k in range(self.jacobi_iter):
            self.clear_iter_vector()
            self.jacobi()

    def global_step(self):
        dim = 3 * self.N
        dh2_inv = self.dh_inv**2
        dh = self.dh

        # A = dh2_inv * self.M + self.stiffness * self.I
        A = dh2_inv * self.M + self.stiffness * self.sum_GcTGc
        A = csc_matrix(A)
        xn = self.x.to_numpy().reshape(dim)
        vn = self.v.to_numpy().reshape(dim)
        f_ext = self.f_ext.to_numpy().reshape(dim)
        sn = xn + dh * vn + (dh**2) * linalg.inv(self.M) @ f_ext
        p = self.x_proj.to_numpy().reshape(dim)
        b = dh2_inv * self.M @ sn + self.stiffness * self.sum_GcTGc @ p
        # b = dh2_inv * self.M @ sn + self.stiffness * p
        # b = csc_matrix((b, (np.arange(dim), np.zeros(dim))), shape=(dim, 1))
        x_star, info = linalg.cg(A, b, x0=xn)

        return x_star

    def update(self):
        self.local_step()
        x_star = self.global_step()
        self.updatePosVel(x_star)


cube = Object()

window = ti.ui.Window("FEM Simulation", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
canvas.set_background_color((0.2, 0.2, 0.3))
wait = input("PRESS ENTER TO CONTINUE.")
while window.running:
    cube.update()

    camera.position(0.5, 0.5, 2)
    camera.lookat(0.5, 0.5, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.particles(cube.x, radius=0.002, color=(0.8, 0.8, 0.8))
    scene.mesh(cube.x, cube.faces, color=(0.5, 0.5, 0.5))
    canvas.scene(scene)
    window.show()