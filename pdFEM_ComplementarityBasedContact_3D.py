import taichi as ti
from scipy.sparse import lil_matrix, dia_matrix, csc_matrix, linalg
import numpy as np
import math

ti.init(arch=ti.gpu, debug=True)


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
        self.N_surfaces = self.N_x * self.N_y * 2 + (self.N_z - 2) * self.N_x * 2 + (self.N_z - 2) * (self.N_y - 2) * 2
        self.dx = 0.5 / self.N_x

        # physical quantities
        self.mass = 1
        self.gravity = 5.0
        self.jacobi_iter = 10
        self.jacobi_alpha = 0.1
        self.stiffness = 1000
        self.epsilon = 1e-20
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
        self.num_of_iterate = 50

        # vectors and matrices
        self.x = ti.Vector.field(3, ti.f32, self.N)
        self.x_proj = ti.Vector.field(3, ti.f32, self.N)
        self.x_iter = ti.Vector.field(3, ti.f32, self.N)
        self.x_old = ti.Vector.field(3, ti.f32, self.N)
        self.x_b = ti.Vector.field(3, ti.f32, self.N)
        self.count = ti.field(ti.i32, self.N)
        self.v = ti.Vector.field(3, ti.f32, self.N)
        self.elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, 5)
        self.elements_V0 = ti.field(ti.f32, 5)
        self.elements_Dm = ti.Matrix.field(3, 3, ti.f32, 5)
        self.f_ext = ti.Vector.field(3, ti.f32, self.N)
        self.f = ti.Vector.field(3, ti.f32, self.N)

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
        self.updateLameCoeff()
        self.GcT, self.sum_GcTGc = self.initialize_Gc()
        self.initialize_elements()
        self.surface_nodes_np = np.zeros(self.N_surfaces, dtype=int)
        self.initialize_surface(self.surface_nodes_np)
        self.surface_nodes = ti.field(ti.i32, self.N_surfaces)
        self.surface_nodes.from_numpy(self.surface_nodes_np)

        # precompute
        self.A = (self.dh_inv**2) * self.M + self.stiffness * self.sum_GcTGc
        self.A = csc_matrix(self.A)
        self.AinvIic = self.precompute()
        # np.set_printoptions(threshold=np.inf)

        # gripper
        self.gripper_left_pos = ti.Vector([0.0, 0.0, 0.0])
        self.gripper_right_pos = ti.Vector([1.0, 0.0, 0.0])
        self.gripper_left_normal = ti.Vector([math.sqrt(3.0), 1.0, 0.0]).normalized()
        self.gripper_right_normal = ti.Vector([-math.sqrt(3.0), 1.0, 0.0]).normalized()
        self.gripper_len = 1.0

        # floor
        self.floor_h = 0.1

        # contact
        self.C = np.array([-1 for i in range(self.N_surfaces)])
        self.obstacle = np.array([-1 for i in range(self.N_surfaces)]) # left = 1, right = 2, floor = 3

        # debug
        self.before_contact = True
        print(self.surface_nodes)

    @ti.func
    def ijk_2_index(self, i, j, k):
        return k * self.N_x * self.N_y + j * self.N_x + i

    def ijk_2_index_py(self, i, j, k):
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
            self.x_b[index] = ti.Vector([0.0, 0.0, 0.0])
            self.v[index] = ti.Vector([0.0, 0.0, 0.0])
            self.f_ext[index] = ti.Vector([0.0, -self.gravity * self.mass, 0.0])

    def initialize_surface(self, surface_nodes_np: ti.types.ndarray()):
        sid = 0
        for i in range(self.N_x):
            for j in range(self.N_y):
                surface_nodes_np[sid] = self.ijk_2_index_py(i, j, 0)
                sid += 1
                surface_nodes_np[sid] = self.ijk_2_index_py(i, j, self.N_z - 1)
                sid += 1
        for i in range(self.N_x):
            for k in range(1, self.N_z - 1):
                surface_nodes_np[sid] = self.ijk_2_index_py(i, 0, k)
                sid += 1
                surface_nodes_np[sid] = self.ijk_2_index_py(i, self.N_y - 1, k)
                sid += 1
        for j in range(1, self.N_y - 1):
            for k in range(1, self.N_z - 1):
                surface_nodes_np[sid] = self.ijk_2_index_py(0, j, k)
                sid += 1
                surface_nodes_np[sid] = self.ijk_2_index_py(self.N_x - 1, j, k)
                sid += 1
        surface_nodes_np.sort()

    @ti.kernel
    def initialize_elements(self):
        for i in range(5):
            Dm = self.compute_D(i)
            self.elements_Dm[i] = Dm
            self.elements_Dm_inv[i] = Dm.inverse()
            self.elements_V0[i] = ti.abs(Dm.determinant()) / 6

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

    def precompute(self):
        data = np.ones(self.N_surfaces*3)
        col = np.arange(self.N_surfaces*3)
        row = []
        for i in range(self.N_surfaces):
            ind = self.surface_nodes[i]
            row.append(3 * ind + 0)
            row.append(3 * ind + 1)
            row.append(3 * ind + 2)
        row_nd = np.array(row)
        Iic = csc_matrix((data, (row_nd, col)), shape=(3*self.N, 3*self.N_surfaces))
        AinvIic = linalg.inv(self.A) @ Iic
        return csc_matrix(AinvIic)

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
        return self.compute_D(i) @ self.elements_Dm_inv[i % 5]

    @ti.func
    def compute_F_in_local(self, i):
        return self.compute_D_in_local(i) @ self.elements_Dm_inv[i % 5]

    @ti.func
    def compute_P(self, i):
        F = self.compute_F(i)
        F_T = F.inverse().transpose()
        J = max(F.determinant(), 0.01)
        return self.LameMu[None] * (F - F_T) + self.LameLa[None] * ti.log(J) * F_T

    @ti.kernel
    def compute_elastic_force(self):
        for i in range(self.N):
            self.f[i] = ti.Vector([0, 0, 0])

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
    def initialize_iter_vector(self):
        for i in range(self.N):
            self.x_proj[i] = self.x[i]

    @ti.func
    def clear_iter_vector(self):
        for i in range(self.N):
            self.x_iter[i] = ti.Vector([0.0, 0.0, 0.0])
            self.count[i] = 0

    @ti.kernel
    def clear_C(self, C: ti.types.ndarray(), obstacle: ti.types.ndarray()):
        for i in range(self.N_surfaces):
            C[i] = -1

        for i in range(self.N):
            obstacle[i] = -1

    @ti.kernel
    def is_contact(self, C: ti.types.ndarray()) -> int:
        res = 0
        for i in range(self.N_surfaces):
            if C[i] != -1:
                res += 1
        return res

    @ti.kernel
    def contact_detect(self, C: ti.types.ndarray(), obstacle: ti.types.ndarray()):
        for i in range(self.N):
            self.x_b[i] = ti.Vector([0.0, 0.0, 0.0])
        for i in range(self.N_surfaces):
            ind = self.surface_nodes[i]
            old_pos = self.x[ind]
            new_pos = old_pos + self.dh * self.v[ind]
            flag_left = (new_pos - self.gripper_left_pos).dot(self.gripper_left_normal) <= 0 and (old_pos - self.gripper_left_pos).dot(self.gripper_left_normal) >= 0
            flag_right = (new_pos - self.gripper_right_pos).dot(self.gripper_right_normal) <= 0 and (old_pos - self.gripper_right_pos).dot(self.gripper_right_normal) >= 0
            flag_floor = (new_pos[1] - self.floor_h) <= 0 and (old_pos[1] - self.floor_h) >= 0

            if flag_left and flag_right:
                print("SOMETHING WRONG !!!")

            if flag_left:
                t_left = (self.gripper_left_pos - self.x[ind]).dot(self.gripper_left_normal) \
                         / (self.v[ind].dot(self.gripper_left_normal))
                # self.x[ind] = self.x[ind] + t_left * self.v[ind]
                self.x_b[ind] = self.x[ind] + t_left * self.v[ind]
                C[i] = ind
                obstacle[i] = 1

            if flag_right:
                t_right = (self.gripper_right_pos - self.x[ind]).dot(self.gripper_right_normal) \
                          / (self.v[ind].dot(self.gripper_right_normal))
                # self.x[ind] = self.x[ind] + t_right * self.v[ind]
                self.x_b[ind] = self.x[ind] + t_right * self.v[ind]
                C[i] = ind
                obstacle[i] = 2

            # floor
            if flag_floor:
                t_floor = (self.floor_h - self.x[ind][1]) / self.v[ind][1]
                # self.x[ind] = self.x[ind] + t_floor * self.v[ind]
                self.x_b[ind] = self.x[ind] + t_floor * self.v[ind]
                C[i] = ind
                obstacle[i] = 3

    @ti.func
    def jacobi(self):
        for it in range(self.N_tetrahedron):
            F = self.compute_F_in_local(it)
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
    def updatePos(self, x_star: ti.types.ndarray()):
        for i in range(self.N):
            x_new = ti.Vector([x_star[3*i+0], x_star[3*i+1], x_star[3*i+2]])
            self.x[i] = x_new

    @ti.kernel
    def updateVel(self):
        for i in range(self.N):
            self.v[i] = (self.x[i] - self.x_old[i]) * self.dh_inv
        # print(self.v[self.ijk_2_index(self.N_x-1,0,self.N_z-1)])

    @ti.kernel
    def local_step(self):
        self.initialize_iter_vector()
        # Jacobi Solver
        for k in range(self.jacobi_iter):
            self.clear_iter_vector()
            self.jacobi()

    def assemble_B1(self, C):
        first = True
        AinvIic_iter = np.arange(1)
        for i in range(self.N_surfaces):
            # if C[c_ptr] == self.surface_nodes[i]:
            if C[i] != -1:
                if first:
                    first = False
                    a1 = self.AinvIic.getcol(3 * i + 0).toarray()
                    a2 = self.AinvIic.getcol(3 * i + 1).toarray()
                    a3 = self.AinvIic.getcol(3 * i + 2).toarray()
                    AinvIic_iter = np.hstack((a1, a2, a3))
                else:
                    a1 = self.AinvIic.getcol(3 * i + 0).toarray()
                    a2 = self.AinvIic.getcol(3 * i + 1).toarray()
                    a3 = self.AinvIic.getcol(3 * i + 2).toarray()
                    AinvIic_iter = np.hstack((AinvIic_iter, a1, a2, a3))
        return csc_matrix(AinvIic_iter)

    @ti.kernel
    def correct_x_star(self, C: ti.types.ndarray(), x_star: ti.types.ndarray(), x_corrected: ti.types.ndarray()):
        for i in range(3*self.N):
            x_star[i] = x_corrected[i, 0]
        for i in range(self.N_surfaces):
            if C[i] != -1:
                ind = C[i]
                x_star[3 * ind + 0] = self.x_b[ind][0]
                x_star[3 * ind + 1] = self.x_b[ind][1]
                x_star[3 * ind + 2] = self.x_b[ind][2]

    def global_step(self, sn, length):
        dim = 3 * self.N
        dh2_inv = self.dh_inv**2

        xn = self.x.to_numpy().reshape(dim)
        p = self.x_proj.to_numpy().reshape(dim)

        if length > 0:
            Iic = lil_matrix((3*self.N, 3*length))
            col_id = 0
            for i in range(self.N_surfaces):
                if self.C[i] == -1:
                    continue
                # assemble Iic
                ind = self.C[i]
                Iic[3 * ind + 0, col_id] = 1
                col_id += 1
                Iic[3 * ind + 1, col_id] = 1
                col_id += 1
                Iic[3 * ind + 2, col_id] = 1
                col_id += 1
            # assemble Acc
            first = True
            Aic_np = np.arange(1)
            Acc_np = np.arange(1)
            for i in range(self.N_surfaces):
                if self.C[i] == -1:
                    continue
                # assemble col
                ind = self.C[i]
                a1 = self.A.getcol(3 * ind + 0).toarray()
                a2 = self.A.getcol(3 * ind + 1).toarray()
                a3 = self.A.getcol(3 * ind + 2).toarray()
                if first:
                    first = False
                    Aic_np = np.hstack((a1, a2, a3))
                else:
                    Aic_np = np.hstack((Aic_np, a1, a2, a3))
            Aic = lil_matrix(Aic_np)
            first = True
            for i in range(self.N_surfaces):
                if self.C[i] == -1:
                    continue
                # assemble row
                ind = self.C[i]
                b1 = Aic.getrow(3 * ind + 0).toarray()
                b2 = Aic.getrow(3 * ind + 1).toarray()
                b3 = Aic.getrow(3 * ind + 2).toarray()
                if first:
                    first = False
                    Acc_np = np.vstack((b1, b2, b3))
                else:
                    Acc_np = np.vstack((Acc_np, b1, b2, b3))
            Acc = lil_matrix(Acc_np)

            B1 = self.assemble_B1(self.C)
            B2 = Iic - B1 @ Acc
            B3 = np.hstack((B1.toarray(), B2.toarray()))
            B3 = csc_matrix(B3)
            PURt = (Aic - Iic @ Acc).transpose()
            PULt = Iic.transpose()
            VPt = np.vstack((PURt.toarray(), PULt.toarray()))
            data = np.ones(6 * length)
            offset = np.array([0])
            I2c2c = dia_matrix((data, offset), shape=(6 * length, 6 * length))
            B4 = I2c2c - VPt @ B3

            x_b = self.x_b.to_numpy().reshape(dim)
            b = dh2_inv * self.M @ sn + self.stiffness * self.sum_GcTGc @ (p - x_b)

            bt = csc_matrix(b)
            ans1, info = linalg.cg(self.A, b, x0=xn)
            ans1 = csc_matrix(ans1).transpose() # col vector
            ans2 = np.hstack(((bt @ B2).toarray(), (bt @ B1).toarray()))
            ans3 = linalg.spsolve(B4, csc_matrix(ans2).transpose())
            ans3 = csc_matrix(ans3).transpose()
            x_corrected = (ans1 + B3 @ ans3).toarray()   # col vector
            x_star = np.zeros(3*self.N)
            self.correct_x_star(self.C, x_star, x_corrected)
        else:
            b = dh2_inv * self.M @ sn + self.stiffness * self.sum_GcTGc @ p
            x_star, info = linalg.cg(self.A, b, x0=xn)
        return x_star

    @ti.func
    def locate_C(self, C, ind):
        ptr = -1
        for i in ti.static(range(self.N_surfaces)):
            if int(C[i]) == ind:
                ptr = i
        return ptr

    @ti.kernel
    def initialize_solution(self, sn: ti.types.ndarray(), obstacle: ti.types.ndarray()):
        for i in range(self.N):
            if obstacle[i] == -1:
                self.x[i] = ti.Vector([sn[3 * i + 0], sn[3 * i + 1], sn[3 * i + 2]])
            else:
                self.x[i] = self.x_b[i]

    @ti.kernel
    def update_C(self, C: ti.types.ndarray(), r: ti.types.ndarray(), obstacle: ti.types.ndarray()):
        epsilon = self.epsilon
        C_vec = ti.Vector([C[i] for i in range(self.N_surfaces)])
        for i in range(self.N_surfaces):
            ind = self.surface_nodes[i]
            if obstacle[i] == 1:
                # left
                r_vec = ti.Vector([r[3 * ind + 0, 0], r[3 * ind + 1, 0], r[3 * ind + 2, 0]])
                r_n = r_vec.dot(self.gripper_left_normal)
                if r_n < 0:
                    # remove from C
                    obstacle[i] = -1
                    C[i] = -1
                    continue

            if obstacle[i] == 2:
                # right
                r_vec = ti.Vector([r[3 * ind + 0, 0], r[3 * ind + 1, 0], r[3 * ind + 2, 0]])
                r_n = r_vec.dot(self.gripper_right_normal)
                if r_n < 0:
                    # removed from C
                    obstacle[i] = -1
                    C[i] = -1
                    continue

            if obstacle[i] == 3:
                # floor
                r_n = r[3 * ind + 1, 0]
                # print(r_n)
                if r_n < 0:
                    # removed from C
                    obstacle[i] = -1
                    C[i] = -1
                    continue

            if obstacle[i] == -1:
                # no contact
                # check left
                xl = self.x[ind] - self.gripper_left_pos
                dis = xl.dot(self.gripper_left_normal)
                if dis < 0:
                    C[i] = ind
                    obstacle[i] = 1
                    continue
                # check right
                xr = self.x[ind] - self.gripper_right_pos
                dis = xr.dot(self.gripper_right_normal)
                if dis < 0:
                    C[i] = ind
                    obstacle[i] = 2
                    continue
                # check floor
                dis = self.x[ind][1] - self.floor_h
                if dis < 0:
                    C[i] = ind
                    obstacle[i] = 3
                    continue

    def sort(self, c):
        c.sort()
        start = 0
        for i in range(self.N_surfaces):
            if c[i] == -1:
                start = i
            else:
                break
        return start

    def update(self):
        dim = 3 * self.N
        dh = self.dh
        dh2_inv = self.dh_inv**2

        self.x_old.copy_from(self.x)

        xn = self.x.to_numpy().reshape(dim)
        vn = self.v.to_numpy().reshape(dim)
        f_ext = self.f_ext.to_numpy().reshape(dim)
        sn = xn + dh * vn + (dh ** 2) * linalg.inv(self.M) @ f_ext
        sn_spm = csc_matrix(sn).transpose()
        self.clear_C(self.C, self.obstacle)
        self.contact_detect(self.C, self.obstacle)
        length = self.is_contact(self.C)
        new_length = length

        while True:
            print("before: ", self.C)
            # print(self.obstacle)
            length = new_length
            self.initialize_solution(sn, self.obstacle)
            for i in range(20):
                self.local_step()
                x_star = self.global_step(sn, length)
                self.updatePos(x_star)

            self.compute_elastic_force()
            fint = csc_matrix(self.f.to_numpy().reshape(dim)).transpose()
            x_spm = csc_matrix(self.x.to_numpy().reshape(dim)).transpose()
            r = dh2_inv * self.M @ (x_spm - sn_spm) - fint
            self.update_C(self.C, r.toarray(), self.obstacle)
            new_length = self.is_contact(self.C)
            print("after: ", self.C)
            print(length, new_length)
            if length == new_length:
                self.updateVel()
                print(self.v[2])
                break


cube = Object()

window = ti.ui.Window("FEM Simulation", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
canvas.set_background_color((0.2, 0.2, 0.3))
# wait = input("PRESS ENTER TO CONTINUE.")
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