import taichi as ti
import math

ti.init(arch=ti.gpu)

# init
init_x, init_y, init_z = 0.3, 0.3, 0.3
N_x = 3
N_y = 3
N_z = 3
# N_x = 2
# N_y = 2
N = N_x * N_y * N_z
# axis-x + axis-y + axis-z + diagonal_xy + diagonal_xz + diagonal_yz
N_edges = (N_x - 1) * N_y * N_z + (N_y - 1) * N_x * N_z + (N_z - 1) * N_x * N_y \
    + (N_x - 1) * (N_y - 1) * N_z + (N_x - 1) * (N_z - 1) * N_y + (N_y - 1) * (N_z - 1) * N_x
N_tetrahedron = 5 * (N_x - 1) * (N_y - 1) * (N_z - 1)
N_faces = 4 * (N_x - 1) * (N_y - 1) \
                       + 4 * (N_x - 1) * (N_z - 1) \
                       + 4 * (N_y - 1) * (N_z - 1)
dx = 0.5/N_x

# physical quantities
m = 1
g = 0.5
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time integration)
dh = h/substepping

# simulation components
x = ti.Vector.field(3, ti.f32, N)
v = ti.Vector.field(3, ti.f32, N)
# total_energy = ti.field(ti.f32, ())
grad = ti.Vector.field(3, ti.f32, N)
elements_Dm_inv = ti.Matrix.field(3, 3, ti.f32, N_tetrahedron)
elements_V0 = ti.field(ti.f32, N_tetrahedron)

# geometric components
tetrahedrons = ti.Vector.field(4, ti.i32, N_tetrahedron)
faces = ti.field(ti.i32, N_faces * 3)


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

        for m in range(1):
            # init faces
            fid = 0
            for i, j in ti.ndrange(N_x - 1, N_y - 1):
                faces[fid + 0] = ijk_2_index(i, j, 0)
                faces[fid + 1] = ijk_2_index(i + 1, j, 0)
                faces[fid + 2] = ijk_2_index(i + 1, j + 1, 0)
                faces[fid + 3] = ijk_2_index(i, j, 0)
                faces[fid + 4] = ijk_2_index(i + 1, j + 1, 0)
                faces[fid + 5] = ijk_2_index(i, j + 1, 0)

                faces[fid + 6] = ijk_2_index(i, j, N_z - 1)
                faces[fid + 7] = ijk_2_index(i + 1, j, N_z - 1)
                faces[fid + 8] = ijk_2_index(i + 1, j + 1, N_z - 1)
                faces[fid + 9] = ijk_2_index(i, j, N_z - 1)
                faces[fid + 10] = ijk_2_index(i + 1, j + 1, N_z - 1)
                faces[fid + 11] = ijk_2_index(i, j + 1, N_z - 1)
                fid += 12

            for i, k in ti.ndrange(N_x - 1, N_z - 1):
                faces[fid + 0] = ijk_2_index(i, 0, k)
                faces[fid + 1] = ijk_2_index(i + 1, 0, k)
                faces[fid + 2] = ijk_2_index(i, 0, k + 1)
                faces[fid + 3] = ijk_2_index(i, 0, k + 1)
                faces[fid + 4] = ijk_2_index(i + 1, 0, k)
                faces[fid + 5] = ijk_2_index(i + 1, 0, k + 1)

                faces[fid + 6] = ijk_2_index(i, N_y - 1, k)
                faces[fid + 7] = ijk_2_index(i + 1, N_y - 1, k)
                faces[fid + 8] = ijk_2_index(i, N_y - 1, k + 1)
                faces[fid + 9] = ijk_2_index(i, N_y - 1, k + 1)
                faces[fid + 10] = ijk_2_index(i + 1, N_y - 1, k)
                faces[fid + 11] = ijk_2_index(i + 1, N_y - 1, k + 1)
                fid += 12

            for j, k in ti.ndrange(N_y - 1, N_z - 1):
                faces[fid + 0] = ijk_2_index(0, j, k)
                faces[fid + 1] = ijk_2_index(0, j, k + 1)
                faces[fid + 2] = ijk_2_index(0, j + 1, k)
                faces[fid + 3] = ijk_2_index(0, j + 1, k)
                faces[fid + 4] = ijk_2_index(0, j, k + 1)
                faces[fid + 5] = ijk_2_index(0, j + 1, k + 1)

                faces[fid + 6] = ijk_2_index(N_x - 1, j, k)
                faces[fid + 7] = ijk_2_index(N_x - 1, j, k + 1)
                faces[fid + 8] = ijk_2_index(N_x - 1, j + 1, k)
                faces[fid + 9] = ijk_2_index(N_x - 1, j + 1, k)
                faces[fid + 10] = ijk_2_index(N_x - 1, j, k + 1)
                faces[fid + 11] = ijk_2_index(N_x - 1, j + 1, k + 1)
                fid += 12


@ti.kernel
def initialize():
    YoungsModulus[None] = 1e3
    paused = True
    # init position and velocity
    for i, j, k in ti.ndrange(N_x, N_y, N_z):
        index = ijk_2_index(i, j, k)
        x[index] = ti.Vector([init_x + i * dx, init_y + j * dx, init_z + k * dx])
        v[index] = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def compute_D(i):
    a = tetrahedrons[i][0]
    b = tetrahedrons[i][1]
    c = tetrahedrons[i][2]
    d = tetrahedrons[i][3]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a], x[d] - x[a]])


@ti.kernel
def initialize_elements():
    for i in range(N_tetrahedron):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/6

# ----------------------core-----------------------------
@ti.func
def compute_R_3D(F):
    R, S = ti.polar_decompose(F, ti.f32)
    return R

@ti.kernel
def compute_gradient():
    # clear gradient
    for i in grad:
        grad[i] = ti.Vector([0, 0, 0])

    # gradient of elastic potential
    for i in range(N_tetrahedron):
        Ds = compute_D(i)
        # elements_Dm_inv is initialized in the function named initialize_elements()
        F = Ds@elements_Dm_inv[i]
        # co-rotated linear elasticity
        R = compute_R_3D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # first Piola-Kirchhoff tensor
        P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R
        # assemble to gradient
        # H is the derivative of energy to the position
        H = elements_V0[i] * P @ (elements_Dm_inv[i].transpose())
        a, b, c, d = tetrahedrons[i][0],tetrahedrons[i][1],tetrahedrons[i][2], tetrahedrons[i][3]
        gb = ti.Vector([H[0, 0], H[1, 0], H[2, 0]])
        gc = ti.Vector([H[0, 1], H[1, 1], H[2, 1]])
        gd = ti.Vector([H[0, 2], H[1, 2], H[2, 2]])
        ga = -gb - gc - gd
        grad[a] += ga
        grad[b] += gb
        grad[c] += gc
        grad[d] += gd


@ti.kernel
def update():
    # perform time integration
    for i in range(N):
        # symplectic integration
        # elastic force + gravitation force, dividing mass to get the acceleration
        acc = -grad[i] - ti.Vector([0.0, g, 0.0])
        v[i] += dh * acc
        x[i] += dh * v[i]
        if x[i][1] < 0.1:
            x[i][1] = 0.1
            if v[i][1] < 0:
                v[i][1] = 0.0

    # explicit damping (ether drag)
    for i in v:
        v[i] *= ti.exp(-dh*5)


@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))


# init once and for all
meshing()
initialize()
initialize_elements()
updateLameCoeff()

window = ti.ui.Window("FEM Simulation", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
wait = input("PRESS ENTER TO CONTINUE.")
while window.running:
    for i in range(10):
        compute_gradient()
        update()

    camera.position(0.5, 0.5, 2)
    camera.lookat(0.5, 0.5, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.particles(x, radius=0.005, color=(0.8, 0.8, 0.8))
    scene.mesh(x, faces, color=(0.5, 0.5, 0.5))
    canvas.scene(scene)
    window.show()

