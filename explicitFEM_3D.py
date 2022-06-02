import taichi as ti
import math

ti.init(arch=ti.gpu)

# init
init_x, init_y, init_z = 0.3, 0.3, 0.3
N_x = 10
N_y = 10
N_z = 10
# N_x = 2
# N_y = 2
N = N_x * N_y * N_z
# axis-x + axis-y + axis-z + diagonal_xy + diagonal_xz + diagonal_yz
N_edges = (N_x - 1) * N_y * N_z + (N_y - 1) * N_x * N_z + (N_z - 1) * N_x * N_y \
    + (N_x - 1) * (N_y - 1) * N_z + (N_x - 1) * (N_z - 1) * N_y + (N_y - 1) * (N_z - 1) * N_x
N_tetrahedron = 5 * (N_x - 1) * (N_y - 1) * (N_z - 1)
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
edges = ti.Vector.field(2, ti.i32, N_edges)


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
def initialize():
    YoungsModulus[None] = 5e5
    paused = True
    # init position and velocity
    for i, j, k in ti.ndrange(N_x, N_y, N_z):
        index = ijk_2_index(i, j, k)
        x[index] = ti.Vector([init_x + i * dx, init_y + j * dx, init_z + k * dx])
        v[index] = ti.Vector([0.0, 0.0, 0.0])

# both Ds and Dm?         F = DsDm^-1
@ti.func
def compute_D(i):
    a = tetrahedrons[i][0]
    b = tetrahedrons[i][1]
    c = tetrahedrons[i][2]
    d = tetrahedrons[i][3]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a], x[d] - x[a]])

# @ti.func
# def compute_V0(i):
#     a = tetrahedrons[i][0]
#     b = tetrahedrons[i][1]
#     c = tetrahedrons[i][2]
#     d = tetrahedrons[i][3]
#     ab = x[b] - x[a]
#     ac = x[c] - x[a]
#     ad = x[d] - x[a]
#     return ti.abs((ab.cross(ac)).dot(ad))

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

# @ti.kernel
# def compute_total_energy():
#     for i in range(N_tetrahedron):
#         Ds = compute_D(i)
#         F = Ds @ elements_Dm_inv[i]
#         # co-rotated linear elasticity
#         R = compute_R_2D(F)
#         Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
#         element_energy_density = LameMu[None]*((F-R)@(F-R).transpose()).trace() + 0.5*LameLa[None]*(R.transpose()@F-Eye).trace()**2
#
#         total_energy[None] += element_energy_density * elements_V0[i]

@ti.kernel
def update():
    # perform time integration
    for i in range(N):
        # symplectic integration
        # elastic force + gravitation force, dividing mass to get the acceleration
        acc = -grad[i] - ti.Vector([0.0, g, 0.0])
        v[i] += dh * acc
        x[i] += dh * v[i]

    # explicit damping (ether drag)
    for i in v:
        v[i] *= ti.exp(-dh*5)

    for j, k in ti.ndrange(N_y, N_z):
        x[ijk_2_index(0, j, k)] = ti.Vector([init_x, init_y + j * dx, init_z + k * dx])
        v[ijk_2_index(0, j, k)] = ti.Vector([0.0, 0.0, 0.0])


    # x[ijk_2_index(0, N_y, 0)] = ti.Vector([init_x, init_y + N_y * dx, init_z])
    # v[ijk_2_index(0, N_y, 0)] = ti.Vector([0.0, 0.0, 0.0])
    #
    # x[ijk_2_index(N_x, N_y, 0)] = ti.Vector([init_x + N_x * dx, init_y + N_y * dx, init_z])
    # v[ijk_2_index(N_x, N_y, 0)] = ti.Vector([0.0, 0.0, 0.0])

    # enforce boundary condition
    # for j in range(N_y):
    #     ind = ijk_2_index(0, j)
    #     v[ind] = ti.Vector([0, 0])
    #     x[ind] = ti.Vector([init_x, init_y + j * dx])  # rest pose attached to the wall

    # for i in range(N):
    #     if x[i][0] < init_x:
    #         x[i][0] = init_x
    #         v[i][0] = 0


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
# for i in range(N_tetrahedron):
#     print(f"x in the tet {i} is {tetrahedrons[i]}")
#     print(x[tetrahedrons[i][0]])
#     print(x[tetrahedrons[i][1]])
#     print(x[tetrahedrons[i][2]])
#     print(x[tetrahedrons[i][3]])
#     print(" inv Dm is ")
#     print(elements_Dm_inv[i])
updateLameCoeff()

window = ti.ui.Window("FEM Simulation", (800, 800), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
while window.running:
    for i in range(10):
        compute_gradient()
        # print(grad[300])
        update()

    camera.position(0.5, 0.5, 2)
    camera.lookat(0.5, 0.5, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0.5, 1, 2), color=(1, 1, 1))
    scene.particles(x, radius=0.005, color=(0.8, 0.8, 0.8))
    canvas.scene(scene)
    window.show()

# gui = ti.GUI('Linear FEM', (800, 800))
# while gui.running:
#     # numerical time integration
#     for i in range(substepping):
#         compute_gradient()
#         update()
#     print(" 1 frame passed")
#     # render
#     pos = x.to_numpy()
#     for i in range(N_edges):
#         a, b = edges[i][0], edges[i][1]
#         gui.line((pos[a][0], pos[a][1]),
#                  (pos[b][0], pos[b][1]),
#                  radius=1,
#                  color=0xFFFF00)
#     # gui.line((init_x, 0.0), (init_x, 1.0), color=0xFFFFFF, radius=4)
#
#     # text
#     gui.text(
#         content=f'9/0: (-/+) Young\'s Modulus {YoungsModulus[None]:.1f}', pos=(0.6, 0.9), color=0xFFFFFF)
#     gui.text(
#         content=f'7/8: (-/+) Poisson\'s Ratio {PoissonsRatio[None]:.3f}', pos=(0.6, 0.875), color=0xFFFFFF)
#
#     gui.show()