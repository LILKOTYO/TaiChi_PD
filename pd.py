# this is a pd implementation in a games 103 style.
# the difference between the pd paper is that the gradient of energy W is more physical

import taichi as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.init(arch=ti.gpu, default_fp=real)

# try to simulate a 64x64x64 soft body
# length: x     width: y    height: z
dim_length = 9
dim_width = 9
dim_height = 9
cell_size = 1.0 / dim_height

mass = 1.0
stiffness = 50
max_steps = 1024
n_objects = dim_length * dim_width * dim_height
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(3, dtype=real)

# vectors init
q = scalar()
last_q = scalar()
p = scalar()
v = scalar()
force = scalar()
vertices = ti.Vector.field(3, real, n_objects)
link_anchor_a = ti.field(ti.i32)
link_anchor_b = ti.field(ti.i32)
link_length = scalar()

# matrix init
M_builder = ti.linalg.SparseMatrixBuilder(3*n_objects, 3*n_objects, max_num_triplets=3*n_objects)
H_builder = ti.linalg.SparseMatrixBuilder(3*n_objects, 3*n_objects, max_num_triplets=3*n_objects)

# init the links (constraints / edges) and the origin length
links = [[1, 0, 0], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
         [-1, 0, 0], [-1, 1, 0], [-1, -1, 0], [-1, 0, 1], [-1, 0, -1],
         [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
         [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]]
sqrt2 = ti.sqrt(2)
len = [1, sqrt2, sqrt2, sqrt2, sqrt2, 1, sqrt2, sqrt2, sqrt2, sqrt2, 1, 1, 1, 1, sqrt2, sqrt2, sqrt2, sqrt2]
links = [ti.Vector(v) for v in links]


@ti.func
def copy(src: ti.template(), dst: ti.template()):
    assert src.shape == dst.shape
    for I in ti.grouped(src):
        dst[I] = src[I]


def allocate_fields():
    ti.root.dense(ti.i, 3*n_objects).place(p, q, last_q, v, force)


@ti.func
def get_coordinate(idx: ti.i32):
    z = idx // (dim_length * dim_width)
    tmp1 = idx % (dim_length * dim_width)
    y = tmp1 // dim_length
    x = tmp1 % dim_length
    return x, y, z


@ti.func
def get_index(x: ti.i32, y: ti.i32, z: ti.i32):
    return x + y * dim_length + z * dim_length * dim_width


@ti.kernel
def reset():
    for i in range(n_objects):
        x, y, z = get_coordinate(i)
        x[i] = ti.Vector([x, y, z])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        force[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.func
def update_force():
    for a in range(n_objects):
        a_x, a_y, a_z = get_coordinate(a)
        for k in ti.grouped(links):
            b_x = a_x + links[k][0]
            b_y = a_y + links[k][1]
            b_z = a_z + links[k][2]
            b = get_index(b_x, b_y, b_z)
            ab = p[b] - p[a]
            current_len = ab.norm()
            force[a] += stiffness * ab.normalized() * (current_len - len[k])


@ti.func
def pre_calculate():
    # Set the mass matrix and the Hessian matrix
    dim = 3 * n_objects
    for i in range(dim):
        M_builder[i, i] += mass
    for

@ti.kernel
def forward():
    pre_calculate()
    for i in range(max_steps):



@ti.kernel
def clear_tensor():
    for i in range(n_objects):
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        force[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def set_vertices():
    for i in range(n_objects):
        vertices[i] = p[i]


allocate_fields()
reset()



