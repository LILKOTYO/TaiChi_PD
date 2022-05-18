import taichi as ti
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.init(arch=ti.gpu, default_fp=real)

# try to simulate a 64x64x64 soft body
dim_length = 64
dim_width = 64
dim_height = 64

max_steps = 1024
n_objects = dim_length * dim_width * dim_height
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

x = vec()
v = vec()
force = vec()

# is too big to store all ltime step information!
def allocate_fields():
    ti.root.dense(ti.i, max_steps).dense(ti.j, max_steps).place(x, v, force)


@ti.kernel
def reset():
    for I in ti.grouped(x):
        print(x[I])


allocate_fields()
reset()