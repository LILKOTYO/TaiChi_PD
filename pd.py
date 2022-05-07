import taichi as ti

ti.init(arch=ti.gpu)

# try to simulate a 64x64x64 soft body
dim_length = 64
dim_width = 64
dim_height = 64

max_num_particles = dim_length * dim_width * dim_height
particles = ti.Vector.field(3, ti.f64, shape=max_num_particles)


def particleAt(x, y, z):
    idx = z * dim_length * dim_width + y * dim_length + x
    return particles[idx]


def particleIdx(x, y, z):
    idx = z * dim_length * dim_width + y * dim_length + x
    return idx


@ti.func
def particleCoor(idx):
    tmp = idx
    z = int(tmp / (dim_length * dim_width))
    tmp = tmp % (dim_length * dim_width)
    y = int(tmp / dim_length)
    x = tmp % dim_length
    return x, y, z


@ti.kernel
def init():
    # init position
    for idx in range(dim_length * dim_width * dim_height):
        x, y, z = particleCoor(idx)
        particles[idx][0] = x
        particles[idx][1] = y
        particles[idx][2] = z


