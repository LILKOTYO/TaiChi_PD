import taichi as ti

ti.init(arch=ti.gpu)

dim_length = 9
dim_width = 9
dim_height = 9

n_objects = (dim_length + 1) * (dim_width + 1) * (dim_height + 1)
n_faces = dim_length * dim_width * 2 * (dim_height + 1) \
    + dim_width * dim_height * 2 * (dim_length + 1) \
    + dim_height * dim_length * 2 * (dim_width + 1)
n_edges = dim_length * (dim_width + 1) * (dim_height + 1) \
    + dim_width * (dim_height + 1) * (dim_length + 1) \
    + dim_height * (dim_length + 1) * (dim_width + 1) \
    + n_faces

print(n_objects, n_faces, n_edges)

# x = ti.Vector.field(3, ti.f32)
# block1 = ti.root.dense(ti.i, 10).place(x)
# y = x
# print(y[0])
# y[0] = ti.Vector([1.0, 2.0, 3.0])
# print(y[0])
# print(x[0])

# @ti.func
# def get_coordinate(idx: ti.i32):
#     z = idx // (dim_length * dim_width)
#     tmp1 = idx % (dim_length * dim_width)
#     y = tmp1 // dim_length
#     x = tmp1 % dim_length
#     return x, y, z


# @ti.kernel
# def test01():
#     x, y, z = get_coordinate(262000)
#     print(x, y, z)
#
#
# test01()

