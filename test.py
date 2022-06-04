import taichi as ti

ti.init(arch=ti.gpu)

z = 3
print(z**2)


# z = 1
# x = []
# print(x)
# x.append([2, 0])
# print(x)
# x.append([2, z * 5])
# print(x)
# x.append([2, 2])
# print(x)
#
# for i in range(len(x)):
#     print(x[i][0])

# x = ti.Vector(3, ti.f32)
# x = ti.Vector.field(3, ti.f32, 5)
#
# @ti.kernel
# def fill_in():
#     for i in range(5):
#         a = float(i)
#         x[i] = ti.Vector([a, a, a])
#
#
# fill_in()
# print(x)
# xa = x[2] - x[0]
# print(xa)
# xd = (x[2].cross(x[1])).dot(x[3])
# print(xd)
# dim_length = 63
# dim_width = 63
# dim_height = 63
#
#
# @ti.kernel
# def test():
#     I = ti.Matrix([[1.0, 2.0], [3.0, 4.0]])
#     print(I)
#
#
# test()
# spring = ti.Vector.field(2, ti.i32, dim_height)
#
#
# @ti.kernel
# def test():
#     for i in spring:
#         print(i)
#         print(spring[i])
#
#
# test()
# n_objects = (dim_length + 1) * (dim_width + 1) * (dim_height + 1)
# n_faces = dim_length * dim_width * 2 * (dim_height + 1) \
#     + dim_width * dim_height * 2 * (dim_length + 1) \
#     + dim_height * dim_length * 2 * (dim_width + 1)
# n_edges = dim_length * (dim_width + 1) * (dim_height + 1) \
#     + dim_width * (dim_height + 1) * (dim_length + 1) \
#     + dim_height * (dim_length + 1) * (dim_width + 1) \
#     + n_faces
#
# print(n_objects, n_faces, n_edges)

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

