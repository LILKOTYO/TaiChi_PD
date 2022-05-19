import taichi as ti

ti.init(arch=ti.gpu)

dim_length = 64
dim_width = 64
dim_height = 64
x = ti.Vector.field(3, ti.f32)
block1 = ti.root.dense(ti.i, 10).place(x)
y = x
print(y[0])
y[0] = ti.Vector([1.0, 2.0, 3.0])
print(y[0])
print(x[0])

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

