import taichi as ti
ti.init(arch=ti.cuda)

N = 10
# array = ti.Vector.field(3, ti.f32, (N, N))
#
# @ti.kernel
# def test():
#     for i in ti.grouped(array):
#         print(i)
links = [[-1, 0], [1, 2], [0, -1], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
links = [ti.Vector(v) for v in links]


@ti.kernel
def test():
    for i in ti.static(links):
        print(i)
    x = ti.abs(-3)
    print(x)

test()
