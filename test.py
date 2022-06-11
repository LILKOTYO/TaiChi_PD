import taichi as ti
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, linalg, lil_matrix
import math
ti.init(arch=ti.gpu)
q = 1
w = 2
a = q < 5 and w > 3
print(a)

# a = np.zeros(5)
#
# @ti.func
# def test2(ind):
#     print(ind)
#     return 5
#
#
# @ti.kernel
# def test(a: ti.types.ndarray(), ind: int) -> ti.types.ndarray():
#     vec = ti.Vector([a[i] for i in range(5)])
#     print(vec)
#     return np.arange(5)
#
#
# b = test(a, 3)
# print(a)
# print(b)

# a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# A = csc_matrix(a)
# b = np.array([[0.0, 2.0, 3.0]])
# B = csc_matrix(b)
# c = np.array([0.0, 0.0, 0.0])
# C = csc_matrix(c)
# print(C.get_shape())
# print(B.get_shape())
# print(C.toarray())
# print(B.toarray())
# Bt = B.transpose()
# print(type(A@Bt))
# print(type(Bt))
# res = (A@Bt + Bt).toarray()
# print(res)



# b = np.arange(5)
# print(b)
# B = csc_matrix(b)
# print(B.toarray())
# A = B.transpose() @ B
# print(type(A))
# print(A.toarray().transpose())
# A = lil_matrix((3, 3))
# A[0, 0] = 1
# A[0, 2] = 5
# A[2, 2] = 6
# B = lil_matrix((3, 3))
# B[1, 1] = 3
# B[2, 2] = 7
# # C_np = np.hstack((A.toarray(), B.toarray()))
# # print(C_np)
# D = A @ B
# E = A - D
# print(type(D))
# print(D.toarray())
# print(type(E))
# print(type(A))
# print(E.toarray())
# A = lil_matrix((3, 4))
#
# def change(a: lil_matrix):
#     a[0, 0] = 1
#     a[0, 2] = 5
#     a[2, 2] = 6
#
# change(A)
# print(A.toarray())
# A[0, 2] = 1 - A[0, 2]
# print(type(A[0, 2]))
# print(A.toarray())
# print(type(A))
# ti.init(arch=ti.gpu)
# a = np.array([-1 for i in range(10)])
# print(a)
#
# @ti.kernel
# def test(array: ti.types.ndarray()):
#     for i in range(10):
#         array[i] = 2
#
#
# test(a)
# print(a)


# a = ti.field(ti.i32, 30)
# b = np.arange(30)
# a.from_numpy(b)
# print(a)
#
# @ti.kernel
# def test():
#     ti.append(a, 0, 1)
#
#
# test()
# print(a)
# A = csc_matrix((3, 4))
# A[0, 0] = 1
# A[0, 2] = 5
# A[2, 2] = 6
# # x = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
# x = np.array([[0], [1], [2], [3]])
# b = A @ x
# print(type(b))
# B = csc_matrix(b)
# print(B.toarray())
# print(B.get_shape())
#
# a1 = A.getcol(0).toarray()
# a2 = A.getcol(2).toarray()
# b = np.hstack((a1, a1, a2))
# print(b)
# a = np.array([0, 0, 0, 0])
# print(type(a))
# b = np.zeros(5)
# print(type(b))
# c = np.arange(12)
# print(type(c))
# data = np.ones(12)
# print(data)
# A = csr_matrix((3, 4))
# B = csr_matrix((3, 4))
# B[0, 0] = 1
# l = [A]
# l.append(B)
# print(len(l))
# for i in range(len(l)):
#     print(l[i].toarray())
# A = ti.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
# print(A)
# Iic = ti.Matrix([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
# print(Iic)
# B = A @ Iic
# print(B)
# print(type(B))

# @ti.kernel
# def add(result: ti.types.ndarray(), param: int):
#     for i in range(5):
#         result[i] += param
#
#
# add(one, 3)
# print(one)

# A = np.array([[ 3. ,  2. , -1. ],
#               [ 2. , -2. ,  4. ],
#               [-1. ,  0.5, -1. ]])
# solve = linalg.factorized(A) # Makes LU decomposition.
# rhs1 = np.array([1, -2, 0])
# x = solve(rhs1) # Uses the LU factors.
# print(x)
# a = ti.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
# print(a)
# b = a.to_numpy().reshape(9)
# print(b)
# print(a)

# center = ti.Vector([0.0, 0.0, 0.0])
# a = ti.Vector([0.0, 3.0, 0.0])
# print(center)
# center.copy_from(a)
# print(center)
# n = 10
# ex = np.ones(n)
# data = np.array([ex, 2 * ex, ex])
# offsets_a = np.array([-1, 0, 1])
# A = dia_matrix((data, offsets_a), shape=(n, n))
# print(A.toarray())
# offsets_i = np.array([0])
# I = dia_matrix((ex, offsets_i), shape=(n, n))
# B = (3 * A) @ ex
# print(B)

# K = ti.Vector.field(3, ti.f32, 2)
# print(K[0][0])


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

