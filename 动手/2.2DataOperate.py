import tensorflow as tf

#
# # print(tf.__version__)
#
# 2.2.1 创建tensor

# 三类方法：
# 1.常量
# 1.1 tf.constant(value, dtype=None, shape=None)，tf.convert_to_tensor(value, dtype=None)，tf.fill(dims, value)。
#       value：可以是range(3)，可以是list [[1,2,3],[2,3,5]]，也可以是numpy实例np.ones(shape)。
#       shape：一维格式shape=[2]或(2)，二维格式shape=[2,3]或(2,3),三维格式shape=[2,3,4]或(2,3,4)。
#       dims：一维格式[2]或(2)，二维格式[2,3]或(2,3),三维格式[2,3,4]或(2,3,4)。
print(tf.constant([[0, 1, 2], [3, 4, 5]]))  # 默认shape不变
print(tf.constant([[0, 1, 2], [3, 4, 5]], shape=[1, 6]))  # shape可以重塑
# print(tf.constant([[0, 1, 2], [3, 4, 5]], shape=[1, -1]))  # shape中行、列不能用-1来省略，报错！
print(tf.convert_to_tensor([[0, 1, 2], [3, 4, 5]]))  # 没有shape参数
print(tf.fill([2, 3, 2], 8))  # 创建元素相同的tensor,dims不能省略参数
# 1.2 tf.zeros(shape, dtype=dtypes.float32)，tf.ones(shape, dtype=dtypes.float32)。
#       shape：一维格式[2]或(2)，二维格式[2,3]或(2,3),三维格式[2,3,4]或(2,3,4)。
print(tf.zeros([2, 3, 2]))
print(tf.ones([2, 3, 2]), tf.int32)
# 2.随机量
# 正态分布：
#   tf.random.normal(shape,mean=0.0,stddev=1.0,dtype=dtypes.float32,seed=None)
# 截断正态分布：去掉梯度消失部分的正态分布。生成的值遵循一个正态分布，但不会大于平均值2个标准差。
#   tf.random.truncated_normal(shape,mean=0.0,stddev=1.0,dtype=dtypes.float32,seed=None)
#       shape：格式[3,4]
#       mean：均值，默认0。数据类型为dtype的张量值或Python值。
#       stddev：标准差，默认1。数据类型为dtype的张量值或Python值。
#       dtype：类型，常省略，默认dtype=tf.float32。格式dtype=tf.xx，不能是int类型，会报错。
#       seed：一个Python整数。是随机种子。
# 均匀分布：
#   tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None)
#       minval, maxval：生成值的范围，[minval, maxval)，左闭右开。
#                       float型，默认范围为[0, 1)，int型，至少maxval必须明确指定。
#       dtype：默认dtype=tf.float32。
print(tf.random.normal([3, 4]))
print(tf.random.truncated_normal(shape=[2, 3]))
print(tf.random.uniform(shape=[2, 3]))
print(tf.random.uniform([2, 3], dtype=tf.float64))  # 32位和64位精度不同，值看不出区别
# x = tf.random.uniform([2, 3])
# y = tf.random.uniform([2, 3], dtype=tf.float64)
# print(x + y)  # 32位和64位tensor不能相加，需要统一类型
print(tf.random.uniform([2, 3], minval=2, maxval=10, dtype=tf.float32))
print(tf.random.uniform([2, 3], maxval=10, dtype=tf.int32))
print(tf.random.uniform([2, 3], minval=2, maxval=10, dtype=tf.int64))

# 3.序列
# tf.range(start, limit=None, delta=1, dtype=None),
# 创建元素取值为整数序列的tensor，没有shape参数，默认是一维shape=(4,)
#       start：起始数字，默认0。
#       limit：结束数字。如果只有一个参数，默认是limit。
#       delta：步长，默认1。
#       dtype：默认dtype=tf.int32。
print(tf.range(4))
print(tf.range(4, dtype=tf.float32))
#
# 4.补充
# 如需为tensor指定shape，可以调用tf.reshape(tensor, shape)方法。
#       tensor：可以是tf.constant()，tf.Variable()，tf.range()的返回值。
#       shape：格式[2,3]或(2,3)，行、列可以省略一个（写成-1）
print(tf.reshape(tf.range(4), shape=[1, -1]))
print(tf.reshape(tf.range(4), [-1, 2]))
# #
# x = tf.constant(range(12))
# # print(x.shape)
# # print(len(x))
# X = tf.reshape(x, (3, 4))
# Y = tf.reshape(x, (-1, 4))  # 省略
# # print(X)
# # print(Y)
# # y = tf.zeros((2, 3, 4)) # 同 y = tf.zeros([2, 3, 4])一样，shape用[]和()都可以。
# # print(y)
# # z = tf.ones((3, 4))
# # print(z)
# # X = tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# # print(X)
#
# # X = tf.random.normal(shape=[3, 4], mean=0, stddev=1)  # 均值，标准差
# # print(X)
#
# #
# # 2.2.2 运算
# X = tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# # print(X)
# # print(Y)
# # print(X + Y)
# # print(X * Y)  # 元素相乘
# # print(X / Y)  # 元素相除
# # Y = tf.cast(Y, tf.float32)
# # print(tf.exp(Y))  # e的指数，必须变成浮点数
#
# # X = tf.cast(X, tf.int32)
# # Y = tf.cast(Y, tf.int32)
# # print(Y.shape, ",", X.shape)
# # Z = tf.matmul(X, tf.transpose(Y))  # 矩阵乘法，元素必须是同等类型
# # print(Z)
#
# # # 将多个tensor连结（concatenate），好吧其实就是上下/左右位置放一下
# # Z0 = tf.concat([X, Y], axis=0)  # 上下位置，上X，下Y
# # Z1 = tf.concat([X, Y], axis=1)  # 左右位置，左X，右Y
# # print(Z0)
# # print(Z1)
#
# # Z = tf.equal(X, Y)  # 判断相同位置的元素值是否相等
# # print(Z)
# #
# # z = tf.reduce_sum(X)  # 所有元素求和，得到只有一个元素的tensor
# # print(z)
#
# # # X = tf.constant([[3, 6], [0, -10]])
# # X = tf.cast(X, tf.float32)
# # print(X)
# # # 下面，不管向量还是矩阵，都按照向量的范数公式计算。而不是按照标准矩阵的L1，L2范数计算。
# # print(tf.norm(X))  # Eukl_norm,默认是ord=2，矩阵：元素平方和开根号【不是L2范数，矩阵L2范数是奇异值最大值】；向量：是欧几里得范数/L2范数，元素平方和开根号=向量长度。
# # print(tf.norm(X, ord=1))  # L1_norm，矩阵：是所有元素绝对值之和【不是L1范数，矩阵L1范数是最大列和】；向量：是L1范数，向量所有元素绝对值之和
#
# # # 2.2.3 广播机制：形状不相同的矩阵相加时，触发广播机制
# # A = tf.reshape(tf.constant(range(3)), (3, 1))
# # B = tf.reshape(tf.constant(range(2)), (1, 2))
# # print(A)
# # print(B)
# # print(A + B)
#
# # 2.2.4 索引
# # print(X)
# # print(X[1:3][0])
# # X = tf.Variable(X)
# # X[1, 2].assign(9)  # 矩阵的第二行第三列元素赋值为9
# # print(X)
#
# # X[1:2, :].assign(tf.ones(X[1:2, :].shape, dtype=tf.float32) * 12)  # 注意不要把1:2写成1,2 !!
# # print(X)
#
# # 2.2.5 运算的内存开销
# X = tf.cast(X, dtype=tf.float32)
# Y = tf.cast(Y, dtype=tf.float32)
# print(X)
# print(Y)
# X = tf.Variable(X)
# before = id(Y)
# print(before)
# Y = Y + X
# print(id(Y) == before)  # 内存改变
#
# # 指定结果到特定内存，用前面介绍的索引来进行替换操作
# print(X)
# print(Y)
# Z = tf.Variable(tf.zeros_like(Y))  # 通过zeros_like创建和Y形状相同且元素为0的tensor
# before = id(Z)
# Z[:].assign(X + Y)  # 把X + Y的结果通过[:]写进Z对应的内存中
# print(Z)
# print(id(Z) == before)  # 这一句输出就是废话，肯定是真的，因为前边儿就把id(Z)赋值给了before。
#
# # 上面的操作X+Y还是开了内存，如果想避免这个临时内存开销，我们可以使用assign_{运算符全名}函数。
# # Z = tf.add(X, Y)  # 这个应该和X+Y效果一样
# Z = X + Y
# print(id(Z) == before)
# before = id(X)
# X.assign_add(Y)
# print(id(X) == before)
#
# # 2.2.6 tensor 和 NumPy 相互变换
# # 通过array函数和asnumpy函数令数据在NDArray和NumPy格式之间相互变换
# import numpy as np
#
# P = np.ones((2, 3))
# D = tf.constant(P)  # 将NumPy实例变换成tensor实例
# print(D)
#
# P1 = np.array(D)  # 将NDArray实例变换成NumPy实例
# print(P1)
