import numpy as np

a = np.array([1, 2, 3, 4])
print(a)
print(type(a))
# 形状，行列数
print(a.shape)
# 1行，-1表示占位符，只有一行的话自动匹配列数
# 如果行数不能整除，则报错
a = a.reshape((1, -1))
print(a.shape)
print(a)
a = np.array([1, 2, 3, 4, 5, 6])
# a = a.reshape((5, -1))
a = a.reshape((3, -1))
print(a.shape)
print(a)
print(a[2, 0])
a = np.zeros((4, 2))
print(a)
a = np.ones((3, 2))
print(a)
a = np.full((4, 3), 3)
print(a)
# eye函数，创建单位矩阵
a = np.eye(3)
print(a)
# random.random函数
a = np.random.random((3, 4))
print(a)

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
# -2: 从倒数第二行开始到最后一行，1:3 从第二列开始两列
print(a[-2:, 1:3])
# 第二行，倒数第二个数
print(a[1, -2])

# 整数操作的不同
b = a[1:2, 2:4]
print(b.shape)  # 二维，因为没有整数操作

b = a[1, 2:4]
print(b.shape)  # 一维，因为进行了整数操作

# 0-2行的第二列每个元素+10
a[np.arange(3), 1] += 10
a[np.arange(3), [1, 1, 1]] += 10
a[[0, 1, 2], [1, 1, 1]] += 10
print(a)

# 获取a中所有>10的元素
# 法一
result_index = a > 10
print(result_index)
print(a[result_index])
# 法二
print(a[a > 10])

# numpy数组元素数据类型
a = np.array([1, 2, 3.1])
print(a.dtype)
a = np.array([1.1, 2.6], dtype=np.int64)
print(a.dtype)
# a+b a-b，矩阵对应位置元素加减
# 数组赋初值时候，总是丢掉最外层的[]！！！
a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])
print(a + b)
print(a - b)
print(a * b)
print(np.multiply(a, b))
print(a / b)
print(np.divide(a, b))
print(np.sqrt(a))
# 矩阵相乘
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.dot(b))
print(np.dot(a, b))

# 常用函数
print(a)
print(np.sum(a))
print(np.sum(a, axis=0))  # 列求和
print(np.sum(a, axis=1))  # 行求和
print(np.mean(a))
print(np.mean(a, axis=0))  # 列平均值
# uniform生成一定范围内的随机小数
print(np.random.uniform(3, 5))
# tile表示矩阵分别在行列上复制的重数
print(np.tile(a, (1, 3)))  # 行重复1次，列重复3次
# argsort每行、列按照从小到大的顺序排列，返回下标
a = np.array([[3, 6, 2, 1],
              [4, 2, 0, 4]])
print(a.argsort())  # 默认按行升序
print(a.argsort(axis=0))  # 按列升序排序，显示下标
# 矩阵转置
print(a)
print(a.T)
print(np.transpose(a))
# 实现a的每一行都加b
# 法一：
a = np.array([[1, 2, 3, 5],
              [3, 4, 2, 5],
              [1, 2, 3, 4],
              [2, 3, 3, 2]])
b = np.array([5, 6, 8, 8])
for i in range(4):
    a[i, :] += b
print(a)
# 法二
a += np.tile(b, (4, 1))
print(a)
# 法三：广播，自动补全缺失维度
b = np.array([5, 6, 8, 8])
print(a + b)

