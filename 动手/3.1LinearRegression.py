import tensorflow as tf
from time import time

# 3.1 线性回归
# 3.1.2.2 矢量计算表达式
a = tf.ones((1000,))
b = tf.ones((1000,))
# 向量相加的一种方法是，将这两个向量按元素逐一做标量加法。
start = time()
c = tf.Variable(tf.zeros((1000,)))
for i in range(1000):
    c[i].assign(a[i] + b[i])  # assign分配、指定，为变量指定一个新值，赋值。
print(time() - start)

# 向量相加的另一种方法是，将这两个向量直接做矢量加法。
start = time()
c.assign(a + b)
print(time() - start)
