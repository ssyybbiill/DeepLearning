import tensorflow as tf


# print(tf.__version__)

# 2.3 自动求梯度：本节将介绍如何使用tensorflow2.0提供的GradientTape来自动求梯度。
# 2.3.1 简单示例
# 思考：tf.constant和tf.Variable区别？
# 一个常量一个变量，用tf.Variable()保证参数可训练
# 下面参数中使用tf.Variable而不是constant的原因是，指定dtype方式不同，Variable直接在生成时候指定，constant需要赋值给另一个值用cast强转
# 例1：
# # x = tf.reshape(tf.Variable(range(4), dtype=tf.float32), (4, 1))
# # x = tf.reshape(tf.constant(range(4), dtype=tf.float32), (4, 1))
# x = tf.reshape(tf.range(4, dtype=tf.float32), (4, 1))
# with tf.GradientTape() as t:  # 上下文胶带
#     t.watch(x)  # watch函数把需要计算梯度的变量x加进来了，确保某个tensor被tape追踪
#     y = 2 * tf.matmul(tf.transpose(x), x)
# dy_dx = t.gradient(y, x)
# print(dy_dx)

# 2.3.2 训练模式和预测模式   文不对题？？？
# x = tf.reshape(tf.range(4, dtype=tf.float32), (-1, 2))
# print(x)
# with tf.GradientTape(persistent=True) as g:  # 上下文胶带
#     # persistent=True，此时不会自动释放资源了，需要自己删掉tape
#     g.watch(x)  # watch函数把需要计算梯度的变量x加进来了，确保某个tensor被tape追踪
#     y = x * x
#     z = y * y
# dz_dx = g.gradient(z, x)
# dy_dx = g.gradient(y, x)
# print(dz_dx)
# print(dy_dx)
# del g

# 2.3.3 对Python控制流求梯度
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c


a = tf.random.normal((1, 1), dtype=tf.float32)
with tf.GradientTape() as t:
    t.watch(a)
    c = f(a)
print(t.gradient(c, a) == c / a)

# 打印dtypes和random模块中所有的成员或属性。
print(dir(tf.dtypes))
print(dir(tf.random))

print(help(tf.ones))
