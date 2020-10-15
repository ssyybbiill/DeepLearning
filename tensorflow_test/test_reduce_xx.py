import tensorflow as tf

X = tf.constant([[1, 2, 3], [4, 5, 6]])
print(tf.reduce_sum(X, axis=0, keepdims=True),  # keepdims，保持原来X的二维不变，shape=(1, 3)，不变
      tf.reduce_sum(X, axis=1, keepdims=True))  # 保持原来的shape=(2, 1)
print(tf.reduce_sum(X, axis=0))  # 没有keepdims，默认是False，维度就变小了，变成一维向量了，shape=（3，）
print(tf.reduce_sum(X))  # 不指定axis，默认是所有元素求和，变成一个标量，shape=（）
