import tensorflow as tf

#
# 测试两个tensor的“==”关系
# "=="，用在两个tensor之间，表示两个tensor的对应元素是否相等
y_yyy = tf.constant([6, 5, 9, 3, 2, 3, 2, 4, 4, 8, 2, 5])
y_hat = tf.constant([0, 7, 9, 0, 6, 0, 2, 4, 4, 2, 2, 6])
print(y_hat == y_yyy)
print("==================================")
print(tf.cast(
    y_hat == y_yyy
    , dtype=tf.int64))  # 输出：tf.Tensor([0 0 1 0 0 0 1 1 1 0 1 0], shape=(12,), dtype=int64)
print("00000000000000000000000000000000000000000000000000000000000")
print(
    tf.reduce_sum(
        tf.cast(
            y_hat == y_yyy
            , dtype=tf.int64)
    )
)  # 输出：tf.Tensor(5, shape=(), dtype=int64)
print(
    tf.reduce_sum(
        tf.cast(
            y_hat == y_yyy
            , dtype=tf.int64)
    ).numpy()
)  # 输出：5，共5个元素相等
