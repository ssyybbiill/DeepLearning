import tensorflow as tf
import numpy as np

test = np.array([[1, 2, 3],
                 [2, 13, 4],
                 [5, 4, 3],
                 [1, 2, 7]])

print(np.argmax(test))  # 直接当成一维数组来查，13最大，是第五个元素，从0开始，对应的下标是4
print(np.argmax(test, 0))  # 直接当成二维数组来查，
print(np.argmax(test, 1))
print("========================")
test = tf.Variable([[1, 2, 3],
                    [2, 13, 4],
                    [5, 4, 3],
                    [1, 2, 7]])

print(tf.argmax(test, axis=None))  # 直接当成一维数组来查，13最大，是第五个元素，从0开始，对应的下标是4
print(tf.argmax(test, 0))  # 直接当成二维数组来查，
print(tf.argmax(test, 1))

# test = np.array([[[19, 2, 3],
#                   [2, 21, 2]],
#
#                  [[5, 4, 3],
#                   [1, 2, 3]],
#
#                  [[5, 4, 6],
#                   [1, 2, 3]],
#
#                  [[15, 14, 13],
#                   [11, 12, 3]]
#                  ])
# # 本例中，
# # test形状是4*2*3，这个特别重要，axis=0，就是4个同一位置的元素比较，axis=1就是2个元素比较，axis=2就是3个元素比较。
# # 再举个例子，
# # test形状是3*7*5*10，这个特别重要，axis=0，就是3个同一位置的元素比较，axis=1就是7个元素比较，axis=2就是5个元素比较，axis=3就是10个元素比较。
# print(np.argmax(test, axis=None))  # axis=None和省略结果相同，直接当成一维数组来查，21最大，是第5个元素，从0开始，对应的下标是4
# print("xxxxxxxxx")
# print(np.argmax(test, 0))  # axis = 0，其实是在第0维，也就是shape的第一个数4对应的那一维，比较4个元素的值。输出的是shape除了4之外的2*3的数组。
#
# print("xxxxxxxxx")
# print(np.argmax(test, 1))  # axis = 1，其实是在第1维，也就是shape的第二个数2对应的那一维，比较2个元素的值。输出的是shape除了2之外的4*3的数组。
#
# print("xxxxxxxxx")
# print(np.argmax(test, 2))  # axis = 2，其实是在第2维，也就是shape的第三个数3对应的那一维，比较3个元素的值。输出的是shape除了3之外的4*2的数组。

test = np.array([[[1, 2, 3],
                  [2, 21, 2]],

                 [[15, 4, 11],
                  [1, 2, 3]],

                 [[5, 4, 6],
                  [1, 2, 3]],

                 [[15, 1],
                  [11, 12]]
                 ])
# 不一致时
print(np.argmax(test, axis=None))  # axis=None和省略结果相同，直接当成一维数组来查，21最大，是第5个元素，从0开始，对应的下标是4
print("xxxxxxxxx")
print(np.argmax(test, 0))
print("xxxxxxxxx")
print(np.argmax(test, 1))
print("xxxxxxxxx")
# print(np.argmax(test, 2))
