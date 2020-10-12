import tensorflow as tf
from tensorflow import data as tfdata
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init
from tensorflow import losses
from tensorflow.keras import optimizers


def func(x):
    true_w = [2, -3.4]
    true_b = 2.0
    # 方法一：x1,x2分开乘
    y = true_w[0] * x[:, 0] + true_w[1] * x[:, 1] + true_b

    # 方法二：x1,x2一起乘
    true_w = tf.reshape([2, -3.4], shape=[2, 1])
    y = tf.matmul(x, true_w) + true_b  # 矩阵乘法用matmul，元素乘法用*；
    # 这里之所以可以加一个数True_b，shape不同，是因为“+”是元素相加，触发了广播机制，而matmuti没有广播机制

    return y


# 3.3.1-3.3.2，生成数据集+划分数据集
def getData():
    # 产生数据集
    num_inputs = 2
    num_examples = 1000

    features = tf.random.normal(shape=(num_examples, num_inputs), stddev=1.0)  # 正态分布产生随机数据
    labels = func(features)
    noise = tf.random.normal(labels.shape, stddev=0.01)
    labels = labels + noise

    # 下面是，数据批次划分

    batch_size = 10
    # 将训练数据的特征和标签组合，使用from_tensor_slices将数据放入队列
    dataset = tfdata.Dataset.from_tensor_slices((features, labels))
    # 使用shuffle()，随机打乱数据集顺序，不用shuffle就是按顺序划分，buffer_size 参数应大于等于样本数
    # dataset = dataset.shuffle(buffer_size=num_examples)
    # batch把dataset按照batch_size分批次，得到一个list集合。默认drop_remainder=False时，保留不足批次的部分，如果是True，就是舍去。
    dataset = dataset.batch(batch_size)
    # dataset = dataset.batch(batch_size).repeat()  # repeat表示重复次数，默认是None，表示数据序列无限延续

    # # 输出
    #
    # # 输出所有batch的list集合。
    # # print(list(dataset.as_numpy_iterator()))
    #
    # # 输出其中一个batch，两种方法，官方推荐way2！
    # print("way1")
    # data_iter = iter(dataset)
    # for X, y in data_iter:
    #     print(X, y)
    #     break
    # print("way2")
    # for (batch_num, (X, y)) in enumerate(dataset):
    #     print((X, y))  # batch_num是批次号，标识符，也可以起其他名字
    #     break

    return dataset, features, labels


if __name__ == '__main__':
    dataset, features, labels = getData()
    model = keras.Sequential()
    model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01))) #
    loss = losses.MeanSquaredError()  # 均方误差损失作为模型的损失函数
    trainer = optimizers.SGD(learning_rate=0.03)  # 梯度下降算法

    num_epochs = 3
    for epoch in range(0, num_epochs):
        for (batch_num, (X, y)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                current_loss = loss(y, model(X, training=True))  # (y_true, y_pred)
            grads = tape.gradient(current_loss, model.trainable_variables)
            # print(grads)
            trainer.apply_gradients(zip(grads, model.trainable_variables))
        current_loss = loss(labels, model(features))
        print("Epoch %2d, loss：%2.5f" % (epoch, current_loss))

    # 比较学到的模型参数和真实的模型参数
    print("true_w =", [2, -3.4], "，pre_w", model.get_weights()[0])
    print("true_b = ", 2.0, "，pre_b=", model.get_weights()[1])
