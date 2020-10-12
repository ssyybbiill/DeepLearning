import tensorflow as tf
import matplotlib.pyplot as plt  # from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 虽然底下没用到，但还是得有，因为3d实际上用得到
import numpy as np
import random
from sklearn.model_selection import train_test_split


# 这个打算画出并列的3d图，但是没有找到3d中两个图并列的方法，如果需要，以后再看官方文档吧。

def func(x):
    True_W = [2, -3.4]
    True_b = 2.0
    # 方法一：x1,x2分开乘
    y = True_W[0] * x[:, 0] + True_W[1] * x[:, 1] + True_b

    # 方法二：x1,x2一起乘
    True_W = tf.reshape([2, -3.4], shape=[2, 1])
    y = tf.matmul(x, True_W) + True_b  # 矩阵乘法用matmul，元素乘法用*；
    # 这里之所以可以加一个数True_b，shape不同，是因为“+”是元素相加，触发了广播机制，而matmuti没有广播机制

    return y


# 只利用Tensor和GradientTape来实现一个线性回归
# 1.获取所有数据，包括训练数据和测试数据
# 首先，通过向输入中添加随机高斯（Normal）噪声来合成训练数据：
def getData():
    num_inputs = 2  # 特征向量维数是2，x1,x2
    num_examples = 1000

    # x = tf.random.normal(shape=[num_examples, num_inputs])
    x = tf.random.normal((num_examples, num_inputs))  # shape=(10, 2) # print(x[:, 0])  # shape=(10,)
    noise = tf.random.normal(shape=[num_examples])  # shape=(10,)
    noise = tf.random.normal(shape=[num_examples, 1])  # shape=(10,1)
    y = func(x) + noise
    return x, y


# 自己写的，划分训练集和测试集的方法
def depart_train_test(x, y):
    rate = 0.3
    test_index = random.sample(range(0, len(x) - 1), round(rate * len(x)))  # round四舍五入，向偶取整
    x_test = []
    x_train = []
    y_test = []
    y_train = []
    # x.ix[test_index]
    for i in range(0, len(x)):
        if i in test_index:
            x_test.append(x[i])
            y_test.append(y[i])
        else:
            x_train.append(x[i])
            y_train.append(y[i])
    return x_train, x_test, y_train, y_test


# 2.定义线性模型。
# 让我们定义一个简单的类来封装变量和计算：
class Model(object):
    def __init__(self):
        # self.W = tf.reshape([5.0, 4], (2, 1)) #不应该在这里reshape，应该在下面计算的时候reshape，否则会嵌套很厉害
        self.W = tf.Variable([5.0, 4])
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        W = tf.reshape(self.W, (2, 1))  # 在计算的时候再reshape
        return tf.matmul(x, W) + self.b


# 3.定义损失函数。
# 损失函数衡量给定输入的模型输出与目标输出的匹配程度。目的是在训练过程中尽量减少这种差异。
# 使用标准的L2损失MSE，也称为最小平方误差、最小均方误差：
def loss(target_y, predicted_y):
    # n = len(target_y)
    n = 2
    # return (predicted_y - target_y) ** 2 / n
    return tf.reduce_mean(tf.square(target_y - predicted_y))  # reduce_mean是求所有元素的平均值


# 4.定义训练循环
# 利用网络和训练数据，使用梯度下降训练模型以更新权重变量（W）和偏差变量（b）以减少损失。
# 我们tf.train.Optimizer推荐的实现中包含了梯度下降方案的许多变体。
def train(model, inputs, outputs, learning_rate, batch_size):
    with tf.GradientTape() as tape:
        current_loss = loss(outputs, model(inputs))
    grads = tape.gradient(current_loss, [model.W, model.b])
    # print(grads)
    model.W.assign_sub(learning_rate * grads[0])
    model.b.assign_sub(learning_rate * grads[1])


# 最后，我们通过反复训练数据运行的，看看W和b发展。
if __name__ == '__main__':
    x, y = getData()
    # x_train, x_test, y_train, y_test = depart_train_test(x, y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # 工具方法
    # x, y = getData2()  # x是2维特征
    model = Model()
    y_hat = model(x)
    print(loss(y, y_hat))
    print("=============================")
    Ws, bs = [], []
    epochs = range(10)
    for epoch in epochs:
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(y, model(x))  # 这里使用model(x)，而不是y_hat，是因为model不断变化,loss也变化。而y_hat是一个固定值。
        train(model, x, y, learning_rate=0.1, batch_size=len(y))
        print(
            "Epoch %2d: W=[%1.2f,%1.2f] b=%1.2f, loss=%2.5f" % (epoch, Ws[-1][0], Ws[-1][1], bs[-1], current_loss))

    # 画二维图
    # plt.scatter(x[:, 0], y, c='b')
    # plt.scatter(x[:, 0], y_hat, c='r')  # 训练之前的效果。
    # plt.scatter(x[:, 0], model(x), c='g')  # 训练完成的效果。
    # plt.show()

    # 画三维图

    # 先画原图
    # 画模型曲面，预测
    n = 8
    x1 = np.linspace(-10, 10, n)
    x2 = np.linspace(-10, 10, n)
    # y = func((np.reshape([x1, x2], [2, len(x1)])).T)
    x1, x2 = np.meshgrid(x1, x2)

    True_W = [2, -3.4]
    True_b = 2.0
    y = True_W[0] * x1 + True_W[1] * x2 + True_b
    # plt.subplot(221)
    fig1 = plt.figure(1)
    ax1 = Axes3D(fig1)
    ax1.plot_surface(x1, x2, y, cmap="rainbow")
    # plt.show()

    W_hat = Ws[-1]  # model.W.numpy()
    b_hat = bs[-1]  # model.b.numpy()
    y_hat1 = W_hat[0] * x1 + W_hat[1] * x2 + b_hat
    fig2 = plt.figure(2)
    ax2 = Axes3D(fig2)

    ax2.plot_surface(x1, x2, y_hat1, cmap="rainbow")

    ax1.view_init(elev=0, azim=10)  # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
    ax2.view_init(elev=0, azim=30)  # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴

    plt.show()

    # 其他设置
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # # ax.plot_surface(x[:, 0], x[:, 1], y, c='b')
    # ax.scatter3D(x[:, 0], x[:, 1], y, c='b')
    # # ax.scatter3D(x[:, 0], x[:, 1], y_hat, c='r')
    # ax.scatter3D(x[:, 0], x[:, 1], model(x), c='g')
    # ax.set_xlabel("x1")
    # ax.set_ylabel("x2")
    # ax.set_zlabel("y")
    # ax.view_init(elev=0, azim=180)  # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
    # plt.show()
