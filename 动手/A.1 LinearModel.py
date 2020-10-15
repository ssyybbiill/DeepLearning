import tensorflow as tf
import matplotlib.pyplot as plt  # from matplotlib import pyplot as plt


# 只利用Tensor和GradientTape来实现一个线性回归
# 1.获取训练数据
# 首先，通过向输入中添加随机高斯（Normal）噪声来合成训练数据：
def getData():
    True_W = 3.0
    True_b = 2.0
    NUMBER_EXAMPLES = 10
    x = tf.random.normal(shape=[NUMBER_EXAMPLES])
    noise = tf.random.normal(shape=[NUMBER_EXAMPLES])
    y = x * True_W + True_b + noise
    return x, y


# 2.定义线性模型。
# 让我们定义一个简单的类来封装变量和计算：
class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


# 3.定义损失函数。
# 损失函数衡量给定输入的模型输出与目标输出的匹配程度。目的是在训练过程中尽量减少这种差异。
# 使用标准的L2损失MSE，也称为最小平方误差、最小均方误差：
def loss(target_y, predicted_y):
    # n = len(target_y)
    # n = 1
    # return (predicted_y - target_y) ** 2 / n
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# 4.定义训练循环
# 利用网络和训练数据，使用梯度下降训练模型以更新权重变量（W）和偏差变量（b）以减少损失。
# 我们tf.train.Optimizer推荐的实现中包含了梯度下降方案的许多变体。
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss(outputs, model(inputs))
    dW, db = tape.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


if __name__ == '__main__':
    # 5.最后，我们通过反复训练数据运行的，看看W和b发展。
    x, y = getData()
    model = Model()
    y_hat = model(x)

    Ws, bs = [], []  # 记录w和b的历史结果。
    epochs = range(10)
    for epoch in epochs:
        print("before:", Ws)
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        print("later:", Ws)
        # print(model(x))
        current_loss = loss(y, model(x))  # 这里使用model(x)，而不是y_hat，是因为model不断变化,loss也变化。而y_hat是一个固定值。
        train(model, x, y, learning_rate=0.1)
        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" % (epoch, Ws[-1], bs[-1], current_loss))

    # 画图
    plt.scatter(x, y, c='b')
    plt.scatter(x, y_hat, c='r')  # 训练之前的效果。
    plt.scatter(x, model(x), c='g')  # 训练完成的效果。
    plt.show()

    #
    #
    #
    # print(model(3))
    # print(model(3).numpy())  # 把tensor张量类型，转换为数值类型。
    # assert model(3.0).numpy() == 15.0
