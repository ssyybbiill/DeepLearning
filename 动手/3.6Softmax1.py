import tensorflow as tf
import numpy as np
import _3_5Fashion_MINIST as _3_5

print(tf.__version__)

from tensorflow.keras.datasets import fashion_mnist

num_inputs = 784  # num_inputs是特征数。num_examples 是样本数，没用到
num_outputs = 10  # num_outputs是输出/类别数。
batch_size = 256
num_epochs, lr = 5, 0.1


def get_fashion_minist_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # print(len(x_train))  # 60000
    # print(len(x_test))  # 10000
    # print(x_test[20])
    x_train = tf.cast(x_train, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
    x_test = tf.cast(x_test, tf.float32) / 255  # 把0-255之间的灰度值，变成0-1之间的值
    # print(x_test[20])
    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return x_train, y_train, x_test, y_test, train_iter, test_iter


def softmax(logits, axis=-1):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)


# # 这里的model是方法，其实可以弄成一个类的
# def net_model(X, W):
#     # 把x_train(样本数，28，28）reshap成一个X（样本数，特征数）
#     X = tf.reshape(X, shape=(-1, W.shape[0]))
#     logits = tf.matmul(X, W) + b
#     return softmax(logits)


# Model类，输出的是在各个类别上的概率分布probability
class Model(object):
    def __init__(self):
        # 随机初始化参数,W和b都是一个样本的，不是所有样本的
        # softmax回归的权重和偏差参数分别为784×10和1×10的矩阵。Variable来标注需要记录梯度的向量。
        self.W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
        self.b = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))

    def __call__(self, X):
        # W = tf.reshape(self.W, (num_inputs, num_outputs))  # W应该不需要reshape，验证，不需要可以删掉！！
        # 把x_train(样本数，28,28）reshap成一个X（样本数，特征数）
        X = tf.reshape(X, shape=(-1, self.W.shape[0]))
        logits = tf.matmul(X, self.W) + self.b
        probability_y = softmax(logits)
        return probability_y


# 交叉熵，应该是一个数，标量，不是向量。
# 是所有样本的平均，最后需要reduce_mean的。
def cross_entropy_loss_avg(y, y_hat):
    y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]), dtype=tf.int32)
    # return -tf.reduce_sum(y * tf.math.log(y_hat)) #你这种方式有点慢？其实我觉得也不慢呀，毕竟0乘一个数，应该算起来很简单吧？
    # 由于y只有0或1，直接采用boolean_mask(tensor,mask)，只计算y中为1时对应的log(y_hat)。mask既可以是True/False序列，也可以是0/1序列。
    # 为什么加上1e-8？为了不为0，因为log的定义域必须大于0。
    cross_entropy_loss_vector = -tf.math.log(tf.boolean_mask(y_hat, y) + 1e-8)  # 如果每个样本只属于一类的话，这个是shape=（样本数，）的一维向量
    return tf.reduce_mean(cross_entropy_loss_vector)  # 最终返回的，是所有样本的平均loss


# 《动手》原码中的交叉熵，应该是一个数，标量，不是向量。
# 是所有样本的loss和，最后需要reduce_sum的。是考虑了误差之后才用reduce_sum而非reduce_mean。
def cross_entropy_loss_sum(y, y_hat):
    y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]), dtype=tf.int32)
    cross_entropy_loss_vector = -tf.math.log(tf.boolean_mask(y_hat, y) + 1e-8)  # 如果每个样本只属于一类的话，这个是shape=（样本数，）的一维向量
    return tf.reduce_sum(cross_entropy_loss_vector)  # 最终返回的，是所有样本的loss总和


def accuracy(y_hat, y):
    # return tf.reduce_sum(tf.argmax(y_hat, axis=1) == y) / len(y_hat)
    # 这里利用了一个前提：索引正好是该类的编号y
    return np.mean((tf.argmax(y_hat, axis=1) == y))  # 直接除以长度不就好了嘛，装什么洋腔洋调的？！！不过话说回来，确实代码少！


# 类似accuracy，我们可以评价模型net在数据集data_iter上的准确率。
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y, dtype=tf.int64)
        y_hat = tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64)
        acc_sum += np.sum(y_hat == y)
        n += y.shape[0]
    return acc_sum / n  # 这次没有用mean![手动狗头]


# 分析半天，还是《动手》中的reduce_sum好用！唉！不过我乐意！
def train_ch3_me(model, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:  # 这里的X，y是一个batch_size=256的训练样本。
            with tf.GradientTape() as tape:
                probability_y = model(X)  # 概率分布probability
                current_loss = loss(y, probability_y)  # 这里应该是一个数，标量，数值是本batch_size=256的所有损失平均值
            grads = tape.gradient(current_loss, params)
            if trainer is None:
                # 如果没有传入优化器，则使用原先编写的小批量随机梯度下降
                for i, param in enumerate(params):
                    # params.assign_sub(lr * grads[i] / batch_size)  # 为什么除以batch_size？是因为《动手》原代码中的loss，用的是reduce_sum。
                    params.assign_sub(lr * grads[i])  # 你此处不用除以batch_size，是因为你的loss用的是reduce_mean！
            else:
                # tf.keras.optimizers.SGD 直接使用是随机梯度下降 theta(t+1) = theta(t) - learning_rate * gradient
                # 这里使用批量梯度下降，需要对梯度除以 batch_size, 对应原书代码的 trainer.step(batch_size)
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))
            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += current_loss.numpy() * batch_size  # current_loss是一个batch_size=256的平均值，而这里需要加整个batch_size的总loss，故乘以batch_size。
            # 因为最后一个batch是96<256，因此，乘以256会使得总loss偏大，由于每个epoch都是最后一个batch的loss偏大，这样累积下来，
            # 整个train_l_sum会偏大的比较多，【误差挺大的，而且体现在loss上，影响不小！因此不太合理！】
            y_hat = tf.argmax(probability_y, axis=1)  # 最大概率对应下标就是预测类y_hat。axis=1，说明是行最大值。

            # train_acc_sum是全部批次中预测正确的个数加一起，就是所有训练集中的正确个数
            train_acc_sum += tf.reduce_sum(  # 求和，也就是统计向量y和y_hat中，相等元素的个数，每个批次中预测正确的个数
                tf.cast(
                    # y既是下标号也是类别号
                    # 为什么要用argmax？因为softmax回归是一种分类方法，就是要输出最大y_hat(i)对应的那个下标i，也就是概率最大的类，即预测类别号！
                    # 《动手》原代码中，model/net的结果叫y_hat不太好，准确来说应该叫概率probability_y，tf.argmax的结果叫y_hat更好一些，因为这个才是与y对应的预测类别号。
                    y_hat == tf.cast(y, dtype=tf.int64)
                    , dtype=tf.int64)
            ).numpy()
            n += y.shape[0]  # 累计处理过的训练集样本数256，批次累加

        test_acc = evaluate_accuracy(test_iter, model)  # 在测试集上的准确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))  # 这里的loss是该epoch的平均loss。acc也是平均值。
    return 0


# 《动手》中的reduce_sum方法，更合理！
def train_ch3(model, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:  # 这里的X，y是一个batch_size=256的训练样本。
            with tf.GradientTape() as tape:
                probability_y = model(X)  # 概率分布probability
                current_loss = loss(y, probability_y)  # 这里应该是一个数，标量，数值是本batch_size=256的所有损失之和（当然也有96的）
            grads = tape.gradient(current_loss, params)
            if trainer is None:
                # 如果没有传入优化器，则使用原先编写的小批量随机梯度下降
                for i, param in enumerate(params):
                    params.assign_sub(lr * grads[i] / batch_size)  # 为什么除以batch_size？是因为这里用的是reduce_sum。
            else:
                # tf.keras.optimizers.SGD 直接使用是随机梯度下降 theta(t+1) = theta(t) - learning_rate * gradient
                # 这里使用批量梯度下降，需要对梯度除以 batch_size, 对应原书代码的 trainer.step(batch_size)
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))
            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += current_loss.numpy()  # current_loss就是整个batch_size的总loss，直接加就行
            y_hat = tf.argmax(probability_y, axis=1)  # 最大概率对应下标就是预测类y_hat。axis=1，说明是行最大值。

            # train_acc_sum是全部批次中预测正确的个数加一起，就是所有训练集中的正确个数
            train_acc_sum += tf.reduce_sum(  # 求和，也就是统计向量y和y_hat中，相等元素的个数，每个批次中预测正确的个数
                tf.cast(
                    # y既是下标号也是类别号
                    # 为什么要用argmax？因为softmax回归是一种分类方法，就是要输出最大y_hat(i)对应的那个下标i，也就是概率最大的类，即预测类别号！
                    # 《动手》原代码中，model/net的结果叫y_hat不太好，准确来说应该叫概率probability_y，tf.argmax的结果叫y_hat更好一些，因为这个才是与y对应的预测类别号。
                    y_hat == tf.cast(y, dtype=tf.int64)
                    , dtype=tf.int64)
            ).numpy()
            n += y.shape[0]  # 累计处理过的训练集样本数256，批次累加

        test_acc = evaluate_accuracy(test_iter, model)  # 在测试集上的准确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))  # 这里的loss是该epoch的平均loss。acc也是平均值。
    return 0


# 预测
# test_iter为什么会有元素？之前计算test_acc的时候没有耗尽iter资源吗？为什么还会有next()呢？
def prediction(model, test_iter):
    X, y = iter(test_iter).next()  # 这里的X，y是一个batch_size=256的训练样本。
    # X, y = next(iter(test_iter))
    # X, y = iter(test_iter).__next__()
    # print(iter(test_iter).next())
    true_labels = _3_5.get_fashion_mnist_labels(y.numpy())
    # print(len(true_labels)) # 256
    pred_labels = _3_5.get_fashion_mnist_labels(tf.argmax(model(X), axis=1).numpy())

    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    # _3_5.show_fashion_mnist(X[0:9], titles[0:9])
    _3_5.show_fashion_mnist(X[0:19], titles[0:19])  # 也能输出多个，但只能在一行上输出，画图函数有问题呀，怎么改？？？
    # _3_5.show_fashion_mnist(X, titles)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, train_iter, test_iter = get_fashion_minist_data()
    model = Model()
    y_hat_train = model(x_train)  # 最好不单独拿出来，如果用不到，数据就直接不传出x_train了！
    # CEL = cross_entropy_loss(y_train, y_hat_train)
    trainer = tf.keras.optimizers.SGD(lr)
    # train_ch3_me(model, train_iter, test_iter, cross_entropy_loss_avg, num_epochs, batch_size, [model.W, model.b], lr,
    #              trainer)
    train_ch3(model, train_iter, test_iter, cross_entropy_loss_sum, num_epochs, batch_size, [model.W, model.b], lr,
              trainer)

    prediction(model, test_iter)
