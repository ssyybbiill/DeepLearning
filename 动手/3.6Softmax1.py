import tensorflow as tf
import numpy as np

print(tf.__version__)

from tensorflow.keras.datasets import fashion_mnist

num_inputs = 784  # num_inputs是特征数。num_examples = 1000是样本数，没用到
num_outputs = 10  # num_outputs是输出/类别数。
batch_size = 256


def get_fashion_minist_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_test[20])
    x_train = tf.cast(x_train, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
    x_test = tf.cast(x_test, tf.float32) / 255  # 把0-255之间的灰度值，变成0-1之间的值
    print(x_test[20])
    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return x_train, y_train, x_test, y_test, train_iter, test_iter


def softmax(logits, axis=-1):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)


# # 这里的model是方法，其实可以弄成一个类的
# def net_model(X, W):
#     # 把x_train(样本数，28,28）reshap成一个X（样本数，特征数）
#     X = tf.reshape(X, shape=(-1, W.shape[0]))
#     logits = tf.matmul(X, W) + b
#     return softmax(logits)


# Model类
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
        return softmax(logits)


def cross_entropy_loss(y, y_hat):
    y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]), dtype=tf.int32)
    # return -tf.reduce_sum(y * tf.math.log(y_hat)) #你这种方式有点慢？其实我觉得也不慢呀，毕竟0乘一个数，应该算起来很简单吧？
    # 由于y只有0或1，直接采用boolean_mask(tensor,mask)，只计算y中为1时对应的log(y_hat)。mask既可以是True/False序列，也可以是0/1序列。
    # 为什么加上1e-8？为了不为0，因为log的定义域必须大于0。
    return -tf.math.log(tf.boolean_mask(y_hat, y) + 1e-8)  # 如果每个样本只属于一类的话，返回的是个shape=（样本数，）的一维向量


def accuracy(y_hat, y):
    # return tf.reduce_sum(tf.argmax(y_hat, axis=1) == y) / len(y_hat)
    # 这里利用了一个前提：索引正好是该类的编号y
    return np.mean((tf.argmax(y_hat, axis=1) == y))  # 直接除以长度不就好了嘛，装什么洋腔洋调的？！！不过话说回来，确实代码少！


# 类似accuracy，我们可以评价模型net在数据集data_iter上的准确率。
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y, dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n  # 这次没有用mean![手动狗头]


def train_ch3(model, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = model(X)
                current_loss = cross_entropy_loss(y, y_hat)
            grads = tape.gradient(current_loss, params)
            if trainer is None:
                # 如果没有传入优化器，则使用原先编写的小批量随机梯度下降
                for i, param in enumerate(params):
                    params.assign_sub(lr * grads[i] / batch_size)  # 为什么除以batch_size，不理解，批量梯度下降？
            else:
                # tf.keras.optimizers.SGD 直接使用是随机梯度下降 theta(t+1) = theta(t) - learning_rate * gradient
                # 这里使用批量梯度下降，需要对梯度除以 batch_size, 对应原书代码的 trainer.step(batch_size)
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))
            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += current_loss.numpy()
            train_acc_sum += tf.reduce_sum(
                tf.cast(
                    # y既是下标号也是类别号
                    # 为什么要用argmax？因为softmax回归是一种分类方法，就是要输出最大y_hat(i)对应的那个下标i，也就是概率最大的类，即预测类别号！
                    tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64)
                    , dtype=tf.int64)
            ).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, model)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    return 0


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, train_iter, test_iter = get_fashion_minist_data()
    model = Model()
    y_hat_train = model(x_train)  # 最好不单独拿出来，如果用不到，数据就直接不传出x_train了！
    CEL = cross_entropy_loss(y_train, y_hat_train)
    num_epochs, lr = 5, 0.1
    trainer = tf.keras.optimizers.SGD(lr)
    train_ch3(model, train_iter, test_iter, CEL, num_epochs, batch_size, [model.W, model.b], lr, trainer)
