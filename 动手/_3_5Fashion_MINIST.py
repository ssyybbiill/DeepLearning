import tensorflow as tf
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist


# 输出全部数组，不省略
# np.set_printoptions(threshold=10000000)  # threshold=np.nan报错，换成一个特别大的数就可以了

# 根据labels编号，获取对应labels的名称
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    rows = 1
    cols = len(images)
    print(cols)
    _, figs = plt.subplots(rows, cols, figsize=(12, 12))  # 这里注意subplot 和subplots 的区别，到底什么区别？？
    for f, img, lbl in zip(figs, images, labels):
        # f.imshow(img.reshape((28, 28)))
        # f.imshow(img.reshape((28, 28)).numpy())
        f.imshow(img)  # 3.6预测时，test_iter中的元素不能reshape怎么办呢？我就直接把reshape删掉，然后就可以了。。。

        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # 注意，TensorFlow数据集默认下载位置：C:\Users\xxx\.keras\datasets\fashion-mnist，
    # 在pycharm中下载不下来可以点击链接手动下载，存放到该位置即可。

    print(len(x_train), len(x_test))
    feature, label = x_train[0], y_train[0]
    print(feature.shape, feature.dtype)
    print(label, type(label), label.dtype)

    X, y = [], []
    for i in range(10):
        X.append(x_train[i])
        y.append(y_train[i])
    print(np.shape(X), np.shape(y))
    show_fashion_mnist(X, get_fashion_mnist_labels(y))

    batch_size = 256
    # print(sys.platform)  # 竟然是win32，不是64吗？
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

    start = time.time()

    i = 0
    for X, y in train_iter:
        i += 1
        print(i)
        print(X, y)
        continue
    print('%.2f sec' % (time.time() - start))
