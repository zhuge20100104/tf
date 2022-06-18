import tensorflow as tf

from matplotlib import pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 元素数不变，每一个元素变成 28*28*1
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# 图像增强器
image_gen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=False,
    zoom_range=0.5
)

image_gen_train.fit(x_train)

# 获取前12张图片，原图，去掉最后一维,所以数组大小为 [12, 28, 28]
# np.squeeze的作用是去掉最后一维为1 的维度
# 这一波是用来显示的原始图像
x_train_subset1 = np.squeeze(x_train[:12])

# 这一波是用来增强的图像，数组大小为 [12, 28, 28, 1]
# 增强完了之后，再去掉最后一维显示
x_train_subset2 = x_train[:12]

fig = plt.figure(figsize=(20, 2))
plt.set_cmap("gray")

# 显示原始图片
for i in range(0, len(x_train_subset1)):
    ax = fig.add_subplot(1, 12, i+1)
    ax.imshow(x_train_subset1[i])
fig.suptitle("Subset of Original Training Images", fontsize=20)
plt.show()

fig = plt.figure(figsize=(20, 2))
# 只显示第一个批次的前12张，而且不做shuffle
for x_batch in image_gen_train.flow(x_train_subset2, batch_size=12, shuffle=False):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i+1)
        # 去掉最后一维，好做展示
        ax.imshow(np.squeeze(x_batch[i]))
    fig.suptitle("Augmented Images", fontsize=20)
    plt.show()
    break
