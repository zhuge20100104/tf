# 画出红点和蓝点的分界线，也就是预测概率为 0.5的分界线
from re import X
import tensorflow as tf
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

# 读入数据标签，生成x_train, y_train
df = pd.read_csv("../dot.csv")
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = x_data.reshape(-1, 2)
y_train = y_data.reshape(-1, 1)

# 1是红色， 0 是蓝色
Y_c = [['red' if y else 'blue'] for y in y_train]

# 转换x的数据类型为tensorflow float类型，否则后面矩阵相乘会因数据类型不同而报错
x_train = tf.cast(x_train, dtype=tf.float32)
y_train = tf.cast(y_train, dtype=tf.float32)

# 创建Dataset
# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络的参数，输入层为两个神经元，隐藏层为11个神经元，共1个隐藏层，输出层为1个神经元

# 用tf.Variable保证参数可训练
w1 = tf.Variable(tf.random.normal([2,11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01,shape=[11]), dtype=tf.float32)

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]), dtype=tf.float32)

# 学习率
lr = 0.005
# 循环轮数
epoch = 800

# 训练部分
for epoch_ in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        print("X_train len: ", len(x_train))
        print(x_train)
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            # 使用均方误差损失函数
            loss_mse = tf.reduce_mean(tf.square(y_train - y))
        # 计算loss对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss_mse, variables)
        # 实现各个参数的梯度更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    if epoch_ % 20 == 0:
        print("epoch: ", epoch_, "loss: ", float(loss_mse))  

# 预测部分
print("***** predict *****")
xx, yy = np.mgrid[-3:3:0.1, -3:3:0.1]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, dtype=tf.float32)
# 将网格坐标送入神经网络进行预测
probs = []
for x_test in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_test], w1) + b1 
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2 
    probs.append(y)

# 画原数据中 对应点和坐标值
x1 = x_data[:, 0] 
x2 = x_data[:, 1]

probs = np.array(probs).reshape(xx.shape)

plt.scatter(x1, x2, color=np.squeeze(Y_c))

## 画出红蓝分界线，概率是0.5的线
plt.contour(xx, yy, probs, levels=[.5])
plt.show()
    




