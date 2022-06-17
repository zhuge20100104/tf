import tensorflow as tf

import numpy as np

SEED = 23455
# 生成0-1之间的随机数
rdm = np.random.RandomState(seed=SEED)
# 32行两列的输入矩阵
x = rdm.rand(32, 2)
# 生成随机的标签值，用于预测
# 噪声大小为 [0,0.1) - 0.05 = [-0.05, 0.05)
y_= [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]

x = tf.cast(x, dtype=tf.float32)

# 两行一列的初始权重值
w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 15000
lr = 0.002

for epoch_ in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        # 均方误差
        loss_mse = tf.reduce_mean(tf.square(y_ - y))
    
    grad = tape.gradient(loss_mse, w1)
    w1.assign_sub(lr * grad)

    if epoch_ % 500 == 0:
        print("After %d training steps, w1 is: \n", epoch_)
        print(w1.numpy(), "\n")
    
print("Final w1 is : ", w1.numpy())



