import tensorflow as tf
import numpy as np

SEED = 23455
## 自定义损失函数
# 酸奶成本1元，利润 99元
# 成本很低，利润很高
# 人们希望多预测些，生成模型系数大于1，往多了预测
COST = 1
PROFIT = 99
rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32, 2)
# y_ 实际值 加上 噪声 [-0.05, 0.05)
y_ = [[x1+x2 + (rdm.rand()/10.0 - 0.05)] for (x1, x2) in x] 

x = tf.cast(x, dtype=tf.float32)
w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 10000
lr = 0.002

for epoch_ in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        # 自定义损失函数
        # 估多了，损失的是成本，估少了损失的是利润
        loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*COST, (y_-y)*PROFIT))
    grads = tape.gradient(loss, w1)
    w1.assign_sub(lr*grads)
    if epoch_ % 500 == 0:
        print("After %d training steps, w1 is: \n" % (epoch_))
        print(w1.numpy(), "\n")

print("Final w1 is: \n", w1.numpy())


