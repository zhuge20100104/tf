# 利用鸢尾花数据集，实现前向传播，反向传播，可视化loss 曲线

from re import X
from tkinter import Y
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import time

## RmsProp优化器
## m(t) = g(t)  V(t)=a* V(t-1) + (1-a)*g(t)^2
# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# 使用相同的seed，保证输入特征和标签一一对应
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
np.random.seed(116)

# 将打乱后的数据集分为训练集和测试集，其中训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因为x的数据类型不一致而报错
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)

# 使用from_tensor_slices 生成训练标签集和测试标签集
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 使用tf.Variable 标记参数可训练
# 生成初始化权重和偏置，加seed是为了使结果一致，实际训练时不写seed
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

# 学习率初始化为 0.1
lr = 0.05
# loss列表
train_loss_results = []
# acc列表
test_acc = []
# 循环500轮
epoch = 500
# 每轮分4个step, loss_all记录4个step的和
loss_all = 0

v_w, v_b = 0, 0
beta = 0.9

# 训练部分
now_time = time.time()
# 数据集级别的循环，每个epoch循环一次数据集
for epoch_ in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            # 过softmax，使输出呈概率分布
            y = tf.nn.softmax(y)
            # 将标签值转换为独热码，方便计算 loss和 accuracy
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y - y_))
            loss_all += loss.numpy()
        
        grads = tape.gradient(loss, [w1, b1])
        v_w = beta * v_w + (1-beta) * tf.square(grads[0]) 
        v_b = beta * v_b + (1-beta) * tf.square(grads[1]) 
        # 更新梯度，梯度下降
        w1.assign_sub(lr* grads[0]/tf.sqrt(v_w))
        b1.assign_sub(lr* grads[1]/tf.sqrt(v_b))
    
    print("Epoch: {}, loss: {}".format(epoch_, loss_all/4))
    train_loss_results.append(loss_all/4)
    # loss_all清零，接着运行下一个epoch
    loss_all = 0

    # 预测部分
    # total_correct, 正确样本数， total_number总样本数
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        # 使用argmax返回最大分类索引
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    # 计算精度
    acc = total_correct/ total_number
    print("Epoch: {}, acc: {}".format(epoch_, acc))
    test_acc.append(acc)

total_time = time.time() - now_time
print("Total time: ", total_time)

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()

