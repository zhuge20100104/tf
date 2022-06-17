import tensorflow as tf

# 反向传播减小学习率的例子

w = tf.Variable(tf.constant(5, dtype=tf.float32))

epoch = 40
# 基础学习率
LR_BASE = 0.2 
# 学习率衰减比率
LR_DECAY = 0.99 
# 喂入多少轮 Batch size之后更新一次学习率
LR_STEP = 1

for epoch_ in range(epoch):
    # 学习率根据epoch_呈指数衰减
    lr = LR_BASE * LR_DECAY ** (epoch_/LR_STEP)
    with tf.GradientTape() as tape:
        # 平方函数
        loss = tf.square(w + 1)
        grads = tape.gradient(loss, w)
        # 更新权重
        # 沿着梯度下降的方向更新权重
        # w = w - lr*grads
        w.assign_sub(lr*grads)
    
    print("After %s epoch, w is %f, loss is %f, lr is %f" % (epoch_, w.numpy(), loss, lr))


