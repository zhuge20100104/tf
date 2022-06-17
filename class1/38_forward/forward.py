import tensorflow as tf

# shape 1 * 4
x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])
w1 = tf.constant([[-0.8, -0.34, -1.4],
                  [0.6, 1.3, 0.25],
                  [0.5, 1.45, 0.9],
                  [0.65, 0.7, -1.2]])

b1 = tf.constant([2.52, -3.1, 5.62])

y = tf.matmul(x1, w1) + b1
print("x1 shape: " , x1.shape)
print("w1 shape: " , w1.shape)
print("b1 shape: ", b1.shape)
print("y shape: " , y.shape)
print("y: ", y)

# 去掉为1 的维度
y_dim = tf.squeeze(y)
# 将输出值转换为0-1的概率值
# 方法是 e^xi/ (e^x1 + e^x2 + e^x3 + .... e^xn )  
y_pro = tf.nn.softmax(y_dim)

print("y_dim: ", y_dim)
print("y_pro: ", y_pro)
