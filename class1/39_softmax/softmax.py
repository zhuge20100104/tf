import tensorflow as tf

y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)

print("After softmax, y_pro is: ", y_pro)
# 所有概率的和加起来等于全样本空间的值
print("The sum of y_pro: ", tf.reduce_sum(y_pro))