import tensorflow as tf

x = tf.constant([[1, 2, 3], [2, 2, 3]])
print("x: ", x)

# 求x中所有数字的平均值
print("mean of x: ", tf.reduce_mean(x))

# axis=1 对行求和
print("sum of x: ", tf.reduce_sum(x, axis=1))