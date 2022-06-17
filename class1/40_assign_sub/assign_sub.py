import tensorflow as tf

# 减1赋值方法
x = tf.Variable(4)
x.assign_sub(1)
print("x: ", x)