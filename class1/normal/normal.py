import tensorflow as tf

d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d: ", d)

# 离均值超过两倍的标准差之外的数据会被去掉
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e: ", e)