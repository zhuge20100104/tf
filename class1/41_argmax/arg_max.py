import numpy as np
import tensorflow as tf

test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])

print("test: ", test)
# 每一列最大值的索引
print("每一列最大值的索引: ", tf.argmax(test, axis=0))
# 每一行最大值的索引
print("每一行最大值的索引: ", tf.argmax(test, axis=1))
