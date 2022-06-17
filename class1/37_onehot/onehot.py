import tensorflow as tf

# 3分类的onehot向量输出
classes = 3 
labels = tf.constant([1, 0, 2])
output = tf.one_hot(labels, depth=classes)
print("result of labels: ", output)