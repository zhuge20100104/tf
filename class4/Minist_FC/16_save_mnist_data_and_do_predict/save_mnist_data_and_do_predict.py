import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 特征向量归一化
x_train, x_test = x_train/255.0, x_test/255.0

# float64转float32
x_test = x_test.astype(np.float32)

# uint8转int32
y_test = y_test.astype(np.int32)

# 保存numpy 数据到本地
np.save('x_batch.npy', x_test)
np.save('y_label.npy', y_test)

# 这里load_model加载 模型，准备开始做predict
model = load_model("./mnist_model")

y_label = y_test[:12]
x_test = x_test[:12]

result = model.predict(x_test)
pred = tf.argmax(result, axis=1).numpy()
print("Predict: ", pred)
print("Actual:  ", y_label)

correct = int(tf.reduce_sum(tf.cast(tf.equal(pred, y_label), tf.int32)))
print("Correct: ", correct)
print("Total: ", 12)