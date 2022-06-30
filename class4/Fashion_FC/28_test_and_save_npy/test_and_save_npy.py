from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# 测试集 归一化
x_test = x_test/ 255.0

x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.int32)

print('Test data length: ', str(x_test.shape[0]))
# Save file to npy files
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

model_save_path = './fashion_model'
model = load_model(model_save_path)


result = model.predict(x_test)
pred = tf.argmax(result, axis=1)

correct_num = tf.reduce_sum(tf.cast(tf.equal(pred, y_test), tf.float32))

print("Accuracy: ", str(float(correct_num*1.0 / y_test.shape[0]*100)), "%")
