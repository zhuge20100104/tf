from PIL import Image
from matplotlib import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

weights_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(weights_save_path)

preNum = int(input('Input the number of test pictures: '))

for i in range(preNum):
    image_path = input('The path of the test picture: ')
    img = Image.open(image_path)
    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)

    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))
    img_arr = 255 - img_arr
    img_arr = img_arr/255.0
    x_predict = img_arr[tf.newaxis, ...]

    result = model.predict(x_predict)
    # 求取向量中值最大的一维的索引
    pred = tf.argmax(result, axis=1)

    print('\n')
    print(int(pred))
    plt.pause(5)
    plt.close()