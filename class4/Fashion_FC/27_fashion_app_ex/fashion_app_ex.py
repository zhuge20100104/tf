from json import load
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

type = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model_save_path = 'fashion_model'

model = load_model(model_save_path)

pre_num = int(input('Pls input the number of test pictures:'))
for i in range(pre_num):
    image_path = input('The path of test picture:')
    img = Image.open(image_path)

    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)

    img = img.resize((28, 28), Image.ANTIALIAS)
    # 彩色图像转灰度，转为numpy array
    img_arr = np.array(img.convert('L'))
    # 图像反色
    img_arr = 255 - img_arr
    # 归一化
    img_arr = img_arr/ 255.0
    print('image array shape: ', str(img_arr.shape))

    # 现图形大小 (28, 28)
    # 在正前方加一维 (1, 28, 28)
    x_predict = img_arr[tf.newaxis, ...]
    print('x predict shape: ', str(x_predict.shape))
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    print('\n')

    print(type[int(pred)])
    plt.pause(5)
    plt.close() 

