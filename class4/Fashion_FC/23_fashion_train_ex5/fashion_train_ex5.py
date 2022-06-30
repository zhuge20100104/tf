import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

np.set_printoptions(threshold=np.inf)

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['sparse_categorical_accuracy'])

model_save_path = "./fashion_model"

if(os.path.exists(model_save_path)):
    print('----------------------load the model---------------------------')
    model = load_model(model_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
        save_weights_only=False,
        save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1,
    callbacks=[cp_callback])

model.summary()

print(model.trainable_variables)
f = open('weights.txt', 'w')
for v in model.trainable_variables:
    f.write(str(v.name)+ '\n')
    f.write(str(v.shape)+ '\n')
    f.write(str(v.numpy())+ '\n')
f.close()

##### 显示训练集和验证集 的acc 和loss 曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Tranining Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()


