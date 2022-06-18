import os 
import tensorflow as tf
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)= mnist.load_data()
# 特征向量归一化
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),  # 拍平成一个向量
    tf.keras.layers.Dense(128, activation='relu'), # 128个神经元的全连接层
    tf.keras.layers.Dense(10, activation='softmax') # 10个神经元的全连接层，输出结果使用softmax做概率分布
])

model.compile(optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/mnist.ckpt"
# 若模型存在检查点，加载已经训练的模型参数，进行断点续训
if(os.path.exists(checkpoint_save_path + ".index")):
    print('load the model')
    model.load_weights(checkpoint_save_path) 

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                save_weights_only=False,
                                                save_best_only=True)

# 喂数据，喂入训练集和验证集
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

model.summary()

# 显示训练集和验证集的acc和loss曲线
acc = history.history["sparse_categorical_accuracy"]
val_acc = history.history["val_sparse_categorical_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

