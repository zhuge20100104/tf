import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# 加入断点续训和检查点
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()

# 训练集和测试集归一化
x_train, x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = './checkpoint/fashion.ckpt'

if os.path.exists(checkpoint_save_path):
    print('load the model')
    model = load_model(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
            save_weights_only=False,
            save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1,
    callbacks=[cp_callback])

# 加载检查点目前保存的最好模型
model = load_model(checkpoint_save_path)
model.summary()
print(model.trainable_variables)
# 保存最好的权重到文件，便于查看
file = open('weights.txt', 'w')

for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()


