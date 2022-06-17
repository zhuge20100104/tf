import tensorflow as tf

import pandas as pd

fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def convert_dataset_to_csv(x_train_, y_train_):
    x_train_save = x_train_.reshape(-1, 28*28)
    print(x_train_save.shape)
    print(x_train_save)
    columns_arr = []
    for i in range(0, 784):
        columns_arr.append("train"+ str(i))
    df = pd.DataFrame(x_train_save, columns=columns_arr)
    df["label"] = y_train_
    return df

train_db = convert_dataset_to_csv(x_train, y_train)
test_db = convert_dataset_to_csv(x_test, y_test)
train_db.to_csv("minist_train.csv", index=False) 
test_db.to_csv("minist_test.csv", index=False)
