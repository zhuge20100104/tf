import tensorflow as tf
import numpy as np
import json


tfrecord_filename = '/tmp/train.tfrecord'
# 创建.tfrecord文件，准备写入
writer = tf.compat.v1.python_io.TFRecordWriter(tfrecord_filename)
for i in range(100):
    img_raw = np.random.random_integers(0,255,size=(30, 7)) # 创建30*7，取值在0-255之间随机数组
    img_raw = bytes(json.dumps(img_raw.tolist()), "utf-8")
    example = tf.compat.v1.train.Example(features=tf.train.Features(
            feature={
            # Int64List储存int数据
            'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[i])), 
            # 储存byte二进制数据
            'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw]))
            }))
    # 序列化过程
    writer.write(example.SerializeToString()) 
writer.close()