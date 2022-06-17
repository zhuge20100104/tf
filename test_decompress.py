from tensorflow.python.ops import parsing_ops
import zlib
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

a = zlib.compress(b"abcd")

tf.compat.v1.disable_eager_execution()

in_bytes = array_ops.placeholder(dtypes.string, shape=[1])
decode_op = parsing_ops.decode_compressed(
              in_bytes, compression_type="ZLIB")

sess = tf.compat.v1.Session()
print(sess.run(decode_op, feed_dict={in_bytes: [a]}))
