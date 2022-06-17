from tensorflow.python.ops import parsing_ops
import zlib
import tensorflow as tf

a = zlib.compress(b"abcd")

tf.compat.v1.disable_eager_execution()
in_bytes = tf.constant([a])
decode_op = parsing_ops.decode_compressed(
              in_bytes, compression_type="ZLIB")

sess = tf.compat.v1.Session()
print(sess.run(decode_op))
