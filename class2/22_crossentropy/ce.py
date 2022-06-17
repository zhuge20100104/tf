import tensorflow as tf


# 交叉熵越小，表示P(x)与Q(x)的分布更加接近，可以通过反复训练Q(x)来使Q(x)的分布逼近P(x)
# 有关交叉熵的更多信息，请参考
# https://blog.csdn.net/b1055077005/article/details/100152102

# -(1*ln(0.6) + 0*ln(0.4)) = 0.5108256
loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
# -(1*ln(0.8) + 0*ln(0.2)) = 0.22314355
loss_ce2 = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])

print("loss_ce1:", loss_ce1)
print("loss_ce2:", loss_ce2)



