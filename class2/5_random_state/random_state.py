import numpy as np

# 伪随机数种子，每次都要重新初始化seed

rdm = np.random.RandomState(seed=1)
a = rdm.rand()
b = rdm.rand(2, 3)

print("a: ", a)
print("b: ", b)
