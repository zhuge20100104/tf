import numpy as np

# 生成等间隔数值点
# 1 2   2  2.5 3.0 3.5 
# x:
# 1 1 1 1 
# 2 2 2 2

# y:
# 2  2.5 3.0 3.5 
# 2  2.5 3.0 3.5 

x, y = np.mgrid[1:3:1, 2:4:0.5]

# 将x,y 拉直，并合并为二维张量，生成二维坐标点
grid = np.c_[x.ravel(), y.ravel()]
print("x: \n", x)
print("y: \n", y)
print("x.ravel(): \n", x.ravel())
print("y.ravel(): \n", y.ravel())
print("grid: \n", grid)