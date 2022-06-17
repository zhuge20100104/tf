from sklearn import datasets

from pandas import DataFrame

import pandas as pd

# 返回iris 数据集的所有特征
x_data = datasets.load_iris().data
# 返回iris 数据集的所有标签
y_data = datasets.load_iris().target

print("x_data from datasets: \n ", x_data)
print("y_data from datasets: \n", y_data)

# 为表格增加行所有和列标签
x_data = pd.DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])

print(x_data.shape)
print(x_data)

# 设置列名对齐
pd.set_option("display.unicode.east_asian_width", True)

print("x_data add index: \n", x_data)

# 新加一列，列标签为类别，数据为 y_data
x_data["类别"] = y_data
print("x_data add a column: \n", x_data)

# 将iris数据集写入文件
x_data.to_csv("../iris.csv", index=True)


