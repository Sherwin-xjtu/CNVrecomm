import numpy as np

import numpy as np
import pandas

a = b = c = 1
if a == b == c:
    print(a,b,c)
exit()
dic = dict()

dic[a] = b
ma = {'nanosv':[1,2,2,1], 'picky':[2,4,5,6]}
df = pandas.DataFrame(ma)
t = df.loc[df['nanosv'].idxmax()]
print(t)
tols_metrics = dict()
tols_metrics['MSE'] = ['MSE']
tols_metrics['MSE'].append('RMSE')

print(tols_metrics)
exit()
data = [[0.4, 0.1, 0.5, 0.3, 0.45, 0.25],[0.4, 0.1, 0.5, 0.3, 0.45, 0.25],[0.4, 0.1, 0.5, 0.3, 0.45, 0.25],[0.4, 0.1, 0.5, 0.3, 0.45, 0.25]]
data2 = [0.4, 0.1, 0.5, 0.3, 0.45, 0.25]
print(2*data2)
print(np.array(data2),np.array(data).shape)
exit()
List3 = np.multiply(np.array(data),np.array(data2))
print(List3.tolist())
data1 = np.array(data)
print(len(data1))
exit()
# x = np.random.random(13611)
# print(x)
# exit()

def func1(amount, num):
    list1 = []
    total = 0
    for i in range(0, num - 1):
        a = np.random.random(amount)  # 生成 n-1 个随机节点
        list1.append(a.tolist()[0])
    list1.sort()  # 节点排序
    list1.append(amount)  # 设置第 n 个节点为amount，即总金额

    list2 = []
    for i in range(len(list1)):
        if i == 0:
            b = list1[i]  # 第一段长度为第 1 个节点 - 0
        else:
            b = list1[i] - list1[i - 1]  # 其余段为第 n 个节点 - 第 n-1 个节点
        list2.append(b)

    # print(list2)

    # for ele in range(0, len(list2)):
    #     total = total + list2[ele]
    li = list2
    maxnum = max(li)
    return maxnum
    # minnum = min(li)
    # li = [x for x in li if x != maxnum]
    # li.append(maxnum)
    # print(li)

    # print("列表元素之和为: ", total)


pro_li = []
for i in range(13611):
    pro_li.append(func1(1, 3))
print(pro_li)
exit()

x = np.random.random(10)
y = np.random.random(10)
print(x, type(x))
# 方法一：根据公式求解
d1 = np.sum(np.abs(x - y))

# 方法二：根据scipy库求解
from scipy.spatial.distance import pdist

X = np.vstack([x, y])
d2 = pdist(X, 'cityblock')
print(d2)
exit()


# 计算信息熵的方法
def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    print(x_value_list)
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    print(ent)


calc_ent(data1)
exit()
