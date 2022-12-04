import pandas as pd


Student_dict = {'姓名':['张三', '李四', '王五', '赵六'],
                '性别':['男', '女', '男', '女'],
                '年龄':[20, 21, 19, 18],
                'Python成绩':[70, 80, 90, 50],
                '评价':['良好', '良好', '良好', '良好'],
                '地址':['A小区10幢', 'A小区11幢','B小区10幢','C小区11幢']}


# 字典创建DataFrame，字典键变DataFrame的列名
df = pd.DataFrame(data=Student_dict, index=['a','b','c','d'])
del df['姓名']
print(df)
print(df.loc[(df['Python成绩'] > 75) | (df['年龄'] > 20)] )
# print(df)