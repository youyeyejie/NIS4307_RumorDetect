import pandas as pd

# 读取三个csv文件
# df1 = pd.read_csv('ex_train_1.csv')
# df2 = pd.read_csv('ex_train_2.csv')
df_all = pd.read_csv('../ex/data.csv')
df3 = pd.read_csv('../split/train.csv')

# 合并 ex_train_1 和 ex_train_2
# df_all = pd.concat([df1, df2], ignore_index=True)
# print(f"合并后的数据行数: {len(df_all)}")

# 去除 id 在 df3 中已存在的条目
df_all = df_all[~df_all['id'].isin(df3['id'])]

# 去重
df_all = df_all.drop_duplicates('id', keep='first')

# 排序
df_all = df_all.sort_values('id')

# 输出合并后的数据行数
print(f"去重和排序后的数据行数: {len(df_all)}")

# 指定以这些数字开头的id前缀
prefixes = ['524', '544', '580', '581', '529', '499', '500', '498', '536']

# 提取以指定前缀开头的条目到df_test
df_test_in = df_all[df_all['id'].astype(str).str.startswith(tuple(prefixes))]

# 剩余的为df_train
df_train = df_all[~df_all['id'].isin(df_test_in['id'])]

# 从df_train中，行号mod8为1的移动到df_val，mod8为2的移动到df_test_out
df_train = df_train.reset_index(drop=True)
df_val = df_train[df_train.index % 8 == 1]
df_test_out = df_train[df_train.index % 8 == 2]
df_train = df_train[~df_train.index.isin(df_val.index) & ~df_train.index.isin(df_test_out.index)]

# 保存结果
df_train.to_csv('../split/ex_train.csv', index=False)
df_val.to_csv('../split/ex_val.csv', index=False)
df_test_in.to_csv('../test/test_in_expected.csv', index=False)
df_test_out.to_csv('../test/test_out_expected.csv', index=False)
print(f"训练集行数: {len(df_train)}")
print(f"验证集行数: {len(df_val)}")
print(f"同源测试集行数: {len(df_test_in)}")
print(f"异源测试集行数: {len(df_test_out)}")
print(f"总行数: {len(df_train) + len(df_val) + len(df_test_in) + len(df_test_out)}")

# 删除测试集的label列并保存
df_test_in = df_test_in.drop(columns=['label'])
df_test_out = df_test_out.drop(columns=['label'])
df_test_in.to_csv('../test/test_in.csv', index=False)
df_test_out.to_csv('../test/test_out.csv', index=False)
