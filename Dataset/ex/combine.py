import pandas as pd

# 读取三个csv文件
# df1 = pd.read_csv('ex_train_1.csv')
# df2 = pd.read_csv('ex_train_2.csv')
df_all = pd.read_csv('../ex/data.csv')
df3 = pd.read_csv('../split/train.csv')
print(f"原始数据行数: {len(df_all)}")

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

# 指定event
event = ['gurlitt-all-rnr-threads', 'ferguson-all-rnr-threads', \
        'prince-toronto-all-rnr-threads', 'germanwings-crash-all-rnr-threads', \
        'sydneysiege-all-rnr-threads', 'ottawashooting-all-rnr-threads']

# 提取指定event的数据
# 按8:1:3划分df_in为df_train_in, df_val_in, df_test_in
df_in = df_all[df_all['event'].isin(event)].reset_index(drop=True)
df_train_in = df_in[df_in.index % 12 < 8]
df_val_in = df_in[df_in.index % 12 == 8]
df_test_in = df_in[df_in.index % 12 > 8]
print(f"同源总行数: {len(df_in)}，训练集行数: {len(df_train_in)}, 验证集行数: {len(df_val_in)}, 测试集行数: {len(df_test_in)}")

df_out = df_all[~df_all['event'].isin(event)].reset_index(drop=True)
# 按8:1:1划分df_out为df_train_out, df_val_out, df_test_out
df_train_out = df_out[df_out.index % 10 < 8]
df_val_out = df_out[df_out.index % 10 == 8]
df_test_out = df_out[df_out.index % 10 > 8]
print(f"异源总行数: {len(df_out)}，训练集行数: {len(df_train_out)}, 验证集行数: {len(df_val_out)}, 测试集行数: {len(df_test_out)}")

# 合并同源和异源数据
df_train = pd.concat([df_train_in, df_train_out], ignore_index=True)
df_val = pd.concat([df_val_in, df_val_out], ignore_index=True)

# 保存结果
df_train.to_csv('../split/ex_train.csv', index=False)
df_val.to_csv('../split/ex_val.csv', index=False)
df_test_in.to_csv('../test/test_in.csv', index=False)
df_test_out.to_csv('../test/test_out.csv', index=False)
print(f"训练集行数: {len(df_train)}")
print(f"验证集行数: {len(df_val)}")
print(f"同源测试集行数: {len(df_test_in)}")
print(f"异源测试集行数: {len(df_test_out)}")
print(f"总行数: {len(df_train) + len(df_val) + len(df_test_in) + len(df_test_out)}")
