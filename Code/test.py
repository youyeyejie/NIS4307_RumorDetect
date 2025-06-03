import pandas as pd
from classify import RumourDetectClass

# 初始化接口实例
# from train_lstm import *
# model_parameter = f'128_256_30_0.009'
# model_path = f'../Output/Model/best_{model_parameter}.pt'
# vocab_path = f'../Output/Model/vocab_{model_parameter}.pkl'
# detector = RumourDetectClass(model_path, vocab_path)

# 初始化接口实例
detector = RumourDetectClass.construct_detector()


test_in = '../dataset/test/test_in.csv'
test_out = '../dataset/test/test_out.csv'

# 读取测试数据（假设test.csv包含'text'列）
test_in_data = pd.read_csv(test_in)
test_out_data = pd.read_csv(test_out)
# 合并测试数据
test_data = pd.concat([test_in_data, test_out_data], ignore_index=True)
test_texts = test_data['text'].tolist()

# 批量预测
predictions = []
for text in test_texts:
    # 调用classify接口（输入字符串，输出int类型0或1）
    pred = detector.classify(text)
    predictions.append(pred)

# 保存结果到DataFrame
test_data['pred_label'] = predictions

# 将预测结果分别保存回原来的in和out文件
id_pred_map = {
    row['id']: detector.classify(row['text']) 
    for _, row in test_data.iterrows()
}

# 3. 直接赋值（比 merge 简单 10 倍！）
test_in_data['pred_label'] = test_in_data['id'].map(id_pred_map)
test_out_data['pred_label'] = test_out_data['id'].map(id_pred_map)
test_in_data.to_csv(test_in, index=False)
test_out_data.to_csv(test_out, index=False)
print(f"预测完成，结果已分别保存至{test_in} 和 {test_out}")

# 计算混淆矩阵
TN = ((test_data['pred_label'] == 0) & (test_data['label'] == 0)).sum()
FP = ((test_data['pred_label'] == 1) & (test_data['label'] == 0)).sum()
FN = ((test_data['pred_label'] == 0) & (test_data['label'] == 1)).sum()
TP = ((test_data['pred_label'] == 1) & (test_data['label'] == 1)).sum()

print(f"|  预测\\真实  | 非谣言（0） | 谣言（1） |")
print(f"|-------------|-------------|-----------|")
print(f"| 非谣言（0） | {TN:<11} | {FN:<9} |")
print(f"| 谣言（1）   | {FP:<11} | {TP:<9} |")

total_in = len(test_in_data)
correct_in = (test_in_data['pred_label'] == test_in_data['label']).sum()
accuracy_in = correct_in / total_in

total_out = len(test_out_data)
correct_out = (test_out_data['pred_label'] == test_out_data['label']).sum()
accuracy_out = correct_out / total_out

total = len(test_data)
correct = (test_data['pred_label'] == test_data['label']).sum()
accuracy = correct / total
print(f"总预测准确率: {accuracy:.2%} ({correct}/{total})，同源准确率: {accuracy_in:.2%} ({correct_in}/{total_in})，异源准确率: {accuracy_out:.2%} ({correct_out}/{total_out})")