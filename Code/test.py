import pandas as pd
from classify import RumourDetectClass  # 假设接口类文件为classify.py，与当前脚本同目录

# 初始化接口实例
detector = RumourDetectClass()

# 读取测试数据（假设test.csv包含'text'列）
test_data = pd.read_csv('../Dataset/test/test.csv')
test_texts = test_data['text'].tolist()

# 批量预测
predictions = []
for text in test_texts:
    # 调用classify接口（输入字符串，输出int类型0或1）
    pred = detector.classify(text)
    predictions.append(pred)

# 保存结果到DataFrame
test_data['pred_label'] = predictions
test_data.to_csv('../Dataset/test/test_predictions.csv', index=False)
print("预测完成，结果已保存至test_predictions.csv")

expected = pd.read_csv('../Dataset/test/test_expected.csv')
total = len(expected)
correct = (test_data['pred_label'] == expected['label']).sum()
accuracy = correct / total
print(f"预测准确率: {accuracy:.2%} ({correct}/{total})")