import pandas as pd
from classify import RumourDetectClass

from train_lstm import *

test_path = '../dataset/test/test_in.csv'
predict_path = test_path.replace('.csv', '_predictions.csv')
expected_path = test_path.replace('.csv', '_expected.csv')

# 初始化接口实例
detector = RumourDetectClass.construct_detector()
detector = RumourDetectClass(model_path, vocab_path, EMBEDDING_DIM, HIDDEN_DIM, DEVICE)

# 读取测试数据（假设test.csv包含'text'列）
test_data = pd.read_csv(test_path)
test_texts = test_data['text'].tolist()

# 批量预测
predictions = []
for text in test_texts:
    # 调用classify接口（输入字符串，输出int类型0或1）
    pred = detector.classify(text)
    predictions.append(pred)

# 保存结果到DataFrame
test_data['pred_label'] = predictions
test_data.to_csv(predict_path, index=False)
print(f"预测完成，结果已保存至{predict_path}")

expected = pd.read_csv(expected_path)
total = len(expected)
correct = (test_data['pred_label'] == expected['label']).sum()
accuracy = correct / total
print(f"预测准确率: {accuracy:.2%} ({correct}/{total})")