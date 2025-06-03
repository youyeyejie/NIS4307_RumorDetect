import pandas as pd
from classify import RumourDetectClass

# 初始化接口实例
# from train_lstm import *
# model_parameter = f'128_256_30_0.012'
# model_path = f'../Output/Model/{model_parameter}.pt'
# vocab_path = f'../Output/Model/vocab_{model_parameter}.pkl'
# detector = RumourDetectClass(model_path, vocab_path)

# 初始化接口实例
detector = RumourDetectClass.construct_detector()


test_path = '../dataset/test/test_in.csv'

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
test_data.to_csv(test_path, index=False)
print(f"预测完成，结果已保存至{test_path}")

total = len(test_data)
correct = (test_data['pred_label'] == test_data['label']).sum()
accuracy = correct / total
print(f"预测准确率: {accuracy:.2%} ({correct}/{total})")