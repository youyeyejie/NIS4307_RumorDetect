import pandas as pd
from classify import RumourDetectClass  # 假设接口类文件为classify.py，与当前脚本同目录
from classify import *
import argparse
from train_lstm import *
# model_path = f'../Output/Model/embedding_{EMBEDDING_DIM}_hidden_{HIDDEN_DIM}_epoch_{EPOCHS}.pt'
# test_path = '../Dataset/test/test.csv'


# parser = argparse.ArgumentParser()
# parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
# parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
# parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
# parser.add_argument('--lr', type=float, default=0.005, help='学习率')
# args = parser.parse_args()

# BATCH_SIZE = 64         # 批大小
# EMBEDDING_DIM = args.embedding_dim     # 嵌入维度
# HIDDEN_DIM = args.hidden_dim        # 隐藏层维度
# EPOCHS = args.epochs             # 训练轮数
# LEARNING_RATE = args.lr    # 学习率



model_parameter = f'{EMBEDDING_DIM}_{HIDDEN_DIM}_{EPOCHS}_{LEARNING_RATE}'
model_path = f'../Output/Model/{model_parameter}.pt'
vocab_path = f'../Output/Model/vocab_{model_parameter}.pkl'
test_path = '../dataset/test/test.csv'
graph_path = f'../Output/Graph/{model_parameter}.png'
predict_path = test_path.replace('.csv', '_predictions.csv')
expected_path = test_path.replace('.csv', '_expected.csv')

# 初始化接口实例
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