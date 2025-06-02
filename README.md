# NIS4307 Rumour Detector

## Description
本项目是上海交通大学2024-2025春季学期 NIS4307 人工智能导论课程的期末项目，旨在利用自然语言处理技术构建一个谣言检测系统。该系统能够对社交媒体上的文本进行分析，识别潜在的谣言信息，并提供相应的可信度评分。

## Structure
```
rumour_detector/
├── Code/               # Python 代码
├── Dataset/            # 数据集
│   ├── split/          # 训练集、验证集
│   ├── ex/             # 额外训练集、验证集
│   └── test/           # 测试集
├── Docs/               # 项目文档
├── Output/             # 输出
│   ├── Graph/          # 图表
│   └── Model/          # 模型
├── Report/             # 项目报告
├── environment.yml     # Conda 环境配置文件
└── README.md           # 项目说明文件
```

## Environment
本项目使用如下环境进行开发和测试：
- Python 3.8
- PyTorch 1.8.2
- CUDA 11.1.1

使用 `Conda` 进行环境管理，环境配置文件 `environment.yml` 已包含所有依赖项。可以使用以下命令创建和激活环境：
```bash
conda env create -f environment.yml
conda activate NIS4307
```

## Interface
本项目提供调用接口类文件 [`classify.py`](Code/classify.py)，可以直接调用该文件中的 `RumourDetectClass` 类进行谣言检测。该类的使用方法如下：

```python
# 导入 RumourDetectClass
from classify import RumourDetectClass
# 初始化接口实例
detector = RumourDetectClass.construct_detector()
# 或者使用指定的模型和词汇表路径
# from train_lstm import *
# detector = RumourDetectClass(model_path, vocab_path, EMBEDDING_DIM, HIDDEN_DIM, DEVICE)
# 调用 classify 方法进行预测
pred = detector.classify("待检测的文本内容")
# pred 将返回一个整数，0 表示非谣言，1 表示谣言
```
批量预测示例:
```python
import pandas as pd
from classify import RumourDetectClass

# from train_lstm import *
# model_parameter = f'{EMBEDDING_DIM}_{HIDDEN_DIM}_{EPOCHS}_{LEARNING_RATE}'
# model_path = f'../Output/Model/{model_parameter}.pt'
# vocab_path = f'../Output/Model/vocab_{model_parameter}.pkl'

test_path = '../dataset/test/test_in.csv'
predict_path = test_path.replace('.csv', '_predictions.csv')
expected_path = test_path.replace('.csv', '_expected.csv')

# 初始化接口实例
detector = RumourDetectClass.construct_detector()
# detector = RumourDetectClass(model_path, vocab_path, EMBEDDING_DIM, HIDDEN_DIM, DEVICE)

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
```

## Contributors
- [马悦钊](mailto:ma_yuezhao@sjtu.edu.cn)
- [李卓恒](mailto:lzhsj32206@sjtu.edu.cn)
- [刘梓芃](mailto:liuzipeng@sjtu.edu.cn)
- [聂鸣涛](mailto:niemingtao@sjtu.edu.cn)

<!-- ## RUN

```bash
# 训练模型
python train_lstm.py --embedding_dim 128 --hidden_dim 256 --epochs 20 --lr 0.005
# 调用接口 后面为模型参数
python classify.py --embedding_dim 128 --hidden_dim 256 --epochs 20 --lr 0.005
``` -->

