# NIS4307 Rumour Detector

## Description
本项目是上海交通大学2024-2025春季学期 NIS4307 人工智能导论课程的期末项目，旨在利用自然语言处理技术构建一个谣言检测系统。该系统能够对社交媒体上的文本进行分析，识别潜在的谣言信息，并提供相应的可信度评分。

## Structure
```
rumour_detector/
├── Code/               # Python 代码
├── Dataset/            # 数据集
│   ├── split/          # 训练集、验证集
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
TODO
```

## Contributors
- [马悦钊](mailto:ma_yuezhao@sjtu.edu.cn)
- [李卓恒](mailto:lzhsj32206@sjtu.edu.cn)
- [刘梓芃](mailto:liuzipeng@sjtu.edu.cn)
- [聂鸣涛](mailto:niemingtao@sjtu.edu.cn)


## TODO

## Todo
- [ ] classify.py `lzp`
- [ ] 优化train_gru `nmt`
    - [ ] 超过MAX_LEN的文本处理：以固定步长（如 MAX_LEN/2）滑动窗口，每个块包含连续的 MAX_LEN 个 token？
    - [ ] 调参
- [ ] 数据集 `lzh`
- [ ] report `myz`