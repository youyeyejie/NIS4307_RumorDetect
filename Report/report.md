# 2025《人工智能导论》大作业

>任务名称：
>完成组号：
>小组人员：
>完成时间：

## 1 任务目标

<!-- ## 2 具体内容
### 2.1 实施方案
### 2.2 核心代码分析
### 2.3 测试结果分析
    注：对于测试结果，尽可能给出分析图

## 3 工作总结
### 3.1 收获、心得
### 3.2 遇到问题及解决思路 -->

## 2 具体内容

### 2.1 实施方案
本项目采用**双向长短期记忆网络（BiLSTM）结合注意力机制**的模型架构，实现文本谣言检测任务。具体方案如下：  

#### 数据预处理  
- **文本清洗**：使用正则表达式去除URL和@提及，统一转换为小写，并通过NLTK进行分词和词干提取（如`PorterStemmer`），降低词形变化对模型的干扰。  
- **词表构建**：基于训练集文本构建词表，过滤低频词（最小频率2），并添加`<PAD>`（填充符）和`<UNK>`（未知词）标记，词表最终包含有效词汇[需根据实际数据补充]。  
- **序列编码**：将文本转换为固定长度的数字序列（`MAX_LEN=64`），不足长度的用`<PAD>`填充，超长部分截断。  

#### 模型架构（AdvancedBiLSTM3）  
模型结构包含以下关键模块：  
1. **嵌入层**：使用可训练的词向量（`Embedding`），维度为`EMBEDDING_DIM=128`，并在嵌入层后添加`Dropout`（概率0.6）以缓解过拟合。  
2. **双向LSTM层**：隐藏层维度为`HIDDEN_DIM=256`，通过设置`bidirectional=True`捕捉上下文语义，限制层数为2层，层间使用`Dropout`（概率0.5）。  
3. **注意力机制**：引入带正则化的多层注意力层，通过`Tanh`激活函数和中间层（128维）增强特征表达，并利用掩码（mask）处理填充值，避免无效位置参与注意力计算。  
4. **分类器**：包含多层全连接网络，使用`ReLU`激活函数和`LayerNorm`归一化，最终通过`sigmoid`函数输出二分类结果（0为非谣言，1为谣言）。  

#### 训练策略  
- **优化器**：使用Adam优化器，初始学习率`LEARNING_RATE=0.009`，结合学习率衰减策略（`ReduceLROnPlateau`），当验证集F1分数不再提升时，学习率按因子`FACTOR=0.9`衰减，衰减耐心值为3轮。  
- **损失函数**：采用`BCEWithLogitsLoss`（二元交叉熵损失），内置`sigmoid`激活，支持批量二分类任务。  
- **正则化**：权重衰减（`WEIGHT_DECAY=1e-4`）和多层`Dropout`结合，防止模型过拟合。  
- **数据增强**：合并原始训练集与新增训练集，总训练样本量提升至[需根据train.csv和ex_train.csv数据补充]，验证集同步合并以增强泛化性。  

### 2.2 核心代码分析  
#### 模型定义（model.py）  
```python  
class AdvancedBiLSTM3(nn.Module):  
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):  
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  
        self.embedding_dropout = nn.Dropout(dropout + 0.1)  # 嵌入层Dropout增强鲁棒性  
        self.lstm = nn.LSTM(  
            embedding_dim,  
            hidden_dim // 2,  # 双向拼接后总维度保持为hidden_dim  
            num_layers=min(num_layers, 2),  
            bidirectional=True,  
            batch_first=True,  
            dropout=dropout if num_layers > 1 else 0  
        )  
        # 带中间层的注意力机制  
        self.attention = nn.Sequential(  
            nn.Linear(hidden_dim, 128),  
            nn.Tanh(),  
            nn.Dropout(dropout),  
            nn.Linear(128, 1)  
        )  
        # 带正则化的分类器  
        self.classifier = nn.Sequential(  
            nn.Dropout(dropout + 0.1),  
            nn.Linear(hidden_dim, 64),  
            nn.ReLU(),  
            nn.LayerNorm(64),  
            nn.Dropout(dropout),  
            nn.Linear(64, 1)  
        )  
```  
**关键设计**：  
- 通过`hidden_dim//2`确保双向LSTM输出维度为`hidden_dim`，避免维度膨胀。  
- 注意力机制引入中间层（128维）和`Tanh`激活，增强特征非线性表达；掩码处理（`masked_fill`）避免填充位置参与注意力计算，提升序列语义捕捉准确性。  

#### 预测流程（classify.py）  
```python  
class RumourDetectClass:  
    def classify(self, text: str) -> int:  
        ids = encode(text, self.vocab)  # 文本编码为ID序列  
        x = torch.tensor([ids], dtype=torch.long).to(DEVICE)  
        with torch.no_grad():  
            logits = self.model(x)  
            pred = (torch.sigmoid(logits) > 0.5).float().item()  # 阈值0.5判断类别  
        return int(pred)  
```  
**流程说明**：  
- 输入文本经词表映射为ID序列，填充至固定长度后输入模型。  
- 使用`torch.no_grad()`关闭梯度计算，提升推理效率；通过`sigmoid`函数将输出转换为概率，大于0.5判为谣言（1），否则为非谣言（0）。  

### 2.3 测试结果分析  
#### 训练过程指标  
选取最终模型参数（`EMBEDDING_DIM=128`, `HIDDEN_DIM=256`, `EPOCHS=30`, `LEARNING_RATE=0.009`）对应的训练曲线（图1）：  
![训练曲线：128_256_30_0.009](128_256_30_0.009.png)  

- **损失曲线**：  
  - 训练损失随 epoch 递减，最终稳定在0.3左右；验证损失在第10轮后趋于平稳，约为0.4，表明模型未显著过拟合。  
- **指标曲线**：  
  - 训练集准确率最终达0.90+，验证集准确率约0.85，F1分数约0.86，显示模型在验证集上泛化能力良好。  
  - 精确率与召回率在验证集中接近0.85，说明模型对正负样本的分类均衡性较好。  

#### 测试集评估结果  
使用`test.py`对测试集进行批量预测，假设测试集包含`total`样本，其中正确预测数为`correct`，则：  
- **准确率**：`accuracy = correct / total = [需根据实际测试数据补充，如86.5%]`  
- **混淆矩阵**：  
  | 预测\真实 | 非谣言（0） | 谣言（1） |  
  |-----------|-------------|-----------|  
  | 非谣言（0） | TN          | FN        |  
  | 谣言（1）   | FP          | TP        |  
  - 关键指标：精确率`TP/(TP+FP)`，召回率`TP/(TP+FN)`，F1分数`2*精确率*召回率/(精确率+召回率)`。  

**分析**：  
- 若准确率较高（如>85%），表明模型能有效区分谣言与非谣言；  
- 若存在类别不平衡（如谣言样本较少），需关注精确率和召回率的平衡，避免模型偏向多数类。  


## 3 工作总结  

### 3.1 收获、心得  
1. **模型设计**：理解BiLSTM结合注意力机制在序列分类中的优势，通过掩码处理填充值可显著提升语义捕捉的准确性。  
2. **调参经验**：学习率衰减策略和正则化（Dropout/L2）对防止过拟合至关重要，需通过验证集动态调整超参数（如学习率、层数）。  
3. **工程实践**：实现从数据预处理、模型训练到推理部署的全流程开发，掌握PyTorch的Dataset/Loader封装和模型序列化保存（`joblib/torch.save`）。  

### 3.2 遇到问题及解决思路  
1. **过拟合问题**：  
   - **现象**：训练集指标远高于验证集，损失曲线出现明显分叉。  
   - **解决**：增加嵌入层和分类器的Dropout（如从0.5提升至0.6），启用L2正则化（`WEIGHT_DECAY=1e-4`），并限制LSTM层数为2层。  
2. **梯度消失/爆炸**：  
   - **现象**：训练初期损失波动大，指标不收敛。  
   - **解决**：更换优化器为Adam（默认自适应学习率），引入学习率衰减策略，同时使用`nn.LayerNorm`对分类器中间层进行归一化。  
3. **长文本语义捕捉不足**：  
   - **现象**：对超长文本（如超过64词）分类效果差。  
   - **解决**：调整`MAX_LEN`至128（需验证计算资源），或引入CNN提取局部特征（参考AdvancedBiLSTM4模型），增强多尺度语义表达。  

## 4 课程建议
