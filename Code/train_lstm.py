import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from model import AdvancedBiLSTM3 as AdvancedBiLSTM
import time
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 超参数设置
BATCH_SIZE = 32         # 批大小
EMBEDDING_DIM = 128     # 嵌入维度(可修改)
HIDDEN_DIM = 256        # 隐藏层维度(可修改)
EPOCHS = 30             # 训练轮数(可修改)
MAX_LEN = 64            # 文本最大长度
LEARNING_RATE = 0.9e-2  # 学习率(可修改)
FACTOR = 0.9            # 学习率衰减因子
WEIGHT_DECAY = 1e-4     # L2正则化
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备选择

# 路径设置
# embedding_dim hidden_dim epochs learning_rate
model_parameter = f'{EMBEDDING_DIM}_{HIDDEN_DIM}_{EPOCHS}_{LEARNING_RATE}'
model_path = f'../Output/Model/{model_parameter}.pt'
vocab_path = f'../Output/Model/vocab_{model_parameter}.pkl'
train_path = '../Dataset/split/train.csv'
ex_train_path = '../Dataset/split/ex_train.csv'  # 新增训练集1
val_path = '../dataset/split/val.csv'
ex_val_path = '../dataset/split/ex_val.csv'
test_path = '../dataset/test/test.csv'
graph_path = f'../Output/Graph/{model_parameter}.png'

# 简单分词器
# def tokenize(text):
#     return re.findall(r'\w+', text.lower())

def tokenize(text):
    # 处理URL和@提及
    text = re.sub(r'http\S+', '<URL>', text)
    text = re.sub(r'@\w+', '@USER', text)
    
    # 使用NLTK分词+词干提取
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(w.lower()) for w in tokens if w.isalpha()]

# 构建词表
def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = idx
            idx += 1
    return vocab

def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab['<PAD>']] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids

class RumorDataset(Dataset):
    # 谣言数据集，返回文本和标签
    def __init__(self, df, vocab):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

    
def evaluate(model, loader):
    # 评估函数，返回准确率、精确率、召回率和F1分数
    model.eval()
    all_preds = []
    all_labels = []
    epoch_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = nn.BCEWithLogitsLoss()(logits, y)
            epoch_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        avg_loss = epoch_loss / len(loader)
    
    # 计算各项指标
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return accuracy, precision, recall, f1, avg_loss

def plot_learning_curve(train_metrics, val_metrics, epochs, save_path):
    """绘制包含损失率和核心指标的双图学习曲线"""
    plt.figure(figsize=(16, 8))
    plt.suptitle(f'BiLSTM - {model_parameter}', fontsize=16, y=0.95)  # 调整总标题位置

    # 子图1：损失率曲线（训练集/验证集）
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_metrics['loss'], marker='o', color='blue', linestyle='-', label='Training Loss')
    plt.plot(range(1, epochs+1), val_metrics['loss'], marker='s', color='orange', linestyle='--', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)  # 限制y轴下限为0，突出损失变化趋势
    
    # 子图2：准确率、精确率、召回率、F1分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_metrics['accuracy'], marker='o', color='green', linestyle='-', label='Training Accuracy')
    plt.plot(range(1, epochs+1), train_metrics['precision'], marker='o', color='red', linestyle='-', label='Training Precision')
    plt.plot(range(1, epochs+1), train_metrics['recall'], marker='o', color='purple', linestyle='-', label='Training Recall')
    plt.plot(range(1, epochs+1), train_metrics['f1'], marker='o', color='black', linestyle='-', label='Training F1')
    plt.plot(range(1, epochs+1), val_metrics['accuracy'], marker='s', color='lightgreen', linestyle='--', label='Validation Accuracy')
    plt.plot(range(1, epochs+1), val_metrics['precision'], marker='s', color='pink', linestyle='--', label='Validation Precision')
    plt.plot(range(1, epochs+1), val_metrics['recall'], marker='s', color='violet', linestyle='--', label='Validation Recall')
    plt.plot(range(1, epochs+1), val_metrics['f1'], marker='s', color='gray', linestyle='--', label='Validation F1')
    
    plt.title('Accuracy/Precision/Recall/F1 Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(ncol=2, loc='lower right')  # 双列图例，移动到右下角
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=3)  # 调整子图间距
    plt.savefig(save_path, dpi=300)  # 保存高分辨率图片
    print(f'学习曲线图表已保存至: {save_path}')

def main():
    # 读取数据集
    print("正在加载数据...")
    train_df = pd.read_csv(train_path)
    ex_train_df = pd.read_csv(ex_train_path)  # 读取新增训练集
    print(f"ex训练集大小: {len(ex_train_df)}")
    print(f"原始训练集大小: {len(train_df)}")
    train_df = pd.concat([train_df,ex_train_df], ignore_index=True)
    print(f"合并后的训练集大小: {len(train_df)}")
    val_df = pd.read_csv(val_path)
    ex_val_df = pd.read_csv(ex_val_path)
    print(f"验证集大小: {len(val_df)}")
    print(f"新增验证集大小: {len(ex_val_df)}")
    val_df = pd.concat([val_df,ex_val_df], ignore_index=True)
    print(f"合并后的验证集大小: {len(val_df)}")
    

    # 构建词表
    print("正在构建词表...")
    vocab = build_vocab(train_df['text'])
    joblib.dump(vocab, vocab_path)  # 保存词表
    
    # 构建数据集
    train_set = RumorDataset(train_df, vocab)
    val_set = RumorDataset(val_df, vocab)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # 初始化模型、优化器和损失函数
    print("正在初始化模型...")
    model = AdvancedBiLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=FACTOR, patience=3, verbose=True
    )
    criterion = nn.BCEWithLogitsLoss()

    # 记录训练过程指标
    train_history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }
    val_history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
    }

    train_start_time = time.time()
    print("开始训练模型...")
    # 训练模型
    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        # 训练一个epoch
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)

        # 计算训练集指标
        train_acc, train_prec, train_rec, train_f1, _ = evaluate(model, train_loader)
        train_history['accuracy'].append(train_acc)
        train_history['precision'].append(train_prec)
        train_history['recall'].append(train_rec)
        train_history['f1'].append(train_f1)
        train_history['loss'].append(avg_loss)

        val_acc, val_prec, val_rec, val_f1, val_loss = evaluate(model, val_loader)
        val_history['accuracy'].append(val_acc)
        val_history['precision'].append(val_prec)
        val_history['recall'].append(val_rec)
        val_history['f1'].append(val_f1)
        val_history['loss'].append(val_loss)
        
        scheduler.step(val_f1)  # 根据验证集F1分数调整学习率
        
        print(f'Epoch {epoch+1}/{EPOCHS}')
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f'已保存最佳模型 (验证集F1: {val_f1:.4f})')
        
        print(f'训练集: Loss={avg_loss:.4f}, Acc={train_acc:.4f}, Prec={train_prec:.4f}, Rec={train_rec:.4f}, F1={train_f1:.4f}')
        print(f'验证集: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}')
        print('-' * 60)
    
    train_end_time = time.time()
    print("\n训练完成!")
    total_seconds = int(train_end_time - train_start_time)
    minutes, seconds = divmod(total_seconds, 60)
    print(f"总训练时间: {minutes}分{seconds}秒")
    # 绘制学习曲线
    plot_learning_curve(train_history, val_history, EPOCHS, graph_path)
    
if __name__ == '__main__':
    main()