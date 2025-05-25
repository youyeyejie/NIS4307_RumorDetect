import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import re
import joblib
import matplotlib.pyplot as plt

# 超参数设置
BATCH_SIZE = 32 # 批大小
EMBEDDING_DIM = 100 # 嵌入维度
HIDDEN_DIM = 128 # 隐藏层维度
EPOCHS = 10 # 训练轮数
MAX_LEN = 64 # 文本最大长度
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 设备选择

model_parameter = f'embedding_{EMBEDDING_DIM}_hidden_{HIDDEN_DIM}_epoch_{EPOCHS}'
model_path = f'../Output/Model/{model_parameter}.pt'
vocab_path = '../Output/Model/vocab.pkl'
train_path = '../dataset/split/train.csv'
val_path = '../dataset/split/val.csv'
test_path = '../dataset/test/test.csv'
graph_path = f'../Output/Graph/{model_parameter}.png'

# 简单分词器
def tokenize(text):
    return re.findall(r'\w+', text.lower())

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

class BiGRU(nn.Module):
    # BiGRU模型定义
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bigru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        # 前向传播
        emb = self.embedding(x)
        _, h = self.bigru(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        out = self.fc(h)
        return out.squeeze(1)

def evaluate(model, loader):
    # 评估函数，返回准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    # 读取数据集
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # 构建词表
    vocab = build_vocab(train_df['text'])
    joblib.dump(vocab, vocab_path)  # 保存词表
    # 构建数据集
    train_set = RumorDataset(train_df, vocab)
    val_set = RumorDataset(val_df, vocab)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    # 初始化模型、优化器和损失函数
    model = BiGRU(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_accuracies = []

    # 训练模型
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        batch_count = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        val_acc = evaluate(model, val_loader)
        val_accuracies.append(val_acc)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')

    # 绘制训练损失和验证准确率曲线
    plt.figure(figsize=(10,4))
    plt.suptitle(f'Parameter: {model_parameter}', fontsize=16)
    plt.subplot(1,2,1)
    plt.plot(range(1, EPOCHS+1), train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1,2,2)
    plt.plot(range(1, EPOCHS+1), val_accuracies, marker='o', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 避免总标题和子图重叠
    plt.savefig(graph_path)
    # plt.show()
        
    # 保存模型checkpoint
    torch.save(model.state_dict(), model_path)
    print(f'模型已保存为{model_parameter}.pt')

if __name__ == '__main__':
    main() 