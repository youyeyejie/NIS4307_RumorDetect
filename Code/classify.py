import torch
import joblib
import argparse
import torch.nn as nn
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.005, help='学习率')
args = parser.parse_args()

BATCH_SIZE = 64         # 批大小
EMBEDDING_DIM = args.embedding_dim     # 嵌入维度
HIDDEN_DIM = args.hidden_dim        # 隐藏层维度
EPOCHS = args.epochs             # 训练轮数
MAX_LEN = 64            # 文本最大长度
LEARNING_RATE = args.lr    # 学习率
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备选择

model_parameter = f'{EMBEDDING_DIM}_{HIDDEN_DIM}_{EPOCHS}_{LEARNING_RATE}'
model_path = f'../Output/Model/{model_parameter}.pt'
vocab_path = f'../Output/Model/vocab_{model_parameter}.pkl'
train_path = '../Dataset/split/train.csv'
ex_train_1_path = '../Dataset/split/ex_train_1.csv'  # 新增训练集1
ex_train_2_path = '../Dataset/split/ex_train_2.csv'  # 新增训练集2
val_path = '../dataset/split/val.csv'
test_path = '../dataset/test/test.csv'
graph_path = f'../Output/Graph/{model_parameter}.png'

def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab['<PAD>']] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids

def tokenize(text):
    # 处理URL和@提及
    text = re.sub(r'http\S+', '<URL>', text)
    text = re.sub(r'@\w+', '@USER', text)
    
    # 使用NLTK分词+词干提取
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(w.lower()) for w in tokens if w.isalpha()]


class AdvancedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        # 使用预训练词向量（需替换为实际加载预训练向量的代码）
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout+0.1)  # 增加嵌入层后的Dropout
        
        # 简化LSTM结构
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim//2,  # 双向拼接后保持总维度不变
            num_layers=min(num_layers, 2),  # 限制层数
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制加入正则化
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # 增加中间层
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # 增加分类器正则化
        self.classifier = nn.Sequential(
            nn.Dropout(dropout+0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Embedding + Dropout
        emb = self.embedding_dropout(self.embedding(x))
        
        # BiLSTM
        outputs, _ = self.lstm(emb)
        
        # 注意力机制（加入mask处理padding）
        seq_len = x.ne(0).sum(dim=1, keepdim=True)
        attn_scores = self.attention(outputs).squeeze(-1)
        
        # 创建mask并填充负无穷
        mask = x.eq(0)
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        
        # 分类器
        return self.classifier(context).squeeze(1)


class RumourDetectClass:
    def __init__(self, model_path):
        # 加载词表和模型参数
        self.vocab = joblib.load(vocab_path)
        self.model = AdvancedBiLSTM(len(self.vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    def preprocess(self, text):
        # 文本预处理（与训练时一致）
        ids = encode(text, self.vocab)
        return ids
    
    def classify(self, text: str) -> int:
        # 预测流程
        ids = self.preprocess(text)
        x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            pred = (torch.sigmoid(logits) > 0.5).float().item()
        return int(pred)

if __name__ == "__main__":
    predict_path = test_path.replace('.csv', '_predictions.csv')
    expected_path = test_path.replace('.csv', '_expected.csv')

    detector = RumourDetectClass(model_path)

    test_data = pd.read_csv(test_path)
    test_texts = test_data['text'].tolist()

    predictions = []
    for text in test_texts:
        pred = detector.classify(text)
        predictions.append(pred)

    test_data['pred_label'] = predictions
    test_data.to_csv(predict_path, index=False)
    print(f"预测完成，结果已保存至{predict_path}")

    expected = pd.read_csv(expected_path)
    total = len(expected)
    correct = (test_data['pred_label'] == expected['label']).sum()
    accuracy = correct / total
    print(f"预测准确率: {accuracy:.2%} ({correct}/{total})")