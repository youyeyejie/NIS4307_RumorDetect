import torch
import torch.nn as nn
class AdvancedBiLSTM1(nn.Module):
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

class AdvancedBiLSTM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 注意力层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)  # [batch, seq_len] → [batch, seq_len, emb_dim]
        
        # BiLSTM
        outputs, (h_n, c_n) = self.lstm(emb)  # outputs: [batch, seq_len, hidden_dim*2]
        
        # 注意力机制
        attn_weights = torch.softmax(self.attention(outputs), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * outputs, dim=1)  # [batch, hidden_dim*2]
        
        # 分类
        context = self.dropout(context)
        out = self.fc(context)  # [batch, 1]
        return out.squeeze(1)

class AdvancedBiLSTM3(nn.Module):
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

class AdvancedBiLSTM4(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 多尺度CNN提取局部特征
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_dim*2, out_channels=64, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]
        ])
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2 + 64*3, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Embedding
        emb = self.embedding_dropout(self.embedding(x))  # [batch, seq_len, emb_dim]
        
        # BiLSTM
        lstm_out, _ = self.lstm(emb)  # [batch, seq_len, hidden_dim*2]
        
        # CNN特征
        cnn_in = lstm_out.transpose(1, 2)  # [batch, hidden_dim*2, seq_len]
        cnn_outs = [torch.relu(conv(cnn_in)) for conv in self.convs]
        cnn_pooled = [torch.max_pool1d(out, kernel_size=out.size(2)).squeeze(2) for out in cnn_outs]
        cnn_features = torch.cat(cnn_pooled, dim=1)  # [batch, 64*3]
        
        # 注意力机制
        attn_scores = self.attention(lstm_out).squeeze(-1)
        mask = x.eq(0)
        attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # [batch, hidden_dim*2]
        
        # 合并特征
        combined = torch.cat([context, cnn_features], dim=1)
        
        # 分类
        return self.classifier(combined).squeeze(1)

class AdvancedBiLSTM5(nn.Module):
    # BiLSTM模型定义
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        # 前向传播
        emb = self.embedding(x)
        output, (h_n, c_n) = self.bilstm(emb)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        h = self.dropout(h)
        out = self.fc(h)
        return out.squeeze(1)
