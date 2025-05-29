import torch
import joblib
import torch.nn as nn
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from model import AdvancedBiLSTM3 as AdvancedBiLSTM  

MAX_LEN = 64  # 文本最大长度
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备选择

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


class RumourDetectClass:
    def __init__(self, model_path,vocab_path,EMBEDDING_DIM=128, HIDDEN_DIM=256,  DEVICE=None):

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
