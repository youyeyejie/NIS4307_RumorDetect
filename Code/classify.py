import torch
import joblib
import torch.nn as nn
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from model import AdvancedBiLSTM3 as AdvancedBiLSTM 
from train_lstm import *

MAX_LEN = 64  # 文本最大长度
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备选择


class RumourDetectClass:
    def __init__(self, model_path,vocab_path,EMBEDDING_DIM=128, HIDDEN_DIM=256,  DEVICE=None):
        # 加载词表和模型参数
        self.vocab = joblib.load(vocab_path)
        self.model = AdvancedBiLSTM(len(self.vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    @classmethod
    def construct_detector(cls):
        return cls(model_path, vocab_path, EMBEDDING_DIM, HIDDEN_DIM, DEVICE)
    
    def classify(self, text: str) -> int:
        # 预测流程
        ids = encode(text, self.vocab)
        x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            pred = (torch.sigmoid(logits) > 0.5).float().item()
        return int(pred)
