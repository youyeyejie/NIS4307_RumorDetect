import torch
import re
from train_gru import *

class RumourDetectClass:
    def __init__(self):
        # 加载词表和模型参数
        self.vocab = build_vocab(pd.read_csv('../dataset/split/train.csv')['text'])  # 重新构建词表（或保存词表文件）
        self.model = BiGRU(len(self.vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
        self.model.load_state_dict(torch.load('../Output/bigru.pt', map_location=DEVICE))
        self.model.eval()

    def preprocess(self, text):
        # 文本预处理（与训练时一致）
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text
    
    def classify(self, text: str) -> int:
        # 预测流程
        text = self.preprocess(text)
        ids = encode(text, self.vocab)
        x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            pred = (torch.sigmoid(logits) > 0.5).float().item()
        return int(pred)