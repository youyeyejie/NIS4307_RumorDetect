import torch
from train_gru import *

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