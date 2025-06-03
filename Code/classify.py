from train_lstm import *

class RumourDetectClass:
    def __init__(self, model_path, vocab_path, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, device=DEVICE):
        # 加载词表和模型参数
        self.vocab = joblib.load(vocab_path)
        self.model = AdvancedBiLSTM(len(self.vocab), embedding_dim, hidden_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    @classmethod
    def construct_detector(cls):
        embedding_dim = EMBEDDING_DIM
        hidden_dim = HIDDEN_DIM
        epochs = EPOCHS
        learning_rate = LEARNING_RATE
        device = DEVICE
        model_path = f'../Output/Model/best_{embedding_dim}_{hidden_dim}_{epochs}_{learning_rate}.pt'
        vocab_path = f'../Output/Model/vocab_{embedding_dim}_{hidden_dim}_{epochs}_{learning_rate}.pkl'
        return cls(model_path, vocab_path, embedding_dim, hidden_dim, device)

    def classify(self, text: str) -> int:
        # 预测流程
        ids = encode(text, self.vocab)
        x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            pred = (torch.sigmoid(logits) > 0.5).float().item()
        return int(pred)
