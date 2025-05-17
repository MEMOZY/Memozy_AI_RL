import torch
from transformers import BertTokenizer, BertModel

# Reward Model 정의 (BERT Freeze + MLP) 그대로 사용
class RewardModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        self.tanh = nn.Tanh()

    def forward(self, captions):
        inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self.mlp[0].weight.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        reward = self.mlp(cls_embedding)
        reward = self.tanh(reward)
        return reward.squeeze(-1)

# --- 예측 함수 ---
def predict_reward(caption, model_path="./reward_model.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    model = RewardModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model([caption])
        pred = pred.item()  # tensor → float
    return round(pred, 2)  # 소수점 2자리까지 보기 편하게

# --- 사용 예시 ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

example_caption = "오늘 가족과 여행을 갔다. 정말 즐거운 시간을 보냈다."
predicted_reward = predict_reward(example_caption)
print(f"예측된 리워드: {predicted_reward}")
