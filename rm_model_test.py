import torch
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

# 1. 토크나이저 및 디바이스 설정
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 데이터셋 정의 (기존과 동일)
class CaptionRewardDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                caption = data['caption']
                rate = data.get('rate', 0)
                try:
                    rate = float(rate)
                except:
                    rate = 0.0
                self.samples.append((caption, rate))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# 3. 모델 정의 (기존과 동일)
import torch.nn as nn
from transformers import BertModel

class RewardModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        for param in self.bert.parameters():
            param.requires_grad = False

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

# 4. 모델 로드
model = RewardModel().to(device)
model.load_state_dict(torch.load("reward_model.pt", map_location=device))
model.eval()

# 5. 데이터 로드
test_dataset = CaptionRewardDataset("gpt_output_with_human_rate_test.jsonl")
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 6. 예측 및 출력
print(f"{'Index':<6} {'GT Rate':<10} {'Pred Rate':<10} {'Caption'}")
print("=" * 80)

with torch.no_grad():
    for i, (captions, true_rates) in enumerate(test_loader):
        preds = model(captions).cpu().tolist()
        for idx in range(len(captions)):
            print(f"{i*8 + idx:<6} {true_rates[idx]:<10.2f} {preds[idx]:<10.2f} {captions[idx][:60]}")

