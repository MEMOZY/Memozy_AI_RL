import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json
import os
import matplotlib.pyplot as plt

# 1. Dataset 정의
class CaptionRewardDataset(Dataset):
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
        caption, rate = self.samples[idx]
        return caption, rate

# 2. Reward Model 정의 (BERT Freeze + MLP)
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

# 3. 학습 및 평가 함수
def train_and_evaluate(train_path, test_path, save_path="./reward_model.pt",
                       epochs=30, batch_size=8, lr=1e-3,
                       device="cuda" if torch.cuda.is_available() else "cpu"):

    # 데이터 로딩
    train_dataset = CaptionRewardDataset(train_path)
    test_dataset = CaptionRewardDataset(test_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 옵티마이저, 손실함수
    model = RewardModel().to(device)
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 결과 저장
    train_losses = []
    test_losses = []

    model.train()
    for epoch in range(epochs):
        total_train_loss = 0

        for captions, rates in train_loader:
            optimizer.zero_grad()
            preds = model(captions)
            rates = rates.to(device).float()
            loss = criterion(preds, rates)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(captions)

        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # --- 평가 ---
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for captions, rates in test_loader:
                preds = model(captions)
                rates = rates.to(device).float()
                loss = criterion(preds, rates)
                total_test_loss += loss.item() * len(captions)

        avg_test_loss = total_test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)
        model.train()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")

    # --- 모델 저장 ---
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장 완료: {save_path}")

    # --- 시각화 ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Reward Model Training & Test Loss")
    plt.legend()
    plt.grid()
    plt.show()

    return model

# 4. 실행 예시
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_file = "./data/RM_data/caption/train/gpt_output_with_auto_rate_train.jsonl"
test_file = "./data/RM_data/caption/test/gpt_output_with_auto_rate_test.jsonl"
trained_model = train_and_evaluate(train_file, test_file)
