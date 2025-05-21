# regression
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

# 시드 고정
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 수정된 CaptionRewardDataset 클래스
class CaptionRewardDataset(Dataset):
    def __init__(self, jsonl_path, oversample=True):
        self.samples = []
        rate_buckets = defaultdict(list)

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
                if oversample:
                    rate_buckets[rate].append((caption, rate))

        # --- Oversampling ---
        if oversample:
            max_count = max(len(samples) for samples in rate_buckets.values())
            balanced_samples = []

            for rate, samples in rate_buckets.items():
                if len(samples) < max_count:
                    # 부족한 샘플을 복제해서 채움
                    oversampled = random.choices(samples, k=max_count - len(samples))
                    balanced_samples.extend(samples + oversampled)
                else:
                    balanced_samples.extend(samples)

            self.samples = balanced_samples
            random.shuffle(self.samples)

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
                       epochs=100, batch_size=8, lr=1e-3,
                       device="cuda" if torch.cuda.is_available() else "cpu"):

    # 데이터 로딩
    train_dataset = CaptionRewardDataset(train_path)
    test_dataset = CaptionRewardDataset(test_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 옵티마이저, 손실함수
    model = RewardModel().to(device)
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    # 결과 저장
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')  # 가장 낮은 테스트 손실 초기화

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

        # --- 가장 좋은 모델 저장 ---
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ 모델 저장됨 (Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f})")

    print(f"\n🎯 최종 최소 Test Loss: {best_test_loss:.4f}")
    print(f"모델 저장 경로: {save_path}")

    # --- 시각화 ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1Loss Loss")
    plt.title("Reward Model Training & Test Loss")
    plt.legend()
    plt.grid()
    plt.savefig("reward_model_loss_plot.png")
    print("그래프 이미지 저장 완료: reward_model_loss_plot.png")
    plt.show()

    return model


# 4. 실행 예시
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_file = "./gpt_output_with_human_rate_train.jsonl"
test_file = "./gpt_output_with_human_rate_test.jsonl"
trained_model = train_and_evaluate(train_file, test_file)
