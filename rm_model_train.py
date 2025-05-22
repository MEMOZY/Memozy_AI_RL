import pandas as pd
import re
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt


from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from xgboost import XGBRegressor


import joblib


# --- JSONL 로딩 ---
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            caption = record.get("caption", "")
            rate = float(record.get("rate", 0.0))
            data.append({"text": caption, "rate": rate})
    return pd.DataFrame(data)


# --- 텍스트 전처리 ---
def preprocess_text(text):
    text = re.sub(r'[^가-힣\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Oversampling 함수 ---
def apply_oversampling(df, num_bins=5):
    rate_buckets = defaultdict(list)

    # 균등하게 분할할 수 있도록 rate를 버킷으로 나눔
    min_rate, max_rate = df['rate'].min(), df['rate'].max()
    bin_size = (max_rate - min_rate) / num_bins

    def bucket_idx(rate):
        return int((rate - min_rate) // bin_size)

    for _, row in df.iterrows():
        idx = bucket_idx(row['rate'])
        rate_buckets[idx].append(row)

    # Oversampling
    max_count = max(len(samples) for samples in rate_buckets.values())
    balanced_samples = []

    for samples in rate_buckets.values():
        if len(samples) < max_count:
            oversampled = random.choices(samples, k=max_count - len(samples))
            balanced_samples.extend(samples + oversampled)
        else:
            balanced_samples.extend(samples)

    random.shuffle(balanced_samples)
    return pd.DataFrame(balanced_samples)


# --- 데이터 로딩 및 전처리 ---
df = load_jsonl("./data/RM_data/caption/human_rated_gpt_output.jsonl")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# --- Oversampling ---
df_balanced = apply_oversampling(df)

# --- BERT 임베딩 ---
bert_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')  # 한국어 포함 멀티링구얼

X = bert_model.encode(df_balanced['cleaned_text'].tolist(), show_progress_bar=True)
y = df_balanced['rate'].values

# --- Train/Test 분할 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 모델 학습 ---
# model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42) # MSE: 0.07910551549026897 MAE: 0.17151935618020955  R² Score: 0.8379005347899992
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42) # MSE: 0.05698540368337747 MAE: 0.1266120111062275 R² Score: 0.889218881686483
# model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42) #MSE: 0.0939286415107616 MAE: 0.17242499403145833 R² Score: 0.8190963099966417
# model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5, verbose=0, random_state=42) #MSE: 0.09919866752873258 MAE: 0.21750172460502057 R² Score: 0.8048769870784058
# model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42) #MSE: 0.07921171972602306 MAE: 0.2027260982172313  R² Score: 0.838990149670999
model.fit(X_train, y_train)

# --- 예측 및 평가 ---
preds = model.predict(X_test)
print("📉 MSE:", mean_squared_error(y_test, preds))
print("📈 MAE:", mean_absolute_error(y_test, preds))
print("🔍 R² Score:", r2_score(y_test, preds))

#---  모델 저장 ---
joblib.dump(model, 'XGBRegressor_model.pkl')

"""
📉 MSE: 0.06523728399644124
📈 MAE: 0.13134160440158613
🔍 R² Score: 0.8713930435161695
"""
### --- 예측 결과 시각화 ---
# 인덱스 순서
x = list(range(len(y_test)))

plt.figure(figsize=(14, 6))

# 실제 rate (빨간 선)
plt.plot(x, y_test, label='Label Rate', color='red', linewidth=2)

# 예측 rate (파란 선)
plt.plot(x, preds, label='Predict Rate', color='blue', linewidth=2)

plt.xlabel("Test Sample Index")
plt.ylabel("Rate")
plt.title("📈 Label Rate vs Predict Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
