# train data를 읽어들여서 -1,-0.5,0,0.5,1의 분포를 확인한다

import json
from collections import Counter
import matplotlib.pyplot as plt

def plot_rate_distribution(jsonl_path):
    rates = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            rate = data.get("rate", 0)
            try:
                rate = float(rate)
                rates.append(rate)
            except:
                continue

    # 원하는 클래스 목록
    class_bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
    rate_counter = Counter(rates)

    # 분포 계산 (class_bins 기준)
    distribution = {str(cls): rate_counter.get(cls, 0) for cls in class_bins}

    # 시각화
    plt.figure(figsize=(6,4))
    plt.bar(distribution.keys(), distribution.values(), color='skyblue')
    plt.title("Rate Class Distribution")
    plt.xlabel("Class (rate)")
    plt.ylabel("Count")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # 분포 출력
    print("Rate Class Distribution:")
    for cls, count in distribution.items():
        print(f"Class {cls}: {count} samples")
    

    return distribution

# 예시 사용
jsonl_path = "data/RM_data/caption/train/gpt_output_with_human_rate_train.jsonl"
dist = plot_rate_distribution(jsonl_path)
print(dist)
