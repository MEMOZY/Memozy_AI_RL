import json
import random

input_path = "./data/RM_data/caption/human_rated_gpt_output.jsonl"
train_output_path = "./data/RM_data/caption/train/gpt_output_with_human_rate_train.jsonl"
test_output_path = "./data/RM_data/caption/test/gpt_output_with_human_rate_test.jsonl"

# 데이터 불러오기
with open(input_path, 'r', encoding='utf-8') as infile:
    data_list = [json.loads(line) for line in infile]

# 섞기
random.shuffle(data_list)

# 분할 비율 계산
split_idx = int(len(data_list) * 0.9)

# train/test 분할
train_data = data_list[:split_idx]
test_data = data_list[split_idx:]

# 저장
with open(train_output_path, 'w', encoding='utf-8') as f_train:
    for item in train_data:
        f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(test_output_path, 'w', encoding='utf-8') as f_test:
    for item in test_data:
        f_test.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Train 데이터: {len(train_data)}개 → {train_output_path}")
print(f"Test 데이터: {len(test_data)}개 → {test_output_path}")
