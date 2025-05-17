import json
import random

input_path = "./data/RM_data/caption/gpt_output.jsonl"
output_path = "./data/RM_data/caption/gpt_output_with_auto_rate.jsonl"

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        # rate가 없거나 빈 경우 랜덤으로 부여
        if not data.get("rate") or str(data["rate"]).strip() == "":
            random_rate = round(random.uniform(-1.0, 1.0), 1)
            data["rate"] = random_rate
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"랜덤 rate 부여 완료 → {output_path}")
