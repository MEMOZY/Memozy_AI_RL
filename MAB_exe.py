import os
import random
import numpy as np
import matplotlib.pyplot as plt
import base64
from openai import OpenAI

# OpenAI 클라이언트 설정
client = OpenAI()

# 프롬프트 A~D 정의
prompts = {
    "A": "짧고 간단한 어린이 스타일로 다음 상황에 맞는 일기를 써줘:\n",
    "B": "감성적이고 섬세한 느낌으로 다음 내용을 일기로 써줘:\n",
    "C": "재미있고 유쾌한 말투로 다음 상황을 일기로 작성해줘:\n",
    "D": "일상적인 문체로 자연스럽게 일기를 써줘:\n"
}

# 상황 정보
info_pool = {
    "장소": ["춘천", "부산", "서울", "여수"],
    "인물": ["친구", "혼자", "가족"],
    "기분": ["행복", "즐거움"]
}

# 이미지 폴더
image_dir = "./data/MAB_data/img"
image_list = os.listdir(image_dir)

# 상황 샘플링
def sample_random_situation():
    place = random.choice(info_pool["장소"])
    people = random.choice(info_pool["인물"])
    mood = random.choice(info_pool["기분"])
    return f"오늘 {place}에 {people}랑 놀러갔어. 기분은 {mood}했어."

# 이미지 base64 인코딩
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# GPT-4o를 통한 일기 생성
def call_gpt_diary_api(prompt_prefix, situation_text, image_path):
    base64_image = image_to_base64(image_path)

    completion = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal:capstone150img:BMxNfNjK",
        messages=[
            {"role": "user", "content": situation_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_prefix},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        },
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content.strip()

# GPT-4o를 통한 보상 평가
def call_reward_model_api(diary_text):
    prompt = f"""
    다음 일기를 읽고 1점 만점으로 평가해줘. 기준은 창의성, 감정 표현, 구성력이고 점수는 -1.00에서 +1.00까지 소수점 둘째 자리로 숫자만 출력해줘.

"""

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            
            {"role": "user", "content": {prompt}},
            {"role": "user", "content": {diary_text}},
        ],
        temperature=0,
        max_tokens=10,
    )

    score_text = completion.choices[0].message.content.strip()
    try:
        return float(score_text)
    except:
        return 0.0

# UCB 알고리즘
class UCBSelector:
    def __init__(self, arms):
        self.arms = arms
        self.counts = {arm: 0 for arm in arms}
        self.values = {arm: 0.0 for arm in arms}
        self.total_counts = 0

    def select_arm(self):
        self.total_counts += 1
        for arm in self.arms:
            if self.counts[arm] == 0:
                return arm
        ucb_values = {}
        for arm in self.arms:
            bonus = np.sqrt(2 * np.log(self.total_counts) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus
        return max(ucb_values, key=ucb_values.get)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

# 메인 루프
ucb = UCBSelector(list(prompts.keys()))
rewards_history = {key: [] for key in prompts.keys()}
num_iterations = 50  # 충분히 작은 수부터 시작 추천

for i in range(num_iterations):
    selected_prompt = ucb.select_arm()
    situation = sample_random_situation()
    image_path = os.path.join(image_dir, random.choice(image_list))

    try:
        diary = call_gpt_diary_api(prompts[selected_prompt], situation, image_path)
        reward = call_reward_model_api(diary)
    except Exception as e:
        print(f"API 호출 실패: {e}")
        reward = 0.0

    ucb.update(selected_prompt, reward)
    rewards_history[selected_prompt].append(reward)

    print(f"[{i+1}/{num_iterations}] Prompt {selected_prompt} | Reward: {reward:.2f}")

# 시각화
plt.figure(figsize=(10, 6))
for key, rewards in rewards_history.items():
    if rewards:
        mean_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
        plt.plot(mean_rewards, label=f"Prompt {key}")
plt.title("UCB: mean reward")
plt.xlabel("iter_num")
plt.ylabel("expected reward")
plt.legend()
plt.grid(True)
plt.show()
