import os
import json
import base64
import random
from openai import OpenAI
import time

# OpenAI 클라이언트 초기화
client = OpenAI(api_key="OPENAI_API_KEY")

# 이미지 인코딩 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 단일 프롬프트
prompt_text = """
역할(Role):
당신은 사용자의 사진일기를 대신 작성하는 어시스턴트입니다.

목표(Goal):
사용자가 제공한 사진과 대화에서 수집한 정보를 바탕으로 자연스럽고 일상적인 느낌의 일기를 작성합니다.

지시사항(Instructions):

- 수집한 정보를 기반으로 일기를 작성하되, 사용자의 감정이나 기분을 추측하지 마세요.
- 사용자가 제공하지 않은 정보를 임의로 추가하지 마세요.
- 일기는 자연스럽고 일상적인 말투로 작성하세요.
- 비속어나 검열은 하지 않아도 괜찮지만, 일기의 흐름에 맞게 자연스럽게 표현하세요.
- 일기의 내용 외에는 추가하지 마세요.(해석, 주석, 부연 설명 없이 순수한 일기 형태로 작성)

출력 형식(Output Format):
자연스럽고 일상적인 말투로 작성된 일기를 제공합니다.
일기의 내용 외에는 출력하지 마세요.
"""

# 장소, 인물, 기분 정보
information = {
    "장소": ["춘천", "부산", "서울","여수"],
    "인물": ["친구", "혼자", "가족"],
    "기분": ["행복", "즐거움"]
}

# 이미지 폴더 경로 (100장 사용)
image_folder = "./data/RM_data/img"
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])[:100] # 100장만 사용

# 출력 파일 경로
output_path = "./data/RM_data/caption/gpt_output.jsonl"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# JSONL 파일로 저장
with open(output_path, "w", encoding="utf-8") as f:
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        base64_image = encode_image(image_path)

        # 이미지당 5개 생성
        for _ in range(5):
            # 랜덤 대화 정보 생성
            place = random.choice(information["장소"])
            people = random.choice(information["인물"])
            mood = random.choice(information["기분"])

            user_dialog = f"오늘 {place}에 {people}랑 놀러갔어. 기분은 {mood}했어."

            try:
                completion = client.chat.completions.create(
                    model="ft:gpt-4o-2024-08-06:personal:capstone150img:BMxNfNjK",
                    messages=[
                        {"role": "user", "content": user_dialog},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
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
                caption = completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"{filename} 처리 중 오류 발생: {e}")
                caption = "ERROR"

            json_line = {
                "image_id": filename,
                "caption": caption,
                "rate": ""  # 사람이 나중에 -1, 0, 1 채점할 부분
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            # ✅ 요청 후 0.2초 대기 (API 과부하 방지)
            time.sleep(0.2)

print(f"총 {len(image_files) * 5}개의 캡션이 {output_path}에 저장되었습니다.")
