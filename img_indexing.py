import os
from PIL import Image, ExifTags
"""
1st step

이미지 폴더 내 모든 이미지 파일을 순차적으로 불러와서
EXIF 정보를 확인하여 회전 정보에 따라 이미지를 변환한 후
새로운 폴더에 저장 숫자(인덱스)로 이름을 변경하여 저장하는 코드
"""

# 원본 이미지 폴더 경로
image_folder = "./data/raw_RM_img_data"
# 저장할 폴더 (없으면 생성)
output_folder = "./data/RM_data/img"
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 이미지 파일 가져오기
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

# 이미지 순차적으로 저장
for i, filename in enumerate(image_files, start=1):
    image_path = os.path.join(image_folder, filename)
    output_path = os.path.join(output_folder, f"{i:03d}.jpg")  # 001, 002, ... 형식
    
    # 이미지 열기
    with Image.open(image_path) as img:
        # EXIF 데이터 가져오기
        exif = img._getexif()
        
        # EXIF 정보가 있는 경우 회전 정보 가져오기
        if exif:
            for tag, value in exif.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == "Orientation":
                    orientation = value
                    # 회전 방향에 따라 이미지 변환
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
                    break

        # 이미지 크기 150x150으로 조정 -> gpt api 비용 절감
        img = img.resize((150, 150))        
        # 이미지 저장
        img.save(output_path, "JPEG")

print(f"{len(image_files)}장의 이미지가 {output_folder}에 올바르게 저장되었습니다.")
