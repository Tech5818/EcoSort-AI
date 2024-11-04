# yolo/yolo.py
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import uuid
import boto3
import dotenv
import io
from googletrans import Translator

translator = Translator()

dotenv.load_dotenv()

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

bucket_name = 'eco-sort'

# YOLO 모델 로드
model_path = 'yolov8n.pt'  # 모델 파일 경로
model = YOLO(model_path)

def process_image_with_yolo(image: Image.Image):
    # PIL 이미지를 OpenCV 형식으로 변환 (BGR로 변환)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 객체 탐지 수행
    results = model(img_cv)

    # 중복 없는 탐지된 객체 이름을 저장할 set
    detected_objects = set()
    
    # 탐지된 객체 정보 수집
    for result in results:
        boxes = result.boxes  # 탐지된 객체의 바운딩 박스 정보
        names = result.names  # 클래스 이름들
        for box in boxes:
            class_id = int(box.cls[0])  # 클래스 ID (정수형으로 변환)
            class_name = names[class_id]  # 클래스 이름
            # "person"과 "dining table"을 제외하고 추가
            if class_name != "person" and class_name != "dining table":
                text = class_name

                translated = translator.translate(text, src="en", dest="ko")

                detected_objects.add(translated.text)

    # 원본 이미지를 S3에 업로드
    image_filename = f'input_image_{uuid.uuid4()}.jpg'  # 고유한 이미지 파일 이름 생성
    image_buffer = io.BytesIO()
    image.save(image_buffer, format='JPEG')
    image_buffer.seek(0)

    s3_client.put_object(
        Bucket=bucket_name,
        Key=image_filename,
        Body=image_buffer,
        ContentType='image/jpeg',
        ACL="public-read"
    )

    # S3의 이미지 URL 형식
    image_url = f"https://{bucket_name}.s3.amazonaws.com/{image_filename}"

    return image_url, list(detected_objects)  # set을 list로 변환하여 반환
