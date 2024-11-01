# yolo/yolo.py
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import uuid
import boto3
import dotenv

dotenv.load_dotenv()

s3_client = boto3.client('s3',  
  aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
  aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
  region_name=os.getenv("AWS_DEFAULT_REGION")
  )

bucket_name = 'eco-sort-image'

# YOLO 모델 로드
model_path = 'yolov8n.pt'  # 모델 파일 경로
model = YOLO(model_path)

def process_image_with_yolo(image: Image.Image):
    # PIL 이미지를 OpenCV 형식으로 변환 (BGR로 변환)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 객체 탐지 수행
    results = model(img_cv)

    box_result_list = []
    
    # 탐지된 객체 정보 수집
    for result in results:
        boxes = result.boxes  # 탐지된 객체의 바운딩 박스 정보
        names = result.names  # 클래스 이름들
        for box in boxes:
            class_id = int(box.cls[0])  # 클래스 ID (정수형으로 변환)
            class_name = names[class_id]  # 클래스 이름
            # box_result_list.append({
            #     "class_name": class_name,
            #     "coordinates": box.xyxy[0].tolist()  # 바운딩 박스 좌표를 리스트로 변환
            # })
            box_result_list.append(class_name)

    # 결과 이미지 시각화
    result_image = results[0].plot()  # 탐지된 이미지 결과를 가져옴
    
    # S3에 결과 이미지 저장
    image_filename = f'output_image_{uuid.uuid4()}.jpg'  # 고유한 이미지 파일 이름 생성
    image_buffer = cv2.imencode('.jpg', result_image)[1]
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=image_filename,
        Body=image_buffer.tobytes(),
        ContentType='image/jpeg',
        ACL="public-read"
    )

    # S3의 이미지 URL 형식
    image_url = f"https://{bucket_name}.s3.amazonaws.com/{image_filename}"

    return image_url, box_result_list

    #----------------------------

