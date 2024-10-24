from fastapi import APIRouter, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse
from PIL import Image
import io
from yolo.yolo import process_image_with_yolo


yolo_router = APIRouter(tags=['yolo'])

@yolo_router.post("/predict")
async def predict(file:UploadFile):
  try:
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    result_image_base64, detected_objects = process_image_with_yolo(img)

    return JSONResponse(content={
          "detected_objects": detected_objects,
          "result_image": result_image_base64  # Base64로 인코딩된 이미지
      })

  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))