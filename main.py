from fastapi import FastAPI
from router.yolo import yolo_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # 모든 출처 허용
  allow_credentials=True,
  allow_methods=["*"],  # 모든 HTTP 메서드 허용
  allow_headers=["*"],  # 모든 헤더 허용
)

app.include_router(yolo_router, prefix="/yolo")


@app.get("/")
async def home():
  return "success"

import uvicorn

if __name__ == "__main__":
  uvicorn.run("main:app", host="localhost", port=8088, reload=True)