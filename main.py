from fastapi import FastAPI
from router.yolo import yolo_router

app = FastAPI()

app.include_router(yolo_router, prefix="/yolo")

@app.get("/")
async def home():
  return "success"

import uvicorn

if __name__ == "__main__":
  uvicorn.run("main:app", host="localhost", port=8088, reload=True)