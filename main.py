from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os

app = FastAPI()

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # 파일을 저장할 경로 설정
        file_location = f"uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        
        # 업로드된 파일 저장
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        
        # 여기서 AI 이미지 처리 로직을 추가할 수 있음.
        # 예: processed_image = your_ai_model_function(file_location)
        
        # 이미지 처리 결과를 반환 (예제에서는 단순히 파일 이름을 반환)
        return {"filename": file.filename}
    
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})

@app.get("/")
async def root():
    return {"message": "Hello World"}

# 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
