from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os

app = FastAPI()

# 파일을 저장할 디렉토리 경로
UPLOAD_DIRECTORY = "uploaded_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb+") as file_object:
        # 파일 데이터를 64KB 단위로 저장
        while content := await file.read(65536):
            file_object.write(content)
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@app.get("/files/{filename}")
async def get_file(filename: str):
    file_location = os.path.join(UPLOAD_DIRECTORY, filename)
    if os.path.exists(file_location):
        return FileResponse(path=file_location, filename=filename)
    return {"error": "File not found."}
