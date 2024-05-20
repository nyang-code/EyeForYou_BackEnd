from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import pandas as pd
import timm
from scipy.ndimage import label
import copy
from PIL import Image
import io
from transformers import pipeline
import torch.nn as nn
import torch.nn.functional as F

class CustomImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomImageClassifier, self).__init__()
        self.base_model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = F.softmax(x, dim=1)
        return x

app = FastAPI()

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 및 클래스 설정
CUSTOM_CLASSES = ['background', 'drink', 'snack', 'Human_hand']

# 글로벌 변수로 모델들을 정의
segmentation_model = None
classification_models = None
transform = None
pipe = None

# 모델 로드 함수
def load_model(model_path, model, device='cpu'):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def initialize_models():
    global segmentation_model, classification_models, transform, pipe

    segmentation_model_path = 'C:/Users/MMC/backend/seg.pt'
    classification_model_paths = {
        "snack": 'C:/Users/MMC/backend/snack.pt',
        "drink": 'C:/Users/MMC/backend/drink.pt'
    }
    
    segmentation_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    segmentation_model.classifier[4] = torch.nn.Conv2d(256, len(CUSTOM_CLASSES), kernel_size=1)
    segmentation_model = load_model(segmentation_model_path, segmentation_model, device)

    classification_models = {
        "snack": CustomImageClassifier(num_classes=11),
        "drink": CustomImageClassifier(num_classes=11)
    }
    classification_models["snack"] = load_model(classification_model_paths["snack"], classification_models["snack"], device)
    classification_models["drink"] = load_model(classification_model_paths["drink"], classification_models["drink"], device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", device=device, trust_remote_code=True)

# 이미지 로드 및 전처리
def load_image(image_data):
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = np.array(img)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 768)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    img_resized = cv2.resize(img, (768, 1024))
    img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가
    return img, img_tensor, img_resized

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions.byte().cpu().numpy()

def select_largest_class(mask, class1_id, class2_id):
    class1_size = np.sum(mask == class1_id)
    class2_size = np.sum(mask == class2_id)
    if class1_size > class2_size:
        mask[mask == class2_id] = 0  # 작은 클래스는 배경으로 설정
        return class1_id
    else:
        mask[mask == class1_id] = 0  # 작은 클래스는 배경으로 설정
        return class2_id

def select_largest_bbox(mask, class_id):
    labeled_mask, num_labels = label(mask == class_id)
    largest_area = 0
    largest_label = 0
    for i in range(1, num_labels + 1):
        area = np.sum(labeled_mask == i)
        if area > largest_area:
            largest_area = area
            largest_label = i
    mask[mask == class_id] = 0  # 모든 객체를 배경으로 설정
    mask[labeled_mask == largest_label] = class_id  # 가장 큰 바운딩 박스만 남기기
    return mask

def get_product_name(class_id, csv_path):
    df = pd.read_csv(csv_path)
    class_to_name = df.set_index('class')['img_prod_nm'].to_dict()
    return class_to_name.get(class_id, "Unknown Product")

def inference(model, image, device='cpu'):
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # 배치를 추가하기 위해 unsqueeze(0)을 사용

        outputs = model(image)
        logits = outputs.cpu().numpy()
        top10_indices = np.argsort(logits[0])[::-1][:10]
        print("Top 10 class indices:", top10_indices)
        print("Logits of top 10 classes:", logits[0][top10_indices])
        
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

def remove_background(image, pipe):
    pillow_image = pipe(image)

    np_image = np.array(pillow_image)
    np_image[np_image == 0] = 255

    modified_image = Image.fromarray(np_image)
    if modified_image.mode == 'RGBA':
        modified_image = modified_image.convert('RGB')

    return modified_image

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    global segmentation_model, classification_models, transform, pipe
    
    if segmentation_model is None or classification_models is None or transform is None or pipe is None:
        initialize_models()

    # 이미지 로드 및 전처리
    image_data = await file.read()
    original_img, img_tensor, img_resized = load_image(image_data)

    # 분할 수행
    original_mask = predict(segmentation_model, img_tensor, device)

    # snack과 drink 클래스 중 더 큰 것을 남기고 나머지는 배경으로 전환
    if np.any(original_mask == 1) or np.any(original_mask == 2):
        remaining_class_id = select_largest_class(original_mask, class1_id=2, class2_id=1)

        # hand 클래스가 식별된 경우에만 가장 큰 바운딩 박스만 남기기
        if np.any(original_mask == 3):
            mask = select_largest_bbox(original_mask, class_id=remaining_class_id)

            # 배경 제거
            pillow_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            background_removed_img = remove_background(pillow_image, pipe)
            background_removed_img = np.array(background_removed_img)

            # 분할된 이미지 변환
            classification_img = transform(background_removed_img)

            # remaining_class_id에 따라 분류 모델 및 csv 파일 선택
            if remaining_class_id == 2:
                classification_model = classification_models["snack"]
                csv_path = 'C:/Users/MMC/backend/snack.csv'
            else:
                classification_model = classification_models["drink"]
                csv_path = 'C:/Users/MMC/backend/drink.csv'

            # 추론 수행
            test_predictions = inference(classification_model, classification_img, device)
            result = get_product_name(test_predictions[0], csv_path)

            # 결과를 백엔드 콘솔에 출력
            print(f"Prediction result: {result}")

            return JSONResponse(content=result)
        else:
            if np.any(original_mask == 2):
                print("Snack corner")
                return JSONResponse(content="과자 코너")
            elif np.any(original_mask == 1):
                print("Drink corner")
                return JSONResponse(content="음료 코너")
    else:
        print("No relevant objects detected")
        return JSONResponse(content={"error": "No relevant objects detected"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
