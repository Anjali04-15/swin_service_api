import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms

from swin_model import load_swin_model

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once at startup
model = load_swin_model()
model.to(device)
model.eval()

# Define image preprocessing to match training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # typical ImageNet means
                         std=[0.229, 0.224, 0.225]),
])

labels = [
    'fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum', 'fresh_orange', 'fresh_tomato',
    'stale_apple', 'stale_banana', 'stale_bitter_gourd', 'stale_capsicum', 'stale_orange', 'stale_tomato'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0)  # batch size 1
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "label": labels[predicted.item()],
        "confidence": confidence.item()
    }
