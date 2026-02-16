
from fastapi import FastAPI, APIRouter, UploadFile
import uvicorn
import tensorflow as tf
import os
import keras
import numpy as np
from PIL import Image
from io import BytesIO
#from tensorflow.keras.models import load_model

brain_cancer_model = 'C:\Users\pc\classifier-brain-cancer\Early_classifier_brain_cancer .ipynb'

app = FastAPI()

router = APIRouter(
    prefix="/brain_cancer",
    tags=["brain_cancer"]
)

@router.post("/predict")
async def predict_brain_cancer(file: UploadFile):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))

    # تحويل إلى numpy
    img_array = np.array(image)
    # التنبؤ
    preds = brain_cancer_model.predict(img_array)
    predicted_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
