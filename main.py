
from fastapi import FastAPI, APIRouter, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import pickle
brain_cancer = pickle.load(open("classifier_brain_cancer.pkl", "rb"))

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
    preds = brain_cancer.predict(img_array)
    predicted_class = int(np.argmax(preds, axis=1)[0])
    if predicted_class == 0:
        predicted_class = "glioma_tumor"
    elif predicted_class == 1:
        predicted_class = "meningioma_tumor"
    elif predicted_class == 2:
        predicted_class = "no_tumor"
    else:
        predicted_class = "pituitary_tumor"
    confidence = float(np.max(preds))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
