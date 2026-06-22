# ── fastapi_app.py ────────────────────────────────────────────────────────────
import io
import warnings
import numpy as np
import tensorflow as tf
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException

from config import CLASS_NAMES, IMG_SIZE, SAVEDMODEL
from loss   import focal_loss

warnings.filterwarnings("ignore")

# ── Response Schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]

# ── App ───────────────────────────────────────────────────────────────────────
app   = FastAPI(title="Brain MRI Classifier")
model = None

@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(
        SAVEDMODEL,
        compile=False,
        custom_objects={"focal_loss": focal_loss()}
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds    = model.predict(img, verbose=0)[0]
    pred_idx = int(np.argmax(preds))

    return PredictionResponse(
        predicted_class     = CLASS_NAMES[pred_idx],
        confidence          = round(float(preds[pred_idx]), 4),
        class_probabilities = {CLASS_NAMES[i]: round(float(preds[i]), 4) for i in range(4)}
    )
