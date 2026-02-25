
from fastapi import FastAPI, APIRouter, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import pickle
'''
i load the model that i trained in the training phase,
and i will use it to predict the class of the image
that the user will upload to the api
and the model is a keras model that i trained on the brain cancer dataset
as will as the model is saved in a pickle file, and i will load it using the pickle library

'''
with open("classifier_brain_cancer.pkl", "rb") as f:
    brain_cancer = pickle.load(f)

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
     # treanform the image to a numpy array and normalize it
    img_array = np.array(image)
    img_array = img_array / 255.0
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
    # uvicorn.run(app, host="localhost", port=8000)
    # can be used for postman servers, or other servers that do not allow access to