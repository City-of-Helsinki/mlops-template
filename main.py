import logging
from typing import List

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from model_util import load_model

app = FastAPI(title="DataHel ML API", description="Generic API for ML model", version="1.0")
model = load_model('latest_model')

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
		allow_headers=["*"],
    max_age=3600,
)





# x = [[...], [...]]
@app.post("/predict", response_model=List[Prediction])
def predict(x: Parameters):
    logging.error(x)
    predictions: List[Prediction] = model.predict([[x.value]])
    response: List[Prediction] = []
    for prediction_value in predictions:
        prediction: Prediction = Prediction(value=prediction_value)
        response.append(prediction)
    return response