import logging
from typing import List

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api_params import Parameters
from api_response import Prediction
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
def predict(p_list: List[Parameters]):
    logging.error('Parameters', p_list)
    # loop trough parameter list
    prediction_values = []
    for p in p_list:
        # convert parameter object to array for model
        parameter_array = [getattr(p, k) for k in vars(p)]
        prediction_values.append(model.predict([parameter_array]))
    # Construct response
    response: List[Prediction] = []
    for predicted_value in prediction_values:
        logging.error("Predicted value:", predicted_value)
        typed_value = type(Prediction.schema()['properties']['value']['type'])(predicted_value)
        prediction: Prediction = Prediction(value=typed_value)
        response.append(prediction)
    return response
