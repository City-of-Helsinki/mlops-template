import logging
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI
from pandera.io import from_yaml
from pydantic import create_model
from starlette.middleware.cors import CORSMiddleware

from api_response import Prediction
from model_util import load_model


def build_parameter_model_definition(yaml_file:str):
    with open(yaml_file) as file:
        yaml_schema = file.read()
    schema = from_yaml(yaml_schema)
    fields = {}
    for col in schema.columns:
        t = schema.dtypes[col].type.type
        # convert object types to string
        if t == np.object_ or t == object:
            t = str
        name = col
        fields[name] = (t, ...)
    return fields

DynamicApiRequest = create_model('DynamicApiRequest', **build_parameter_model_definition('poc/api_params.yaml'))
DynamicApiResponse = create_model('DynamicApiResponse', **build_parameter_model_definition('poc/api_response.yaml'))
# determine response object value field and type
response_value_field = list(DynamicApiResponse.schema()['properties'])[0]
response_value_type = type(DynamicApiResponse.schema()['properties'][response_value_field]['type'])

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
@app.post("/predict", response_model=List[DynamicApiResponse])
def predict(p_list: List[DynamicApiRequest]):
    # loop trough parameter list
    prediction_values = []
    for p in p_list:
        # convert parameter object to array for model
        parameter_array = [getattr(p, k) for k in vars(p)]
        prediction_values.append(model.predict([parameter_array]))
    # Construct response
    response: List[DynamicApiResponse] = []
    for predicted_value in prediction_values:
        typed_value = response_value_type(predicted_value)
        response.append(DynamicApiResponse(**{response_value_field: typed_value}))
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)