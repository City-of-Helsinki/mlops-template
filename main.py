from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI
from pandera.io import from_yaml
from pydantic import create_model
from starlette.middleware.cors import CORSMiddleware

from model_util import unpickle_bundle, ModelSchemaContainer


# Generates dynamically request and response classes for openapi schema and swagger documentation
def build_parameter_model_definition(yaml_schema: str):
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


# Load model and schema definitions from pickled container class
model_and_schema: ModelSchemaContainer = unpickle_bundle('bundle_latest')
# ML model
model = model_and_schema.model

# metrics
metrics = model_and_schema.metrics

# Schema for request (X)
DynamicApiRequest = create_model('DynamicApiRequest', **build_parameter_model_definition(model_and_schema.req_schema))
# Schema for response (y)
DynamicApiResponse = create_model('DynamicApiResponse', **build_parameter_model_definition(model_and_schema.res_schema))

# Determine response object value field and type
response_value_field = list(DynamicApiResponse.schema()['properties'])[0]
response_value_type = type(DynamicApiResponse.schema()['properties'][response_value_field]['type'])

# Start up API
app = FastAPI(title="DataHel ML API", description="Generic API for ML model.", version="1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # allow all origins
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


@app.get("/metrics", response_model=str)
def get_metrics():
    return metrics


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
    # TODO: Response is now array.
    # [
    #     {
    #         "variety": "['Setosa']"
    #     }
    # ]
    for predicted_value in prediction_values:
        # Cast predicted value to correct type and add response value to response array
        typed_value = response_value_type(predicted_value)
        response.append(DynamicApiResponse(**{response_value_field: typed_value}))
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)