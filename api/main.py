import os
from typing import List
import uvicorn
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.params import Depends
from fastapi.responses import HTMLResponse
from pydantic import create_model
from starlette.middleware.cors import CORSMiddleware
import time

import pandas as pd

import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from model_util import (
    unpickle_bundle,
    ModelSchemaContainer,
    build_model_definition_from_dict,
    schema_to_pandas_columns,
)

from metrics import (
    record_metrics_from_dict,
    DriftMonitor,
    pass_api_version_to_prometheus,
    RequestMonitor,
    monitor_input,
    monitor_output,
    generate_metrics, 
    categorical_summary_statistics_function
)

# Authentication
# TODO: use secrets, define in separate module, define for prediction & metrics endpoints separately
security = HTTPBasic()


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"stanleyjobson"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"swordfish"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# / authentication

# MODEL, SCHEMA & METRICS

# Load model and schema definitions & train/val workflow metrics from pickled container class
model_and_schema: ModelSchemaContainer = unpickle_bundle("bundle_latest")
# ML model
model = model_and_schema.model

# Model train/test workflow metrics
metrics = model_and_schema.metrics
# pass metrics to prometheus
_ = record_metrics_from_dict(metrics)

# Schema for request (X)
DynamicApiRequest = create_model(
    "DynamicApiRequest", **build_model_definition_from_dict(model_and_schema.req_schema)
)
# Schema for response (y)
DynamicApiResponse = create_model(
    "DynamicApiResponse",
    **build_model_definition_from_dict(model_and_schema.res_schema)
)

# Determine response object value field and type
response_value_field = list(DynamicApiResponse.schema()["properties"])[0]
response_value_type = type(
    DynamicApiResponse.schema()["properties"][response_value_field]["type"]
)

# DRIFT DETECTION
# TODO 2: drift detection wrapper
# store maxsize inputs in a temporal fifo que for drift detection
input_drift = DriftMonitor(columns=schema_to_pandas_columns(model_and_schema.req_schema),
    backup_file="input_fifo.feather", metrics_name_prefix="input_drift_")


output_drift = DriftMonitor(columns=schema_to_pandas_columns(model_and_schema.res_schema),
    backup_file="output_fifo.feather", metrics_name_prefix="output_drift_", maxsize=10,
    summary_statistics_function=categorical_summary_statistics_function
)
# NOTE: if live-scoring, add separate DriftMonitor for model drift
# collect request processing times, sizes and mean by row processing times
processing_drift = RequestMonitor()

# /drift detection

# Start up API
_ = pass_api_version_to_prometheus()
app = FastAPI(
    title="DataHel ML API", description="Generic API for ML model.", version="1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# TODO: for some reason DriftMonitor metrics are only created when loaded from feather.
# new metrics are not created

# metrics endpoint for prometheus
@app.get("/metrics", response_model=dict)
@input_drift.update_metrics_decorator()
@output_drift.update_metrics_decorator()
@processing_drift.update_metrics_decorator()
def get_metrics(username: str = Depends(get_current_username)):
    # if enough data / new data, calculate and record summary statistics
    # TODO 3: sumstat.calculate().set_metrics() decorator wrapper
    return HTMLResponse(generate_metrics())

# TODO 5: predict timing decorator wrapper
@app.post("/predict", response_model=List[DynamicApiResponse])
@monitor_output(output_drift)
@monitor_input(input_drift)
@processing_drift.monitor()
def predict(p_list: List[DynamicApiRequest]):
    # loop trough parameter list
    prediction_values = []
    for p in p_list:
        # convert parameter object to array for model
        parameter_array = [getattr(p, k) for k in vars(p)]
        prediction_values.append(model.predict([parameter_array]))
    # monitor output
    #processing_drift.put(prediction_values)
    # Construct response
    response: List[DynamicApiResponse] = []

    for predicted_value in prediction_values:
        # Cast predicted value to correct type and add response value to response array
        typed_value = response_value_type(predicted_value[0])
        response.append(DynamicApiResponse(**{response_value_field: typed_value}))
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
