import os
import logging
from log.sqlite_logging_handler import SQLiteLoggingHandler
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.params import Depends
from fastapi.responses import HTMLResponse
from pydantic import create_model
from starlette.middleware.cors import CORSMiddleware
import secrets

# LOGGING
logging.getLogger().addHandler(SQLiteLoggingHandler())
logging.getLogger().setLevel(logging.INFO)

try:
    if "false" == str(os.environ["LOG_PREDICTIONS"]).lower():
        setting_log_predictions = False
    else:
        setting_log_predictions = True
except KeyError:
    setting_log_predictions = False

# LOCAL IMPORTS
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
    categorical_summary_statistics,
    distribution_summary_statistics,
    # simple_text_summary_statistics # uncomment to use
    # you can define more summary statistics functions in metrics or use lambdas
)


# AUTHENTICATION

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


# MODEL, SCHEMA & TRAIN METRICS

# Load model and schema definitions & train/val workflow metrics from pickled container class
model_and_schema: ModelSchemaContainer = unpickle_bundle("bundle_latest")
# ML model
model = model_and_schema.model

# Model train/test workflow metrics
train_val_metrics = model_and_schema.metrics
# pass metrics to prometheus
_ = record_metrics_from_dict(train_val_metrics)

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

# store inputs, outputs & processing data in temporal fifo queues
input_drift = DriftMonitor(
    columns=schema_to_pandas_columns(model_and_schema.req_schema),
    backup_file="input_fifo.feather",
    metrics_name_prefix="input_drift_",
    summary_statistics_function=distribution_summary_statistics,
)

output_drift = DriftMonitor(
    columns=schema_to_pandas_columns(model_and_schema.res_schema),
    backup_file="output_fifo.feather",
    metrics_name_prefix="output_drift_",
    summary_statistics_function=categorical_summary_statistics,
)

processing_drift = RequestMonitor()
# NOTE: if live-scoring, add separate DriftMonitor for model drift


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

# metrics endpoint for prometheus
@app.get("/metrics", response_model=dict)
@input_drift.update_metrics_decorator()  # calculate drift metrics & pass to prometheus
@output_drift.update_metrics_decorator()
@processing_drift.update_metrics_decorator()
def get_metrics(username: str = Depends(get_current_username)):
    return HTMLResponse(generate_metrics())


@app.post("/predict", response_model=List[DynamicApiResponse])
@monitor_output(output_drift)  # add new data to fifos
@monitor_input(input_drift)
@processing_drift.monitor()
def predict(p_list: List[DynamicApiRequest]):
    # loop trough parameter list
    prediction_values = []
    for p in p_list:
        # convert parameter object to array for model
        parameter_array = [getattr(p, k) for k in vars(p)]
        prediction = model.predict([parameter_array])
        prediction_values.append(prediction)
        if setting_log_predictions:
            logging.info({"prediction": str(prediction), "request_parameters": p})
    # Construct response
    response: List[DynamicApiResponse] = []

    for predicted_value in prediction_values:
        # Cast predicted value to correct type and add response value to response array
        typed_value = response_value_type(predicted_value[0])
        response.append(DynamicApiResponse(**{response_value_field: typed_value}))

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)