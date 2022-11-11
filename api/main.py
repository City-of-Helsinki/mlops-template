from typing import List

import uvicorn
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.params import Depends
from fastapi.responses import HTMLResponse
from pydantic import create_model
from starlette.middleware.cors import CORSMiddleware

import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from model_util import (
    unpickle_bundle,
    ModelSchemaContainer,
    build_model_definition_from_dict,
    schema_to_pandas_columns,
)

from metrics import (
    FifoOverwriteDataFrame,
    convert_time_to_seconds,
    convert_metric_name_to_promql,
    record_metrics_from_dict,
    SummaryStatisticsMetrics,
)

from prometheus_client import generate_latest

# Authentication

app = FastAPI()

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

# Load model and schema definitions & train/val workflow metrics from pickled container class
model_and_schema: ModelSchemaContainer = unpickle_bundle("bundle_latest")
# ML model
model = model_and_schema.model

# metrics
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

# Start up API
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

# DRIFT DETECTION
# store maxsize inputs in a temporal fifo que for drift detection
input_columns = schema_to_pandas_columns(model_and_schema.req_schema)
input_fifo = FifoOverwriteDataFrame(
    columns=input_columns, maxsize=10, backup_file="input_fifo.feather"
)
# create summary statistics metrics for the input
input_sumstat = SummaryStatisticsMetrics(
    columns=input_columns, metrics_name_prefix="input_drift_"
)
# calculate summary statistics either periodically or when metrics is called
# example in get_metrics below
# /drift detection

# metrics endpoint for prometheus
@app.get("/metrics", response_model=dict)
def get_metrics(username: str = Depends(get_current_username)):
    # if enough data / new data, calculate and record input summary statistics
    latest_input = input_fifo.flush()
    if not latest_input.empty:
        input_sumstat.calculate(latest_input).set_metrics()
    # print(input_sumstat.get_sumstat())
    return HTMLResponse(generate_latest())


# TODO: PROMEHEUS:
# input / output drift DONE
# processing:
#   - time (total / hist )
#   - general resource usage
#   - request counter


@app.post("/predict", response_model=List[DynamicApiResponse])
def predict(p_list: List[DynamicApiRequest]):
    # loop trough parameter list
    prediction_values = []
    for p in p_list:
        # convert parameter object to array for model
        parameter_array = [getattr(p, k) for k in vars(p)]
        prediction_values.append(model.predict([parameter_array]))
        # store input in fifo
        input_fifo.put([parameter_array])
    # Construct response
    response: List[DynamicApiResponse] = []

    for predicted_value in prediction_values:
        # Cast predicted value to correct type and add response value to response array
        typed_value = response_value_type(predicted_value[0])
        response.append(DynamicApiResponse(**{response_value_field: typed_value}))
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
