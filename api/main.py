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

from prometheus_client import generate_latest, Gauge, Counter, Info


from model_util import (
    unpickle_bundle,
    ModelSchemaContainer,
    build_model_definition_from_dict,
    schema_to_pandas_columns,
)

from metrics import (
    DriftQueue,
    convert_time_to_seconds,
    convert_metric_name_to_promql,
    record_metrics_from_dict,
    SummaryStatisticsMetrics,
    default_summary_statistics_function,
)

# API VERSION INFO

# send api version info to prometheus
from git import Repo

repo = Repo(".")
# reponame # TODO: figure out how to get repo name, working tree returns current folder
head = repo.heads[0]
branch = head.name
commit = head.commit.hexsha
api_version_info = Info("api_version", "api repo name, HEAD branch and HEAD commit")
api_version_info.info({"branch": branch, "commit": commit})
# /api version info

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
input_fifo = DriftQueue(
    columns=input_columns, maxsize=10, backup_file="input_fifo.feather"
)
# create summary statistics metrics for the input
input_sumstat = SummaryStatisticsMetrics(metrics_name_prefix="input_drift_")

output_columns = schema_to_pandas_columns(model_and_schema.res_schema)
output_fifo = DriftQueue(
    columns=output_columns, maxsize=10, backup_file="output_fifo.feather"
)
output_sumstat = SummaryStatisticsMetrics(metrics_name_prefix="output_drift_")

# collect request processing times, sizes and mean by row processing times
request_columns = {
    "processing_time_seconds": float,
    "size_rows": int,
    "mean_by_row_processing_time_seconds": float,
}
# the size we use to calculate summary statistics can be less than with the input/output drifts
request_fifo = DriftQueue(
    columns=request_columns, maxsize=3, backup_file="request_fifo.feather"
)

# for the request processing times it's enough if we know the sample size, mean and top values
# define a custom summary statistics function
def request_summary_statistics_function(df: pd.DataFrame) -> pd.DataFrame:
    """ "pandas.DataFrame.describe(include="all", datetime_is_numeric=True).rename({"count": "sample_size"}).loc[["sample_size", "mean", "max"]]"""
    return default_summary_statistics_function(df).loc[["sample_size", "mean", "max"]]


request_sumstat = SummaryStatisticsMetrics(
    metrics_name_prefix="predict_request_",
    summary_statistics_function=request_summary_statistics_function,
)

# /drift detection

# metrics endpoint for prometheus
@app.get("/metrics", response_model=dict)
def get_metrics(username: str = Depends(get_current_username)):
    # if enough data / new data, calculate and record summary statistics
    latest_input = input_fifo.flush()
    if not latest_input.empty:
        input_sumstat.calculate(latest_input).set_metrics()
    latest_output = output_fifo.flush()
    if not latest_output.empty:
        output_sumstat.calculate(latest_output).set_metrics()
    latest_requests = request_fifo.flush()
    if not latest_requests.empty:
        request_sumstat.calculate(latest_requests).set_metrics()
    return HTMLResponse(generate_latest())


request_counter = Counter(
    "predict_requests", "How many requests have been received in total?"
)
prediction_counter = Counter(
    "predict_request_predictions",
    "How many predictions have been made in total / how many input rows have there been in requests in total? ",
)


@app.post("/predict", response_model=List[DynamicApiResponse])
def predict(p_list: List[DynamicApiRequest]):
    request_counter.inc()
    t_begin = time.time()
    # loop trough parameter list
    input_values = []
    prediction_values = []
    for p in p_list:
        prediction_counter.inc()
        # convert parameter object to array for model
        parameter_array = [getattr(p, k) for k in vars(p)]
        prediction_values.append(model.predict([parameter_array]))
        input_values.append(parameter_array)
    # store temporarily in DriftQueues
    input_fifo.put(input_values)
    output_fifo.put(prediction_values)
    # NOTE: if live-scoring, add separate fifo for model drift

    # Construct response
    response: List[DynamicApiResponse] = []

    for predicted_value in prediction_values:
        # Cast predicted value to correct type and add response value to response array
        typed_value = response_value_type(predicted_value[0])
        response.append(DynamicApiResponse(**{response_value_field: typed_value}))
    t_end = time.time()
    processing_time = t_end - t_begin
    request_fifo.put(
        [[processing_time, len(p_list) + 1, processing_time / (len(p_list) + 1.0)]]
    )
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
