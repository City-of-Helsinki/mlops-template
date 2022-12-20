import logging
import os
import pathlib
from log.sqlite_logging_handler import SQLiteLoggingHandler
from metrics.prometheus_metrics import (
    RequestMonitor,
    DriftMonitor,
    distribution_summary_statistics,
    categorical_summary_statistics,
    pass_api_version_to_prometheus,
    record_metrics_from_dict,
)
import sys


# LOCAL IMPORTS
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)
from model_store import ModelStore, MlFlowModelStore, PickleModelStore

# Do other local imports in similar manner if needed, i.e.
# from ml_pipe import your_module


LOG_DB = "sqlite:///../local_data/logs.sqlite"
BUNDLE = os.getenv("BUNDLE", "../local_data/bundle_latest.pickle")
CONTEXT_PATH = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = str(CONTEXT_PATH.joinpath(BUNDLE))


# Introduce SQL logging after init
logging.getLogger().addHandler(SQLiteLoggingHandler(db_uri=LOG_DB))
logging.getLogger().setLevel(logging.INFO)
logging.info("Initialize API application...")


logging.info(f"Loading model bundle: {MODEL_PATH}")

try:
    if "false" == str(os.environ["LOG_PREDICTIONS"]).lower():
        setting_log_predictions = False
    else:
        setting_log_predictions = True
except KeyError:
    setting_log_predictions = False

# Load model and schema definitions & train/val workflow metrics from model store
model_store_impl = str(os.getenv("MODEL_STORE", "").lower())
logging.info(f"Configured model store: {model_store_impl}")
if "mlflow" == model_store_impl:
    model_store: ModelStore = MlFlowModelStore(
        tracking_uri="file:../local_data/mlruns",
        registry_uri="sqlite:///../local_data/mlflow.sqlite",
    )
else:
    model_store: ModelStore = PickleModelStore(bundle_uri=MODEL_PATH)

# ML model
model = model_store.model

# Model train/test workflow metrics
train_val_metrics = model_store.train_metrics

# pass metrics to prometheus
train_metrics = record_metrics_from_dict(train_val_metrics)

# What for is this?
version_info = pass_api_version_to_prometheus()

# Schema for request (X)
DynamicApiRequest = model_store.request_schema_class
# Schema for response (y)
DynamicApiResponse = model_store.response_schema_class

# Determine response object value field and type
response_value_field = model_store.response_value_field
response_value_type = model_store.response_value_type

processing_drift = RequestMonitor(backup_file="../local_data/processing_fifo.feather")

input_drift = DriftMonitor(
    columns=model_store.request_columns,
    backup_file="../local_data/input_fifo.feather",
    metrics_name_prefix="input_drift_",
    summary_statistics_function=distribution_summary_statistics,
)

output_drift = DriftMonitor(
    columns=model_store.response_columns,
    backup_file="../local_data/output_fifo.feather",
    metrics_name_prefix="output_drift_",
    summary_statistics_function=categorical_summary_statistics,
)
