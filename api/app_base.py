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
# model store path and version if using pickle store
PICKLE_STORE_PATH = os.getenv("PICKLE_STORE_PATH", "../local_data/pickle_store/")
PICKLE_FILENAME = os.getenv("PICKLE_FILENAME", "bundle_latest.pickle")
BUNDLE_PATH = PICKLE_STORE_PATH + PICKLE_FILENAME
CONTEXT_PATH = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = str(CONTEXT_PATH.joinpath(BUNDLE_PATH))

# model store uri, model name and version if using mlflow model store
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:../local_data/mlruns")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", "sqlite:///../local_data/mlflow.sqlite")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "model")
MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION", "latest")

# Introduce SQL logging after init
logging.getLogger().addHandler(SQLiteLoggingHandler(db_uri=LOG_DB))
logging.getLogger().setLevel(logging.INFO)
logging.info("Initialize API application...")

LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS").lower()
if "false" == LOG_PREDICTIONS:
    setting_log_predictions = False
elif "true" == LOG_PREDICTIONS:
    setting_log_predictions = True
else:
    raise ValueError(f"Invalid value for LOG_PREDICTIONS: {LOG_PREDICTIONS}")

# Load model and schema definitions & train/val workflow metrics from model store
model_store_impl = str(os.getenv("MODEL_STORE", "").lower())
logging.info(f"Configured model store: {model_store_impl}")
if "mlflow" == model_store_impl:
    logging.info(f"Loading model from mlflow store: model_name={MLFLOW_MODEL_NAME}, model_version={MLFLOW_MODEL_VERSION}, tracking_uri={MLFLOW_TRACKING_URI}, registry_uri={MLFLOW_REGISTRY_URI}")
    model_store: ModelStore = MlFlowModelStore(
        model_name=MLFLOW_MODEL_NAME,
        model_version=MLFLOW_MODEL_VERSION,
        tracking_uri=MLFLOW_TRACKING_URI,
        registry_uri=MLFLOW_REGISTRY_URI,
    )
elif "pickle" == model_store_impl:
    logging.info(f"Loading model from pickle store: {MODEL_PATH}")
    model_store: ModelStore = PickleModelStore(bundle_uri=MODEL_PATH).load_bundle()
else:
    raise ValueError(f"Invalid value for MODEL_STORE: {model_store_impl}")

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
