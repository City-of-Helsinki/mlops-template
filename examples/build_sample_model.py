import pathlib

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import datetime as dt

from ..model.pickle_model_store import PickleModelStore

df = pd.read_csv("../../analytics_notebook/iris_dataset.csv")
y = df.pop("variety")
X = df

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create linear regression object
classifier = DecisionTreeClassifier(criterion="entropy")

# Train the model using the training sets
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Metrics
metrics_raw = metrics.classification_report(y_test, y_pred, output_dict=True)

# reformat metrics to api/metrics format
metrics_parsed = {}
for species in list(metrics_raw.keys()):
    if isinstance(metrics_raw[species], dict):
        for metric_name in list(metrics_raw[species].keys()):
            metrics_parsed[species + "_" + metric_name] = {
                "value": metrics_raw[species][metric_name],
                "description": "",
                "type": "numeric",
            }

# we can also pass metadata
metrics_parsed["model_update_time"] = {
    "value": dt.datetime.now(),
    "description": "",
    "type": "numeric",
}

# Use dtypes to determine api request and response models
dtypes_x = [{"name": c, "type": X[c].dtype.type} for c in X.columns]
dtypes_y = [{"name": y.name, "type": y.dtype.type}]

CONTEXT_PATH = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = str(CONTEXT_PATH.joinpath(f"local_data/bundle_latest.pickle"))

# Pickle all in single file
model_store = PickleModelStore(bundle_uri=MODEL_PATH)
model_store.persist(classifier, MODEL_PATH, dtypes_x, dtypes_y, metrics_parsed)
print(f"Persisted model to {MODEL_PATH}")

# OR use env variable export MLFLOW_TRACKING_URI=sqlite:////mlflow.sqlite
model_name = "model"
mlflow.set_tracking_uri("sqlite:///mlflow.sqlite")
signature = infer_signature(X_train, y_pred)
mlflow.sklearn.log_model(
    classifier, model_name, signature=signature, registered_model_name=model_name
)
