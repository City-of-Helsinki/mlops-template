import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import datetime as dt

from model_util import pickle_bundle

# Load sample dataset
df = pd.read_csv("iris_dataset.csv")
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

# Pickle all in single file
pickle_bundle(classifier, "bundle_latest", dtypes_x, dtypes_y, metrics_parsed)
