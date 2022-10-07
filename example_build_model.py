import pandas as pd
import pandera as pa
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from model_util import pickle_bundle


# Introspect dataframe schema
def get_schema(pandas_obj: NDFrame) -> str:
    df = None
    if isinstance(pandas_obj, pd.DataFrame):
        df = DataFrame(pandas_obj)
    elif isinstance(pandas_obj, pd.Series):
        df = Series(pandas_obj).to_frame()

    if df is not None:
        schema = pa.infer_schema(df)
        # remove checks
        for c in schema.columns.keys():
            schema.columns[c].checks = []
    return schema.to_yaml()


# Load sample dataset
df = pd.read_csv('iris_dataset.csv')
y = df.pop('variety')
X = df

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create linear regression object
classifier = DecisionTreeClassifier(criterion='entropy')

# Train the model using the training sets
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Metrics
metrics_data = metrics.classification_report(y_test, y_pred)

# Introspect schemas for request and response from dataset
# schema_x = get_schema(X)
# schema_y = get_schema(y)

dtypes_x = [{'name': c, 'type': X[c].dtype.type} for c in X.columns]
dtypes_y = [{'name': y.name, 'type': y.dtype.type}]

print(dtypes_x)
print(dtypes_y)
# Pickle all in single file
pickle_bundle(classifier, 'bundle_latest', dtypes_x, dtypes_y, metrics_data)



