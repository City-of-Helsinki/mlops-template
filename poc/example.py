import pandera as pa
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.generic import NDFrame
from pandera.io import from_yaml
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from model_util import save_model, pickle_all, load_schema


def export_to_class_template(classname: str, filename: str, dataframe: DataFrame = None, series: Series = None,):
    columns = None
    types = []
    if dataframe is not None:
        columns = dataframe.columns.to_list()
        for c in columns:
            types.append(df[c].dtype)
    if series is not None:
        columns = [series.name]
        types = [series.dtype]
    if columns is not None:
        file_path = '{}.text'.format(filename)
        with open(file_path, 'w') as f:
            f.write('from pydantic import BaseModel\n\n\n')
            f.write('class {}(BaseModel):\n'.format(classname))
            for c in columns:
                t = types[columns.index(c)]
                f.write('\t{}: {}\n'.format(c, t))
            f.write('\n')
        print('Wrote file {}'.format(file_path))


def get_schema(pandas_obj: NDFrame):
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

def export_to_yaml(filename: str, pandas_obj: NDFrame):
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
        schema_script = schema.to_yaml()

    if schema_script is not None:
        file_path = '{}.yaml'.format(filename)
        with open(file_path, 'w') as f:
            f.write(schema_script)
        print('Wrote file {}'.format(file_path))

# Load sample dataset
df = pd.read_csv('iris_dataset.csv')
y = df.pop('variety')
X = df
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create linear regression object
classifier = DecisionTreeClassifier(criterion='entropy')

# Train the model using the training sets
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Declare api params object
export_to_class_template('Parameters', 'api_params', dataframe=X)
export_to_class_template('Prediction', 'api_response', series=y)

# Declare dataframe types to yaml
export_to_yaml('api_params', X)
export_to_yaml('api_response', y)

# Metrics
print(metrics.classification_report(y_test, y_pred))

save_model(classifier, 'latest_model')

schema_x = get_schema(X)
schema_y = get_schema(y)
pickle_all(classifier, 'bundle', schema_x, schema_y)
sx = from_yaml(load_schema('bundle', 'schema_x'))
print(sx)



