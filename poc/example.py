
import pandas as pd
from pandas import DataFrame, Series
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from model_util import save_model


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
        file_path = '{}.template'.format(filename)
        with open(file_path, 'w') as f:
            f.write('from pydantic import BaseModel\n\n\n')
            f.write('class {}(BaseModel):\n'.format(classname))
            for c in columns:
                t = types[columns.index(c)]
                f.write('\t{}: {}\n'.format(c, t))
            f.write('\n')
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

# Metrics
print(metrics.classification_report(y_test, y_pred))

save_model(classifier, 'latest_model')
