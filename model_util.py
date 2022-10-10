import pickle
from typing import List

import numpy as np
from pandera.io import from_yaml
from sklearn.base import BaseEstimator


class ModelSchemaContainer:
    model: BaseEstimator
    req_schema: str
    res_schema: str
    metrics: str


def unpickle_bundle(model_id: str) -> ModelSchemaContainer:
    model_path = '{model_id}.pickle'.format(model_id=model_id)
    try:
        with open(model_path, "rb") as f:
            container = pickle.load(f)

        return container
    except FileNotFoundError as nfe:
        print("File not found", model_path, nfe)
        return None


def pickle_bundle(model: BaseEstimator, model_id: str, schema_x, schema_y, metrics):
    file_path = '{model_id}.pickle'.format(model_id=model_id)
    try:
        with open(file_path, "wb") as f:
            container: ModelSchemaContainer = ModelSchemaContainer()
            container.model = model
            container.req_schema = schema_x
            container.res_schema = schema_y
            container.metrics = metrics
            pickle.dump(container, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Persisted model to file {}".format(file_path))
    except FileNotFoundError as nfe:
        print("Cannot write to file: ", file_path, nfe)
        return None

# Generates dynamically request and response classes for openapi schema and swagger documentation
def build_parameter_model_definition(yaml_schema: str):
    schema = from_yaml(yaml_schema)
    return build_model_definition_from_schema(schema)

def build_model_definition_from_schema(schema):
    fields = {}
    for col in schema.columns:
        t = schema.dtypes[col].type.type
        # convert object types to string
        if t == np.object_ or t == object:
            t = str
        name = col
        fields[name] = (t, ...)
    return fields

def build_model_definition_from_dict(column_types: List[dict]):
    fields = {}
    for coltype in column_types:
        t = coltype['type']
        # convert object types to string
        if t == np.object_ or t == object:
            t = str
        name = coltype['name']
        fields[name] = (t, ...)
    return fields