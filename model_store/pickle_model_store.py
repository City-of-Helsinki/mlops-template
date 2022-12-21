import logging
import pickle
from typing import List

import numpy as np
from pydantic import create_model
from pydantic.fields import FieldInfo
from sklearn.base import BaseEstimator

from .model_store import ModelStore


class ModelSchemaContainer:
    """
    req & res schema: [{'name': value, 'type': dtype}]
    """

    model: BaseEstimator
    req_schema: List[dict]
    res_schema: List[dict]
    metrics: dict


class PickleModelStore(ModelStore):
    def __init__(self, bundle_uri="local_data/bundle_latest.pickle"):
        self.bundle_uri = bundle_uri

    def persist(self, classifier, param, dtypes_x, dtypes_y, metrics_parsed):
        return self.__pickle_bundle(classifier, param, dtypes_x, dtypes_y, metrics_parsed)

    def get_model(self) -> BaseEstimator:
        if not self.model:
            self.load_bundle()
        return self.model

    def load_bundle(self):
        # try:
        logging.info(
            f"Open {self.bundle_uri}",
        )
        bundle = self.__load_pickled_bundle(self.bundle_uri)
        self.model = bundle.model
        self.train_metrics = bundle.metrics
        # Schema for request (X)
        self.request_schema_class = self.__create_pydantic_model(
            "DynamicApiRequest", bundle.req_schema
        )
        # Schema for response (y)
        self.response_schema_class = self.__create_pydantic_model(
            "DynamicApiResponse", bundle.res_schema
        )
        self.response_value_field = list(
            self.response_schema_class.schema()["properties"]
        )[0]

        self.response_value_type = type(
            self.response_schema_class.schema()["properties"][
                self.response_value_field
            ]["type"]
        )
        self.request_columns = self.__schema_to_pandas_columns(bundle.req_schema)
        self.response_columns = self.__schema_to_pandas_columns(bundle.res_schema)
        return self

    @staticmethod
    def __load_pickled_bundle(bundle_uri: str) -> ModelSchemaContainer:
        try:
            with open(bundle_uri, "rb") as f:
                container = pickle.load(f)
            return container
        except FileNotFoundError as nfe:
            logging.warning(f"File not found {bundle_uri}", nfe)
            return None

    @staticmethod
    def __pickle_bundle(
        model: BaseEstimator,
        bundle_uri: str,
        schema_x=None,
        schema_y=None,
        metrics: str = None,
    ):
        try:
            with open(bundle_uri, "wb") as f:
                container: ModelSchemaContainer = ModelSchemaContainer()
                container.model = model
                container.req_schema = schema_x
                container.res_schema = schema_y
                container.metrics = metrics
                pickle.dump(container, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Persisted model to file  {bundle_uri}")
        except FileNotFoundError as nfe:
            logging.warning(f"Cannot write to file: {bundle_uri}", nfe)

    @staticmethod
    def __build_model_definition_from_dict(column_types: List[dict]):
        fields = {}
        for coltype in column_types:
            t = coltype["type"]
            # convert object types to string
            if t == np.object_ or t == object:
                t = str
            name = coltype["name"]
            fields[name] = (t, FieldInfo(title=name))
        return fields

    def __create_pydantic_model(self, class_name, column_types: List[dict]):
        return create_model(
            class_name, **self.__build_model_definition_from_dict(column_types)
        )

    @staticmethod
    def __schema_to_pandas_columns(schema):
        """
        Convert ModelSchemaContainer schemas to pandas column definitions
        """
        ret = {}
        for row in schema:
            ret[row["name"]] = row["type"]
        return ret
