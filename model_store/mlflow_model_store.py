import logging
from typing import List

import mlflow.pyfunc
import numpy as np
from mlflow.pyfunc import PyFuncModel
from pydantic import create_model
from pydantic.fields import FieldInfo

from .model_store import ModelStore


class MlFlowModelStore(ModelStore):
    def __init__(
        self,
        model_name="model",
        model_version="latest",
        registry_uri="sqlite:///../local_data/mlflow.sqlite",
        tracking_uri="file:../local_data/mlruns",
    ):

        # Tell mlflow where tracking server data is (could be remote)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)

        # Retrieve model by name and version
        model: PyFuncModel = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )

        # Get model signature for schema
        metadata = model.metadata.to_dict()
        signature = model.metadata.signature

        # Model
        self.model = model

        # metrics
        run = mlflow.get_run(run_id=metadata["run_id"])
        self.train_metrics = self.__parse_metrics(run.data.metrics)

        request_types = []
        for i, n in enumerate(signature.inputs.input_names()):
            request_types.append(
                {"name": n, "type": signature.inputs.numpy_types()[i].type}
            )

        response_types = []
        for i, n in enumerate(signature.outputs.input_names()):
            response_types.append(
                {"name": n, "type": signature.outputs.numpy_types()[i].type}
            )

        # Schema for request (X)
        self.request_schema_class = self.__create_pydantic_model(
            "DynamicApiRequest", request_types
        )
        # Schema for response (y)
        self.response_schema_class = self.__create_pydantic_model(
            "DynamicApiResponse", response_types
        )

        self.response_value_field = list(
            self.response_schema_class.schema()["properties"]
        )[0]

        self.response_value_type = type(
            self.response_schema_class.schema()["properties"][
                self.response_value_field
            ]["type"]
        )
        self.request_columns = self.__schema_to_pandas_columns(request_types)
        self.response_columns = self.__schema_to_pandas_columns(response_types)
        logging.info(f"Retrieved model with run id {metadata['run_id']}")

    @staticmethod
    def __parse_metrics(metrics_raw):
        """
        Parse training metrics from MLFlow to be passed for monitoring.
        """
        metrics_parsed = {}
        for key in list(metrics_raw.keys()):
            metrics_parsed[key] = {
                "value": metrics_raw[key],
                "description": "",
                "type": "numeric",  # WARNING: may be restrictive!
            }

        return metrics_parsed

    @staticmethod
    def __build_model_definition_from_dict(column_definitions: List[dict]):
        fields = {}
        for coltype in column_definitions:
            t = coltype["type"]
            # convert object types to string
            if t == np.object_ or t == object:
                t = str

            name = coltype["name"]
            if not name:
                name = "prediction"
            fields[name] = (t, FieldInfo(title=name))
        return fields

    def __create_pydantic_model(self, class_name: str, column_definitions: List[dict]):
        return create_model(
            class_name, **self.__build_model_definition_from_dict(column_definitions)
        )

    def persist(self, classifier, dtypes_x, dtypes_y, metrics_parsed):
        pass

    @staticmethod
    def __schema_to_pandas_columns(schema):
        """
        Convert ModelSchemaContainer schemas to pandas column definitions
        """
        ret = {}
        for row in schema:
            ret[row["name"]] = row["type"]
        return ret
