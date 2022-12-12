from abc import abstractmethod


class ModelStore:
    model = None
    train_metrics = None
    request_schema_class = None
    response_schema_class = None
    request_columns: dict = None
    response_columns: dict = None
    response_value_type = None
    response_value_field = None

    @abstractmethod
    def persist(self, classifier, param, dtypes_x, dtypes_y, metrics_parsed):
        pass
