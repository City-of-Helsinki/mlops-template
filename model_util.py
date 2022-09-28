import pickle

from sklearn.base import BaseEstimator


def save_model(model: BaseEstimator, model_id: str):
    file_path = '{model_id}.pickle'.format(model_id=model_id)
    try:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        print("Persisted model to file {}".format(file_path))
    except FileNotFoundError as nfe:
        print("Cannot write to file: ", file_path, nfe)
        return None


def load_schema(model_id: str, schema_id: str):
    model_path = '{model_id}.pickle'.format(model_id=model_id)
    try:
        with open(model_path, "rb") as f:
            dictionary = pickle.load(f)
        return dictionary[schema_id]
    except FileNotFoundError as nfe:
        print("File not found", model_path, nfe)
        return None

def pickle_all(model: BaseEstimator, model_id: str, schema_x, schema_y):
    file_path = '{model_id}.pickle'.format(model_id=model_id)
    try:
        with open(file_path, "wb") as f:
            pickle.dump({'model': model, 'schema_x': schema_x, 'schema_y': schema_y}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Persisted model to file {}".format(file_path))
    except FileNotFoundError as nfe:
        print("Cannot write to file: ", file_path, nfe)
        return None

def load_model(model_id: str):
    model_path = '{model_id}.pickle'.format(model_id=model_id)
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError as nfe:
        print("File not found",model_path,  nfe)
        return None