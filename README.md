# mlops_fast_api_poc

## How to use

0. Export your scikit-learn compatible machine learning model (BaseEstimator) to latest_model.pickle
1. Define api parameters in api_params.py, include all feature columns in same order.
2. Define api response in api_response.py, set value attribute to correct type. Default is string.
3. Run server using uvicorn main:app --reload   
4. Prediction api is available: http://127.0.0.1:8000/predict
See example.py for reference.

## Api documentation

Swagger online documentation is available at: http://127.0.0.1:8000/docs

Test request using example model:

    curl -X 'POST' \
      'http://127.0.0.1:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '[
      {
        "sepal_length": 6.7,
        "sepal_width": 3.5,
        "petal_length": 5.2,
        "petal_width": 2.3
      },
      {
        "sepal_length": 6.6,
        "sepal_width": 3,
        "petal_length": 4.4,
        "petal_width": 1.4
      }
    ]'

### Startup development api server

     uvicorn main:app --reload   --reload-include *.pickle  