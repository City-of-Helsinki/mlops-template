# mlops_fast_api_poc

## How to use

0. Export your scikit-learn compatible machine learning model (BaseEstimator) to latest_model.pickle
1. Define api parameters in api_params.py, include all feature columns in same order.
2. Define api response in api_response.py, set value attribute to correct type. Default is string.
3. Run server using uvicorn main:app --reload   

See example.py for reference.

## Api documentation

Swagger online documentation is available at: http://127.0.0.1:8000/docs


### Startup development api server

     uvicorn main:app --reload   