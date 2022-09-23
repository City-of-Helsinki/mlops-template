# mlops_fast_api_poc

## How to use

0. Export your scikit-learn compatible machine learning model (BaseEstimator) to latest_model.pickle
1. Define api parameters in api_params.py
2. Define api response in api_response.py
3. Run server using uvicorn main:app --reload   


### Startup development api server

     uvicorn main:app --reload   