import logging
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.params import Depends
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware

from app_base import (
    input_drift,
    output_drift,
    processing_drift,
    DynamicApiResponse,
    DynamicApiRequest,
    model,
    setting_log_predictions,
    response_value_type,
    response_value_field,
)
from metrics.prometheus_metrics import monitor_output, monitor_input, generate_metrics
from security.http_basic import http_auth_metrics

# Start up API
app = FastAPI(
    title="DataHel ML API", description="Generic API for ML model.", version="1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


@app.get("/metrics", response_model=dict)
@input_drift.update_metrics_decorator()  # calculate drift metrics & pass to prometheus
@output_drift.update_metrics_decorator()
@processing_drift.update_metrics_decorator()
def get_metrics(username: str = Depends(http_auth_metrics)):
    return HTMLResponse(generate_metrics())


@app.post("/predict", response_model=List[DynamicApiResponse])
@monitor_output(output_drift)  # add new data to fifos
@monitor_input(input_drift)
@processing_drift.monitor()
def predict(
    p_list: List[DynamicApiRequest],
):  # , username: str = Depends(auth_predict.auth)):
    # loop trough parameter list
    prediction_values = []
    for p in p_list:
        # pickle: convert parameter object to array for model
        # parameter_array = [getattr(p, k) for k in vars(p)]
        # prediction = model.predict([parameter_array])
        # mlflow: convert to json
        parameter_dict = {k: getattr(p, k) for k in vars(p)}
        X = pd.json_normalize(parameter_dict)
        prediction = model.predict(X)
        # append prediction
        prediction_values.append(prediction)
        if setting_log_predictions:
            logging.info({"prediction": str(prediction), "request_parameters": p})
    # Construct response
    response: List[DynamicApiResponse] = []

    # Cast predicted values to correct type and add response value to response array
    for predicted_value in prediction_values:
        typed_value = response_value_type(predicted_value[0])
        response.append(DynamicApiResponse(**{response_value_field: typed_value}))

    return response


if __name__ == "__main__":
    # logging.info(f"Example post data: {json.dumps(DynamicApiRequest())}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
