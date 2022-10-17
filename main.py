
from typing import List

import structlog
import uvicorn
from fastapi import FastAPI, Security, HTTPException
import logging
from fastapi.params import Depends
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import create_model
from starlette.middleware.cors import CORSMiddleware
from starlette.status import HTTP_403_FORBIDDEN

from log.sqlite_logging_handler import SQLiteLoggingHandler
from model_util import unpickle_bundle, ModelSchemaContainer, build_model_definition_from_dict

# Logging configuration
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=False,
)

logging.getLogger().addHandler(SQLiteLoggingHandler())
logging.getLogger().setLevel(logging.INFO)
log = structlog.get_logger().bind()

# Authentication
API_KEY = "apiKey123"   # TODO: where we want to keep api keys
API_KEY_NAME = "X-API-KEY"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="API key is missing or incorrect in header: {}.".format(API_KEY_NAME)
        )
# / authentication

# Load model and schema definitions from pickled container class
model_and_schema: ModelSchemaContainer = unpickle_bundle('bundle_latest')
# ML model
model = model_and_schema.model

# metrics
metrics = model_and_schema.metrics

# Schema for request (X)
DynamicApiRequest = create_model('DynamicApiRequest', **build_model_definition_from_dict(model_and_schema.req_schema))
# Schema for response (y)
DynamicApiResponse = create_model('DynamicApiResponse', **build_model_definition_from_dict(model_and_schema.res_schema))

# Determine response object value field and type
response_value_field = list(DynamicApiResponse.schema()['properties'])[0]
response_value_type = type(DynamicApiResponse.schema()['properties'][response_value_field]['type'])

# Start up API
app = FastAPI(title="DataHel ML API", description="Generic API for ML model.", version="1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # allow all origins
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


@app.get("/metrics", response_model=dict)
def get_metrics(api_key: APIKey = Depends(get_api_key)):
    return metrics


@app.post("/predict", response_model=List[DynamicApiResponse])
def predict(p_list: List[DynamicApiRequest]):
    # loop trough parameter list
    prediction_values = []
    for p in p_list:
        # convert parameter object to array for model
        parameter_array = [getattr(p, k) for k in vars(p)]
        prediction = model.predict([parameter_array])[0]
        prediction_values.append(prediction)
        log.info({'prediction': prediction, 'request_parameters': p})
        logging.info({'prediction': prediction, 'request_parameters': p})

    # Construct response
    response: List[DynamicApiResponse] = []

    for predicted_value in prediction_values:
        # Cast predicted value to correct type and add response value to response array
        typed_value = response_value_type(predicted_value[0])
        response.append(DynamicApiResponse(**{response_value_field: typed_value}))
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)