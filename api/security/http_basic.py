import secrets
from http.client import HTTPException

from fastapi import Depends
from fastapi.security import (
    HTTPBasic,
    HTTPBasicCredentials
)
from starlette import status
import os

security = HTTPBasic()


def http_auth_metrics(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username_bytes = os.getenv("METRICS_USERNAME", "").encode("utf8")
    correct_password_bytes = os.getenv("METRICS_PASSWORD", "").encode("utf8")

    current_username_bytes = credentials.username.encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def http_auth_predict(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username_bytes = os.getenv("PREDICT_USERNAME", "").encode("utf8")
    correct_password_bytes = os.getenv("PREDICT_PASSWORD", "").encode("utf8")

    current_username_bytes = credentials.username.encode("utf8")
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
