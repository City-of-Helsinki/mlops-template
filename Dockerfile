FROM python:3.9-bullseye

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN python -m pip install -r requirements.txt