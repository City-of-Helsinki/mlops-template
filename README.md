# ML Ops template

# Prequisites

The template was developed and tested with:

 - GitHub Codespaces

and

  with Apple MacBook Pro M1 & macOS Ventura 13.0

 - Docker desctop 4.15.0 (93002)
 - VSC 1.74.0
 - VSC Dev Containers Extension v0.245.2

# Known Issues:

 - nbdev_clean git hook may remove 'parameters' tag from notebook cells, even though it is marked as an allowed key in settings.ini. The tag may need to be re-added manually.
 - nbdev documentation related functions may not work out-of-box with arm64 machines such as M1 macbooks because the container installs amd64 quarto

# old / edit: 
This repository presents proof-of-concept for serving a machine learning model trough FastAPI rest api without coding.

## How to use

0. Run `example_build_model.py`
1. Run server: `uvicorn main:app --reload   --reload-include *.pickle` 
2. Prediction api is available: http://127.0.0.1:8000/predict
3. Automatically generated online api documentation is available at: http://127.0.0.1:8000/docs

or via Docker:

0. Run `docker-compose build`
1. Run `docker-compose up`
2.-3. as above
4. To shut down run `docker-compose down`

To pass git version info to api container, run with `GIT_BRANCH="$(git symbolic-ref -q --short HEAD)" GIT_HEAD="$(git rev-parse --short HEAD)" docker-compose up --build`
this adds the version to container labels and passes it on to prometheus.

Run tests with 'docker exec mlops-fastapi-api python -m unittest test_metrics.py'

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


### Locust load testing

Run:

    locust -f locustfile.py -H http://127.0.0.1:8000

Then open http://127.0.0.1:8000 in browser

Change log_mode in main.py to test how logging method affects to troughput.

Locust settings:  Users 1000 spawn rate: 100


## About logging

There are now two different ways to log structured data introduced.
Standard logger provides easy way to transfer structured data to sqlite. 
From structured format it is easy to load data back to dataframe from log database.
Structlog offers nice way to persist structured data as json.