# ML Ops template

## Creating a new repo from this template

(add link to github instructions)

## How to work with the template:

The template assumes working within a Docker container. Local install may work but is not recommended.

### Codespaces:

Launch codespaces on your repository. For further configuration, edit `.devcontainer/devcontainer.json`

### Local:

Start by cloning your repository.
The template has three modes: two for development `vsc` and `jupyterlab` and one for running the `api`. The mode is given to Docker as an environment variable `MODE`.

#### VSC:

Install VSC and Dev Containers Extension. When you open your repository in VSC, it automatically detects `.devcontainer/devcontianer.json` and suggests opening the repository in a container. Alternatively, press `cmd/ctrl shift P`, type `Remote-Container: Rebuild and Reopen in Container` and press enter. VSC builds the container and attaches to it. 

#### Jupyterlab:

The template installs jupyterlab within the container. To work in JupyterLab, run `MODE=jupyterlab docker-compose up`. The container starts JupyterLab. To access jupyterlab, copy the address from the terminal to your browser. 

#### Working offline:

0. Clone your repo on a network connected device or use codespaces.
1. Add required python packages to `requirements/requirements.in`.
2. Build the image, and within the `requirements` folder run the script `./update_requirements.sh`. 
3. Rebuild the image.
4. Pull the image and transfer it to the offline device. The offline device must have Docker installed. 
5. Run container in jupyterlab mode for development. 

### Running the API:

To start the api as the container entrypoint, run `MODE=api docker-compose up`. This starts the API and leaves the container running. 

To develop interactively with the API running, you may start the API from within your VSC / jupyterlab terminal by running `uvicorn main:app --reload --reload-include *.pickle --host 0.0.0.0` within the api folder. 

## Prequisites

The template was developed and tested with:

 - GitHub Codespaces

and MacBook Pro M1 & macOS Ventura 13.0 with:

 - Docker desctop 4.15.0 (93002)
 - VSC 1.74.0
 - VSC Dev Containers Extension v0.245.2

Additional configuration may be required for other systems.

## Known issues:

 - nbdev_clean git hook may remove 'parameters' tag from notebook cells, even though it should be an allowed key as it is listed in settings.ini. The tag may need to be re-added manually to allow notebook parameterization with papermill.
 - nbdev documentation related functions may not work out-of-box with arm64 machines such as M1 macbooks because the container installs amd64 version of Quarto.

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