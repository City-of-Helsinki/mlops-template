# ML Ops template

![Python version](https://img.shields.io/badge/python-3.10-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/City-of-Helsinki/mlops-template)
![GitHub issues](https://img.shields.io/github/issues/City-of-Helsinki/mlops_template)
![GitHub issues](https://img.shields.io/github/issues-closed-raw/City-of-Helsinki/mlops_template)
![GitHub forks](https://img.shields.io/github/forks/City-of-Helsinki/mlops_template)
![GitHub stars](https://img.shields.io/github/stars/City-of-Helsinki/mlops_template)
![GitHub license](https://img.shields.io/github/license/City-of-Helsinki/mlops_template)

> Generic repository template for small scale MLOps.

## About

Create ETL & machine learning pipelines, model store, API, monitoring & logging - all in a single container, with minimum setup required! 

![mlops process chart & architecture](mlops_template.png)
*ML Ops template generic structure and process flowchart*

The generic ML Ops process and structure of the template is described in the figure above. The core structure of the template is built on three elements: a notebook-based ML pipeline (`ml_pipe/`), a model store with two alternatives (`model_store/`), and and API with generic logging and monitoring functions (`api/`).

In addition the repository contains `requirements/` folder for managing requirements, `Dockerfile` and `compose.yml` for building and running the container, and `examples/` showing two simple examples on how to create a ML model and save it into a model store. The api template is created so that it works with the examples as-is, but is built modular and easy to adopt for your use case.

## Creating a new repo from this template

For your ML project, create new repo from this template ([see GitHub instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)). Commit history will be cleaned and the authors won't have access to your new repository (although sharing your work is encouraged).

> NOTE: Updates to the template cannot be automatically synchronized to child projects!

Checklist for creating the repository:
1. Use dashes '-' instead of underscores '_' for the repository name. This is an [nbdev](https://nbdev.fast.ai) convention.
2. Set your organization as the owner, if you represent one.
3. Once the repository is ready, edit nbdev config file `settings.ini` & `ml_pipe/Makefile` according to your project information. The `lib_path` variable should be set to `ml_pipe/your_repository_name_but_with_underscores_instead_of_dashes`.


## Working with the template:

The template assumes working within containers. Non-container development is possible but not recommended. 

### Option 1: Codespaces

Repositories generated from the template are development-ready with [GitHub Codespaces](https://docs.github.com/en/codespaces/overview).

Codespaces builds a container according to settings in `.devcontainer/devcontainer.json` and attaches to it in `vsc` mode (see 'Running the container' below).

Just launch your repository in a Codespace and start working.

### Option 2: Local


Begin by selecting your tools. Here are two examples for local development: Podman + JupyterLab (recommended), and Docker + VSC (*Visual Studio Code*).

The template container has three modes: two for development `vsc` (default) and `jupyterlab`, and `api` for running the API. The mode is given an environment variable `MODE` when running the container.

*Podman + JupyterLab*

Podman is an open source container runtime tool, that can also operate Docker containers, and shares most of the syntax with Docker. While it is possible to configure VSC Dev Container or VSC Remote Container to use Podman or other open source container framework (see instructions with [minikube](https://benmatselby.dev/post/vscode-dev-containers-minikube/) or [Podman desktop](https://developers.redhat.com/articles/2023/02/14/remote-container-development-vs-code-and-podman#)), this requires some effort compared to Docker with Docker Desktop. In addition, third party Docker extensions may not work. Therefore, it is recommended to develop locally with Jupyterlab, which comes installed within the template container.

```bash
# On MacOS

# Install podman, docker and hyperkit
# These are the only requirements for your system - 
# everything else, including python and it's packages 
# are installed within the container!

$ brew install hyperkit
$ brew install docker docker-compose
$ brew install podman podman-compose
$ brew install podman-desktop
    
# Init podman machine
# (you can configure resources allocated to podman with params)
$ podman machine init --cpus=2

# Start podman machine
$ podman machine start

# Build container
$ cd your_repo
$ podman-compose build

# Start dev container with JupyterLab
$ MODE=jupyterlab podman-compose up

# To stop the dev container
# (in another terminal window or tab)
$ cd your_repo
$ podman-compose down

# To stop podman
$ podman machine stop
```

Check out [Podman docs](https://docs.podman.io/en/latest/index.html) for more information and instructions. 

*Docker Desktop + Visual Studio Code*

[Docker Desktop](https://www.docker.com/) is an open core tool for managing Docker containers. Please note that most non-personal use of Docker Desktop now requires paid license subscription.

To use the template with docker desktop, install and start docker desktop according to the instructions. VSC Extensions work with Docker Desctop with default settings.

Visual Studio Code is an IDE that supports container development. Install VSC and Dev Containers Extension. When you open your repository in VSC, it automatically detects `.devcontainer/devcontianer.json` and suggests opening the repository in a container. Alternatively, press `cmd/ctrl shift P`, type `Remote-Container: Rebuild and Reopen in Container` and press enter. VSC builds the container and attaches to it. 

### Running the API:

To start the API as the container entrypoint, run 
`MODE=api podman-compose up` or `MODE=api docker-compose up This loads the latest trained model from model store, starts the API and leaves the container running. The API requires an existing model store and a stored model to function. 

> NOTE: To launch container in `api` mode, you must first train a model and save it to model store on a persistent volume or mapping, i.e. change the `local_data` volume type in compose and rebuild the container. To avoid accidentaly leaking sensitive data, model stores are by default saved to `tmpfs` storage that is removed every time the container is stopped. The `api` mode will not work without changing this setup.

To develop interactively with the API running, you may start the API from within your VSC / jupyterlab terminal by running `uvicorn main:app --reload --reload-include *.pickle --host 0.0.0.0` within the API folder of the container. This does not require changing the volume types.

To specify model store and model version to load, use environment variables as specified in `api/app_base.py`. The default option loads latest model from pickle store.

### Working offline:

The repository allows developing and running ML models completely offline.

Steps to offline install:

0. Clone your repo on a network connected device or use codespaces.
1. Add required python packages to `requirements/requirements.in`. Try to include all packages you might require, because adding them later without internet access is rather difficult.
2. Build the image, and within the `requirements` folder run the script `./update_requirements.sh`. 
3. Rebuild the image.
4. Pull the image and transfer it to the offline device. The offline device must have Docker installed. 
5. Start container in the `jupyterlab` mode with your choice of container tools.

The API works offline, too, but requires network access for external clients.

## Examples

The `examples/` folder contains simplified single-notebook examples on how to create, train, evaluate and deploy a ML model to model store. There are two notebooks due to two alternatives for the model store. You can try out the API by:

0. Running an example notebook.
1. Running the API with corresponding model store env. 
2. Prediction API is available: http://127.0.0.1:8000/predict
3. Automatically generated online API documentation is available at: http://127.0.0.1:8000/docs
4. Real time metrics are available at: http://127.0.0.1:8000/metrics and at the built-in example Prometheus server at http://127.0.0.1:9090

> NOTE: if working in Codespaces, right-click the links from README to access the port-forwarded endpoints!

The following API request calls for the example model to make a prediction on two new data points:

    curl -X 'POST' \
      'http://127.0.0.1:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -c ':' \
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


## Managing requirements & dependencies

The template uses [pip-tools](https://pypi.org/project/pip-tools/) for calculating dependencies.
Requirements are listed in `*.in` files inside the `requirements/` folder. They are separated in four files:
prefix `api` is for API, `dev` is for developer tools, `model_store` is for model store and basic `requirements.in` for everything else. You should only need to edit `requirements.in`, unless you wish to deviate or add on to the core tools the template was built on.

To update `requirements.txt`, run script `./update_requirements.sh` INSIDE the `requirements/` folder. Do not edit `requirements.txt` manually or with `pip freeze`.

Install new requirements with `pip install -r requirements/requirements.txt` or by rebuilding the container.

The template installs python 3.10 (Ubuntu default).
To specify another python version, edit the dockerfile to set up a virtual environment. The template is tested with python 3.10, but is expected to work with 3.8 and newer. However, other versions may require additional configuration.

## ML Pipeline

The data is processed, models created and evaluated in jupyter notebooks. 
There are template notebooks in `ml_pipe/` folder to help you get started.

- `00_data_etl.ipynb` is for loading and cleaning data or defining functions for doing so. 
- `01_model_class.ipynb` is for defining a ML model class or custom wrappers for existing models. Not needed if you use existing implementations such as sklearn models & simple pipes.
- `02_train_test_validate.ipynb` is for training, testing and validating your ML model. 
- `03_workflow.ipynb` is for running the other notebooks automatically. *Ultimately you should be able to load data, train, test, validate and update a model to model store only by running this notebook inside the container.* For complete automation, you can set up this workflow to be triggered by a change in the ML metrics.

To enhance working with notebooks, two tools are included: nbdev and papermill.

Nbdev allows developing clean code straight from notebooks, with no copy-pasting involved.
The template is planned for taking use of the following two nbdev functions:
 - nbdev_clean  - Clean notebooks to avoid merge conflicts
 - nbdev_export - Export notebooks to a Python module (`ml_pipe/your_module`, set the module name in `settings.ini`)

Other features may be useful but are not guaranteed to work out-of-box with the template. See list of commands with `nbdev_help`.

Check out [nbdev tutorial](https://nbdev.fast.ai/tutorials/tutorial.html) for more info on the tool. You can get most out of nbdev following [fastcore](https://fastcore.fast.ai) principles, which we do yet fully utilize, but is a practice of coding we are headed towards.

[Papermill](https://papermill.readthedocs.io/en/latest/) allows running notebooks from python, parameterized. 


## Model Store

The template includes two alternatives for a model store.

First is a simple pickle store, that stores ML model, input and output schemas and training metrics to a pickle and allows unwrapping these with ease. The pickle store is developed to work with all `scikit-learn` models, and requires additional configuration for other types of models, including non-inheriting custom wrappers for sklearn models. A `date - branch - head - setup` versioning is used for naming the model instances. For each model instance, a versioned pickle is saved, but additionally a `latest`-tagged version is created and loaded by default for convenience of use. This way, you can always revert to a specific version of the model later on, but the default assumes using the latest version. Desired model version can be specified to API with an environment variable.

> NOTE: Because default model versioning depends on git head, commit all changes before running the model workflow, to be able to match model version to source code!

Other option is `mlflow` model store. We do not yet take full advantage of all [mlflow](https://www.mlflow.org/docs/latest/index.html) capabilities, but decided to add the option for further exploration and promising features of the tool. The mlflow model store creates a passive mlflow storage. The user can configure for what is stored in there in addition to the model. You can easily store results, graphs, notebooks, data, metrics. In addition, the mlflow model store can be combined with the mlflow server and UI to compare different model runs in a convenient way. The mlflow store has built in support for most common ml model types, and allows lot of configuration. However, if you are not already familiar with the tool, starting with the pickle store is recommended for simplicity.

Both model stores are by default created in the `local_data/` folder. To ensure persistance of your model store (no matter which option you choose), you should take backups. You can either back-up this folder or volume (according to your configuration) or use [git-svn](https://git-scm.com/docs/git-svn) and optionally [git-lfs](https://git-lfs.com) to version control your model store. You should also include `ml_pipe/` worklflow created notebook copies to the backup.

## Testing

For the ML pipe you can write assertion tests in the ml_pipe notebooks. However, a passing notebook run is already a great starting point for a test. You should separate code and ML tests: code tests should be run with small sample or generated data and intended to ensure that your code works as code, whereas ML tests are to ensure the quality of data and performance of your model. 
Use of automatic acceptance tests, including tests for bias and other ethical aspects is encouraged. Models that do not pass these tests should not be saved to model store - versioned copies of the notebooks and results that are automatically generated are enough to keep track of these experiments.

### Locust load testing for API

API load testing, is recommended. [Locust](https://docs.locust.io/en/stable/) is a recommended tool for this. 

There is an example script in `api/misc/`.

For the example load test, start the api with the example model and run:

    locust -f locustfile.py -H http://127.0.0.1:8000

Then open http://127.0.0.1:8000 in browser to view the results.

Change log_mode in main.py to test how logging method affects to troughput.

Locust settings:  Users 1000 spawn rate: 100


## Logging

Copies of the ml pipe notebooks are saved in `local_data/` automatically when executed through the workflow notebook. These notebooks are labeled with timestamp and experiment setup by default. In addition to versioning of individual pipeline runs, these act as log of individual ML pipe runs. However, if you use MLFlow, more options for experiment tracking are available (see [MLFlow documentation](https://www.mlflow.org/docs/latest/index.html)).

For the API, there are now two different ways to log structured data. Standard logger provides an easy way to transfer structured data to sqlite. From a structured format it is easy to load data back to a dataframe from the log database. Structlog offers a convenient way to persist structured data as json.

## Monitoring

The `compose.yml` file and `monitoring/` folder contain a simple example configuration for monitoring with [Prometheus](https://prometheus.io/docs/introduction/overview/). The container will launch a local Prometheus instance by default. 

Generic functions for monitoring ML models are presented in `api/metrics/` and demonstrated in the api template. Instead of Prometheus histograms, most of the ML metrics are calculated from a fixed-size FIFO queues (see `DriftMonitor`  class in `api/metrics/prometheus_metrics`). This is because the primary function of the ML metrics is to detect drift in data, models and performance, and thus must be comparable throughout the monitoring timeframe. Generic health metrics from [prometheus-client](https://github.com/prometheus/client_python) are also used by default.

Adjust both metrics and monitoring for your needs. 
For a centralized view over multiple algorithms, it is recommended to scrape the local Prometheus instances instead of the API directly. This way you can still view the local time series in case of network issues.

Setting up a specialized tool (e.g. Grafana) is recommended for viewing the metrics in production.

## Security & data protection

> WARNING: Consult cybersecurity and data protection professionals before processing personal data exposing API to external networks.

The template is especially designed for working with personal and sensitive data.

Here is a couple of things that we've considered:

- Ports & network: the template is set up for development. The api and monitoring endpoints are set up for localhost. Check out the configuration before exposing any endpoints to networks.
- Data: Avoid making copies of data. If feasible, load data straight from the source at runtime. If you must copy data locally, store it under `ml_pipe/data/` - this folder is ignored by git. However, it is included in the container volume. Begin development with anonymized or generated data. Utilize [tabular_anonymizer](https://github.com/Datahel/tabular-anonymizer) and  [presidio-text-anonymizer](https://github.com/Datahel/presidio-text-anonymizer).
- Data generated and collected by API is stored under `local_data/` by default. This folder is ignored by git. It is also set up as a `tmpfs` storage in `config.yml` - this means that the contents of the folder only exist in runtime memory of the container and are cleared when the container stops. You may want to change this to a volume or a bind mount - but evaluate the effects on data protection before doing so.
- Protect API endpoints with authentication. Currently the template comes with examples on basic http authentication defined in `api/security/`.
- Use proper security handling for setting and storing passwords, keys, database access tokens etc.
- You can run API and dev as two separate instances of the same image, i.e. run API so that it does not have direct access to training data.
- If you want to share source code publicly, but you process or sensitive data, manually recreate a new, public repository based on your private one. This repository must not come in contact with the real data or secrets to avoid contamination (e.g. accidentaly including sensitive data to a commit) - it is just copy of the code. This is laboursome, but you only have to update this public repo for major updates in the source code - not for every commit.
- Consider if opening your source code will risk data protection or allow malicious or unintended use of the model. Source code cannot always be shared openly.

## Ethical aspects

> NOTE: Data can only show us what is, not what should be.

Include ethical evaluation in your development and testing routines. All data and any models that uses data from, interacts with, or otherwise affects people should be tested for representativeness, bias and possible discrimination. Consider accessibility when improving digital services with ML.


The template comes with an 'ethical issue template' found in `.github/ISSUE_TEMPLATE/ethical.md` which lists the city of Helsinki ethical principles for data and AI - continuously check the compliance of your work. The template also allows reporting ethical issues. However, it is good to have an additional, accessible channels for receiving feedback and concerns from end-users & custormers. It is important to recognize potential ethical issues, even though they could not be verified or solved right away.

## Prequisites

The template was developed and tested with:

    GitHub Codespaces

and MacBook Pro M1 & macOS Ventura 13.2.1 with:

    Docker desktop 4.15.0 (93002)
    VSC 1.74.0
    VSC Dev Containers Extension v0.245.2

and MacBook Pro M1 & macOS Ventura 13.2.1 with:

    podman-desktop 1.1.0
    hyperkit 0.20210107
    podman 4.5.1
    podman-compose 1.0.6
    docker 24.0.2
    docker-compose 2.18.1

Additional configuration may be required for other systems.

## Known issues

 - nbdev_clean git hook may remove 'parameters' tag from notebook cells, even though it should be an allowed key as it is listed in settings.ini. The tag may need to be re-added manually to allow notebook parameterization with papermill.
 - nbdev documentation related functions may not work out-of-box with arm64 machines such as M1 macbooks because the container installs amd64 version of Quarto. You can bypass this by setting `platform` option for docker. However, this makes container build SUPER slow and is thus not a default setting.
 - Source code (git) and model instance version may mismatch if code changes are not committed before updating a model to the model store.
