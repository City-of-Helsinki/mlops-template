# ML Ops template

> Generic template for small scale MLOps.

## About

Create ETL & machine learning pipelines, model store, API, monitoring & logging - all in single container, with minimum setup required! 

![mlops process chart & architecture](mlops_template.png)
*ML Ops template example generic architecture & process chart.*

The generic ML Ops process and structure of the template is described in the figure above. The core structure of the template is built on three elements: a notebook-based ML pipeline (`ml_pipe/`), a model store with two alternatives (`model_store/`), and and API with logging and monitoring properties (`api/`)).

In addition the repository contains `requirements/` folder for managing requirements, `Dockerfile` and `compose.yml` for building and running the container, and `examples/` showing two simple examples on how to create a ML model and save it into a model store. The api template is created so that it works with the examples as-is, but may need to be adjusted for your model.

## Creating a new repo from this template

For your project, create new repo from this template ([see GitHub instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)). Commit history will be cleaned and the authors won't have access to your new repository (although sharing your work is encouraged).

> Note: updates to the template can not be automatically synced to child projects!

Checklist for creating the repository:
1. Use dashes '-' instead of underscores '_' for the repository name. This is a nbdev convention.
2. Set the owner to your organization, if representing one.
3. Once the repository is ready, edit nbdev config file `settings.ini` & `ml_pipe/Makefile` according to your project. The `lib_path` variable should be set to `ml_pipe/your_repository_name_but_with_underscores_instead_of_dashes`.


## Working with the template:

The template assumes working within a Docker container. Non-container use is possible but not recommended.

### Option 1: Codespaces

Launch codespaces on your repository.

Codespaces builds a container according to `.devcontainer/devcontainer.json`.

### Option 2: Local

Clone the repository.

## Running the container

The template container has three modes: two for development `vsc` (default) and `jupyterlab`, and `api` for running the API. The mode is given to Docker as an environment variable `MODE`.

### VSC:

Install VSC and Dev Containers Extension. When you open your repository in VSC, it automatically detects `.devcontainer/devcontianer.json` and suggests opening the repository in a container. Alternatively, press `cmd/ctrl shift P`, type `Remote-Container: Rebuild and Reopen in Container` and press enter. VSC builds the container and attaches to it. 

### JupyterLab:

The template installs JupyterLab within the container. To work in JupyterLab, run `MODE=jupyterlab docker-compose up`. The container starts JupyterLab. To access jupyterlab, copy the address from the terminal to your browser. 

### Working offline with JupyterLab:

The repository allows developing and running ML completely offline. 

0. Clone your repo on a network connected device or use codespaces.
1. Add required python packages to `requirements/requirements.in`.
2. Build the image, and within the `requirements` folder run the script `./update_requirements.sh`. 
3. Rebuild the image.
4. Pull the image and transfer it to the offline device. The offline device must have Docker installed. 
5. Start container in the jupyterlab mode (see above).

The API works offline, too, but may require routing the ports for clients.

### Running the API:

To start the API as the container entrypoint, run `MODE=api docker-compose up`. This loads the latest trained model from model store, starts the API and leaves the container running. The API requires an existing model store and a stored model to function. 

To develop interactively with the API running, you may start the API from within your VSC / jupyterlab terminal by running `uvicorn main:app --reload --reload-include *.pickle --host 0.0.0.0` within the API folder of the container.

To to specify model store and model version to load, use environment variables as specified in `app_base.py`. The default option loads latest model from pickle store.

## Examples

The `examples/` folder contains simplified single-notebook examples on how to create, train, evaluate and deploy a ML model to model store. There are two notebooks due to two alternatives for the model store. You can try out the API by first running an example notebook

0. Runnin an example notebook.
1. Running the API with corresponding model store env. 
2. Prediction API is available: http://127.0.0.1:8000/predict
3. Automatically generated online API documentation is available at: http://127.0.0.1:8000/docs
4. Real time metrics are available at: http://127.0.0.1:8000/metrics and at the built-in example Prometheus server at http://127.0.0.1:9090

To pass git version info to container, run with `GIT_BRANCH="$(git symbolic-ref -q --short HEAD)" GIT_HEAD="$(git rev-parse --short HEAD)" MODE=api docker-compose up`
this adds the version to container labels and passes it on to prometheus.

API predict request sample for the example model:

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
`api` is for API, `dev` is for developer tools, `model_store` for model store and basic `requirements.in` for everything else. In normal use you should only need to edit `requirements.in`. 

To update `requirements.txt`, run `./update_requirements.sh` INSIDE the `requirements/` folder. Do not edit `requirements.txt` manually or with pip freeze.

Install new requirements with `pip install -r requirements/requirements.txt` or by rebuilding the container.

To use the template with a specific version do TODO!

## ML Pipeline

The data processing, model creation and evaluation are to be written in jupyter notebooks, for which there are template notebooks in `ml_pipe/` folder.

- `00_data_etl.ipynb` is for loading and cleaning data or defining functions for doing so. 
- `01_model_class.ipynb` is for defining a ML model class or custom wrappers for existing models. Not needed if you use existing implementations such as sklearn models & simple pipes.
- `02_train_test_validate.ipynb` is for training, testing and validating your ML model. 
- `03_workflow.ipynb` is for the other notebooks automatically. *Ultimately you should be able to load data, train, test, validate and update a model to model store only by running this notebook inside the container.* For complete automation, you can set up this worflow to be triggered by a change in the ML metrics.

To enhance working with notebooks, two tools are included: nbdev and papermill.

Nbdev allows developing clean code straight from notebooks, with no copy-pasting involved.
The template is planned for taking use of the following two nbdev functions:
 - nbdev_clean  - Clean all notebooks in `fname` to avoid merge conflicts
 - nbdev_export - Export notebooks in `path` to Python module (`ml_pipe/your_module`, set the module name in `settings.ini`)

Other features may be useful but are not quaranteed to work out-of-box with the template. See list of commands with `nbdev_help`.

Check out [nbdev tutorial](https://nbdev.fast.ai/tutorials/tutorial.html) for more info on the tool. 

Papermill allows running notebooks from python, parameterized. 


## Testing

For the ML pipe you can write assertion tests in the ml_pipe notebooks. However, a passing notebook run is already a great starting point for a test. Use of automatic acceptance tests before deploying trained models to model store is encouraged.

### Locust load testing for API

Run:

    locust -f locustfile.py -H http://127.0.0.1:8000

Then open http://127.0.0.1:8000 in browser

Change log_mode in main.py to test how logging method affects to troughput.

Locust settings:  Users 1000 spawn rate: 100


## Logging

Copies of the ml pipe notebooks are saved in `local_data/` automatically when executed through the workflow notebook. In addition to versioning of individual pipeline runs, these act as log. However, if you use ML Flow, more options for experiment tracking are available.

For the API, there are now two different ways to log structured data introduced. Standard logger provides easy way to transfer structured data to sqlite. From structured format it is easy to load data back to dataframe from log database. Structlog offers nice way to persist structured data as json.

## Monitoring

The `compose.yml` and `monitoring/` folder contain a simple example configuration for prometheus. The container will launch a local prometheus instance by default. 

Generic functions for monitoring ML algorithms live are presented in `api/metrics/` and demonstrated in the api template. Instead of prometheus histograms, most of the ML metrics are calculated from a fixed-size fifo queues. This is because the primary function of the ML metrics is to detect drift in data, models and performance, and thus must be comparable throughout the monitoring timeframe. Generic health status metrics are also provided.

Adjust both metrics and monitoring for your needs. 
For centralized view over multiple algorithms, it is recommended to scrape the local prometheus instances instead of the API directly. This way you can still view the local time series in case of network issues.

Setting up a specialized tool (e.g. Grafana) is highly recommended for viewing the metrics. For development, the prometheus UI should be enough. 

## Security & data protection

> Warning: Consult a cybersecurity and data protection professionals before touching personal data or exposing anything.

The template is especially designed for working with personal and sensitive data.

Here is a couple of things that we've considered:

- Ports & network: the template is set up for development. The api and monitoring endpoints are set up for localhost. Check out the configuration before exposing any endpoints to networks.
- Data: Avoid making copies of data. If possible, load data straight from the source at runtime. If you must copy data locally, store it under `ml_pipe/data/` - this folder is ignored by git. However, it is included in a volume. Begin development with anonymized or generated data. Utilize [tabular_anonymizer](https://github.com/Datahel/tabular-anonymizer) and  [presidio-text-anonymizer](https://github.com/Datahel/presidio-text-anonymizer).
- Data generated and collected by API is stored under `localdata/` by default. This folder is ignored by git. It is also set up as a `tmpfs` storage in `config.yml` - this means that the contents of the folder only exist in runtime memory of the container and are cleared when the container stops. You may want to change this to a volume or a bind mount - but evaluate the effects on data protection before doing so.
- Protect API endpoints authentication. Currently the template comes with examples on basic http authentication defined in `api/security/`.
- Use a proper security handling for setting and storing passwords, keys, database access tokens etc.
- You can run API and dev as two separate instances of the same image, i.e. run API so that it does not have direct access to training data.
- If you want to share source code publicly, but you utilize personal or sensitive data, manually recreate a new, public repository based on your private one. This repository must not 'touch' the real data or secrets to avoid contamination - it is just copy of the code. This is laboursome, but a way to to avoid accidentally storing and leaking data through git.
- Consider if opening your source code will risk data protection or allow malicious or unintended use of the model. Source code can not always be shared openly.

## Ethical aspects

> NOTE: Data can only show us what is, not what should be.

Include ethical evaluation in your development and testing routines. All data and any models that uses data from, interacts withor otherwise affects people should be tested for representativeness, bias and discrimination. Consider accessibility when improving digital services with ML.


The template comes with an 'ethical issue template' found in `.github/ISSUE_TEMPLATE/ethical.md` which enlists the city of Helsinki ethical principles for data and AI - continuously check out your work for compliance. It is also important to recognize potential ethical issues, even though they could not be verified or solved right away.

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