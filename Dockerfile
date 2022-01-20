FROM ubuntu:20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \ 
apt-get -y upgrade && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential software-properties-common python-is-python3 python3-pip python3-dev git && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN mkdir /app

COPY . /app

RUN cd /app

RUN python -m pip install -U pip

RUN python -m pip install pip-tools

# update and install dev requirements
RUN ./update_install_dev_reqs.sh

# for running the workflow
RUN python -m ipykernel install --user --name python38myenv

# Uncomment to run jupyterlab when container is launched
# (you can change jupyterlab settings in .devcontainer/jupyter-server-config.py):
# RUN jupyter-lab --allow-root --config .devcontainer/jupyter-server-config.py 