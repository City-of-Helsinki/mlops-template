FROM ubuntu:20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \ 
apt-get -y upgrade && \
apt-get install -y --no-install-recommends build-essential software-properties-common python-is-python3 python3-pip python3-dev && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN mkdir /app

COPY . /app

RUN python -m pip install -U pip

RUN pip install -r /app/project_requirements.txt
