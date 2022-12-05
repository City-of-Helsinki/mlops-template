FROM ubuntu:22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \ 
apt-get -y upgrade && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sudo curl build-essential software-properties-common python-is-python3 python3-pip python3-dev git && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN mkdir /app

ADD ./api /app/ 

ADD ./requirements/api-requirements.txt /app/

# to add custom ml module, wrapper or util to api:
# ADD ./dev/module_path /app/module_path

WORKDIR /app

RUN python -m pip install -r api-requirements.txt

# version info (tag & pass)
ARG GIT_BRANCH=unspecified
LABEL git_branch=$GIT_BRANCH
ENV GIT_BRANCH $GIT_BRANCH
ARG GIT_HEAD=unspecified
LABEL git_head=$GIT_HEAD
ENV GIT_HEAD $GIT_HEAD