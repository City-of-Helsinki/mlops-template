#!/bin/bash
## script to update requirements.txt
## prior running, manually write desired updates .in files

# DEV-REQUIREMENTS for developing & updating models
## update dev-requirements.txt
# TODO: cleanup
pip-compile --upgrade --generate-hashes --allow-unsafe --resolver=backtracking -o requirements.txt requirements.in dev-requirements.in api-requirements.in

# to install requirements, run:
# pip install -r requirements.txt
