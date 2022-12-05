#!/bin/bash
## script to update requirements.txt
## prior running, manually write desired updates .in files

# DEV-REQUIREMENTS for developing & updating models
## update dev-requirements.txt
pip-compile --upgrade --generate-hashes --allow-unsafe --resolver=backtracking -o dev-requirements.txt requirements.in dev-requirements.in 

# API_REQUIREMENTS for running the model & api in production
pip-compile --upgrade --generate-hashes --allow-unsafe --resolver=backtracking -o api-requirements.txt requirements.in api-requirements.in

# to install requirements, run:
# pip install -r dev-requirements.txt
# or 
# pip install -r api-requirements.txt
# after this script
