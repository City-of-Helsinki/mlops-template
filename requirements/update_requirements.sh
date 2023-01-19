#!/bin/bash
## script to update requirements.txt
## prior running, manually write desired updates .in files

pip-compile --upgrade --generate-hashes --allow-unsafe --resolver=backtracking -o requirements.txt requirements.in dev-requirements.in api-requirements.in model_store-requirements.in

# to install requirements, run:
# pip install -r requirements.txt
