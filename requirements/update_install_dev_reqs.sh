#!/bin/bash
## script to update requirements.txt
## prior running, manually write desired updates .in files

## update requirements.txt
pip-compile --upgrade --generate-hashes --allow-unsafe -o requirements.txt base-requirements.in requirements.in project-requirements.in

## update min-requirements.txt
pip-compile --upgrade --generate-hashes --allow-unsafe -o min-requirements.txt base-requirements.in project-requirements.in

## install updated requirements
# pip-sync requirements.txt # pip-sync causes errors with 'distutil installed projects'
# use regular pip instead (not as clean repo but should work)
pip install -r requirements.txt
