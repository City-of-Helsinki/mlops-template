#!/bin/bash
## script to update requirements.txt
## prior running, manually write desired updates .in files

## update requirements.txt
pip-compile --generate-hashes --allow-unsafe -o requirements.txt base_requirements.in full_requirements.in project_requirements.in

## update project_requirements.txt
pip-compile --generate-hashes --allow-unsafe -o project_requirements.txt base_requirements.in project_requirements.in

## install updated requirements
pip install -r requirements.txt