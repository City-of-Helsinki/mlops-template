#!/bin/bash
## script to update requirements.txt
## prior running, add requirement to one of the .in files
pip-compile --generate-hashes --allow-unsafe -o requirements.txt base_requirements.in full_requirements.in project_requirements.in
pip install -r requirements.txt
