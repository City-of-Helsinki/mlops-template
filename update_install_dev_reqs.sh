#!/bin/bash
## script to update requirements.txt
## prior running, manually write desired updates .in files

## update requirements.txt
pip-compile --generate-hashes --allow-unsafe -o requirements.txt base-requirements.in requirements.in project-requirements.in

## update min-requirements.txt
pip-compile --generate-hashes --allow-unsafe -o min-requirements.txt base-requirements.in project-requirements.in

## install updated requirements
pip-sync requirements.txt