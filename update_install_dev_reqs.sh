#!/bin/bash
## script to update requirements.txt
## prior running, manually write desired updates .in files

## update dev-requirements.txt
pip-compile --generate-hashes --allow-unsafe -o dev-requirements.txt base-requirements.in dev-requirements.in requirements.in

## update requirements.txt
pip-compile --generate-hashes --allow-unsafe -o requirements.txt base-requirements.in requirements.in

## install updated requirements
pip-sync dev-requirements.txt requirements.txt