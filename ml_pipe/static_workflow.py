# execute workflow of the example notebooks
# to run the script, call python static_workflow.py workflow_setup.yaml
# this file has been added to .gitignore
# NOTE: use curly brackets only to format in global variables!
# hint: you can include additional parameters with sys.argv

# import relevant libraries
import papermill as pm
import os
import sys
import yaml

## parse arguments from workflow_setup.yaml
configfilename = sys.argv[1]
with open(configfilename, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)

# variables
notebooks = config["notebooks"]
data = notebooks["data"]
model = notebooks["model"]
loss = notebooks["loss"]

utils = config["utils"]

# update modules before running just to be sure
os.system("nbdev_build_lib")

# run data notebook
_ = pm.execute_notebook(
    data["notebook"],  # input
    utils["save_notebooks_to"]
    + utils["notebook_save_prefix"]
    + data["notebook"],  # output
    parameters=data["params"],  # params
)
# run model notebook
_ = pm.execute_notebook(
    model["notebook"],
    utils["save_notebooks_to"] + utils["notebook_save_prefix"] + model["notebook"],
    parameters=model["params"],
)
# run loss notebook
_ = pm.execute_notebook(
    loss["notebook"],
    utils["save_notebooks_to"] + utils["notebook_save_prefix"] + loss["notebook"],
    parameters=loss["params"],
)

# optional (uncomment): make backup of the index and workflow notebooks:
# os.system('cp {workflow} {save_notebooks_to}{workflow}')
