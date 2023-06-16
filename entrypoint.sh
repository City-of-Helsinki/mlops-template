#/bin/bash
MODE=${MODE:-vsc}
if [[ $MODE = api ]]
then
    # start api
    cd api
    uvicorn main:app --reload --reload-include *.pickle --host 0.0.0.0

elif [[ $MODE = vsc ]]
then
    # leave container running. default for working with vsc and codespaces.
    tail -f /dev/null

elif [[ $MODE = jupyterlab ]]
then
    # work in jupyterlab
    jupyter-lab --allow-root --ip 0.0.0.0 --port 8888
    
else
    echo "unknown mode: "$MODE", use 'api', 'vsc', 'jupyterlab' or leave empty (defaults to 'vsc')"
fi