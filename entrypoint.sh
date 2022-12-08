#/bin/bash
MODE=${MODE:dev}
if [[ $MODE = api ]]
then
    cd api
    uvicorn main:app --reload --reload-include *.pickle --host 0.0.0.0
elif [[ $MODE = dev]]
then
    cd dev
elif [[ $MODE = devapi]]
    cd api
    uvicorn main:app --reload --reload-include *.pickle --host 0.0.0.0
    cd ..
else
    echo "unknown mode: "$MODE", use 'api', 'dev' or 'apidev' or leave empty (defaults to 'dev')"