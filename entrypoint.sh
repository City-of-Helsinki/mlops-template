#/bin/bash
MODE=${MODE:-dev}
if [[ $MODE = api ]]
then
    cd api
    uvicorn main:app --reload --reload-include *.pickle --host 0.0.0.0

elif [[ $MODE = dev ]]
then
    tail -f /dev/null
    
else
    echo "unknown mode: "$MODE", use 'api', 'dev' or leave empty (defaults to 'dev')"
fi