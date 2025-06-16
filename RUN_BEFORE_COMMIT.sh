#!/bin/bash

# Make this file an executable, run command:
# chmod +x RUN_BEFORE_COMMIT.sh

echo "Running 'pycln .' formatter to remove unused imports"
pycln .
echo "Running 'isort .' formatter to sort imports"
isort .
echo "Running 'black .' code formatter on all files"
black .

echo "Running 'pytest' to test the code"
result=$(pytest tests/)
echo ${result}

if [[ ${result} == *"failed"* ]]; then
    echo "Tests failed, please fix them before commit"
    exit 1
else 
    echo "Tests passed, you can commit the code"
    exit 0
fi
