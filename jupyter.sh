#!/bin/bash

ml purge
ml Python/3.5.2-goolf-1.7.20
source venv-testenv/bin/activate
jupyter lab --no-browser --port=1000 --ip=0.0.0.0
