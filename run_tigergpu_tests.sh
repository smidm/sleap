#!/bin/bash

module load anaconda
module load cudnn/cuda-9.0/7.0.3

conda activate sleap

python -m pytest --ignore="tests/gui" tests/