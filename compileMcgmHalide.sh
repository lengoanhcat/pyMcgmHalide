#!/bin/bash
ulimit -s unlimited
export HL_DEBUG_CODEGEN=0
python3 mcgmOpticalFlow_generate.py
