#!/bin/bash

# cleanup
rm -rf results/*
mkdir -p ./results
rm -rf render_optimization_logs/*
mkdir -p ./render_optimization_logs

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 nohup python3 render_optimization_policies.py $1
