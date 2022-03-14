#!/bin/sh

mkdir -p ./optimize_logs
rm -rf optimize_logs/*
mkdir -p ./optimization_policies
rm -rf optimization_policies/*

nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 2000000 --optimization-log-path optimization_policies -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@35.194.57.226/$1 --verbose 2 &> ./optimize_logs/optimize_0.out
