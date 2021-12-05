mkdir -p ./optimize_logs
rm -rf optimize_logs/*
mkdir -p ./optimization_policies
rm -rf optimization_policies/*

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python3 train.py --algo dqn --env LunarLander-v2 -n 2000000 -optimize --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://root:dummy@10.128.0.28/$1 &> ./optimize_logs/optimize_0.out &
