source venv/bin/activate
mkdir -p ./optimize_logs
rm -rf optimize_logs/*

mkdir -p ./optimization_policies
rm -rf optimization_policies/*

num_gpu=$(($(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) - 1))

for i in $(seq 0 $num_gpu 1); do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 54000000 -optimize --optimization-log-path optimization_policies --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://database:ZrWdchqeNpmbuAXYpy2V@35.194.57.226/kaz7 --verbose 2 &> ./optimize_logs/optimize_0.out &
  sleep 3
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=1 nohup python3 train.py --algo ppo --env LunarLanderContinuous-v2 -n 54000000 -optimize --optimization-log-path optimization_policies --n-trials 150 --sampler tpe --pruner median --study-name $1 --storage mysql://database:ZrWdchqeNpmbuAXYpy2V@35.194.57.226/kaz7 --verbose 2 &> ./optimize_logs/optimize_0.out &
  sleep 3
done
