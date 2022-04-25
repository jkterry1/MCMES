#!/bin/bash

# cleanup
rm -rf results/*
mkdir -p ./results
rm -rf render_optimization_logs/*
mkdir -m ./render_optimization_logs

# declare arrays
declare -a gpu
declare -a pid
declare -a free_gpu

# get number of gpus on the system
num_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# all available gpus are stored here
free_gpu=($(seq 0 1 $((num_gpu-1))))

# get all trials
all_dirs=./optimization_policies/trial*/

for path in ./optimization_policies/trial*/; do
  # get only the end path string
  dirname=$(basename "$path")
  # start job with first available gpu
  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${free_gpu[0]} nohup python3 render_optimization_policies.py $dirname &> "./render_optimization_logs/${dirname}.out" &
  # store the pid-gpu pair
  pid+=($!)
  gpu+=(${free_gpu[0]})

  # remove the gpu from available gpus
  unset free_gpu[0]
  free_gpu=(${free_gpu[@]})

  # while the list of pid is equal to num_gpus
  while [ ${#pid[@]} -ge $num_gpu ]; do
    echo ${pid[@]} ${gpu[@]}
    # for all pids in the pid list
    for i in $(seq 0 $((${#pid[@]}-1)) 1); do
      echo $i ${pid[$i]} ${gpu[$i]}
      # check if the process still exists
      if ! ps -p ${pid[$i]} > /dev/null
        then
          # if it doesn't exist anymore just remove it
          # and free the gpu
          free_gpu+=(${gpu[$i]})
          unset gpu[$i]
          unset pid[$i]
          gpu=(${gpu[@]})
          pid=(${pid[@]})
          break 2
      fi
    done
    sleep 1
  done
done

