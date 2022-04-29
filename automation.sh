#!/bin/bash

if [ "$hostname" = "dream" ]
  echo 'nothing to do'
  exit
else if [ "$hostname" = "prophet" ]
  echo 'nothing to do'
  exit
else if [ "$hostname" = "flocking1" ]
  echo 'nothing to do'
  exit
end

# run optimize
ssh dream 'tmux send-keys -t 0 "./run_optimize.sh" ENTER'
ssh prophet 'tmux send-keys -t 0 "./run_optimize.sh" ENTER'
ssh gcp-flocking-1 'tmux send-keys -t 0 "./run_optimize.sh" ENTER'

# generate logs
ssh dream 'tmux send-keys -t 0 "./run_render_optimization.sh" ENTER'
ssh prophet 'tmux send-keys -t 0 "./run_render_optimization.sh" ENTER'
ssh gcp-flocking-1 'tmux send-keys -t 0 "./run_render_optimization.sh" ENTER'

# backup
ssh dream 'tmux send-keys -t 0 "./backup.sh" ENTER'
ssh prophet 'tmux send-keys -t 0 "./backup.sh" ENTER'
ssh gcp-flocking-1 'tmux send-keys -t 0 "./backup.sh" ENTER'
