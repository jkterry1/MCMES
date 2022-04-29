#!/bin/bash

# go to directory and get the commit hash
cd ~/rl-baselines3-zoo-flocking
hash=$(git log -n 1 --pretty=format:"%H")

# compress the archive
cd ~/
tar -czvf "${hash}.tar.gz" ./rl-baselines3-zoo-flocking

