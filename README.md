# Crowd simulation experiments

This branch consists of two parts: the environment (CrowdAI) and the learning code (coltra-rl).

The environment is a copy of https://github.com/RedTachyon/CrowdAI at the v5 release. To use it, simply open the project using a compatible version of Unity (at least 2021.x) and build it as an executable.

The learning code is a copy of https://github.com/RedTachyon/coltra-rl at the release named MCMES. The code used to run the MCMES experiments is in `coltra-rl/scripts/optimize/`, where `optuna_setup.py` creates the optuna database, `optuna_crowd.py` runs the optimization procedure, and `retrain_crowd.py` can be used to retrain the best-performing agents.
