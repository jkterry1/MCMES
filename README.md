# Monte Carlo Mature Emergence Search (MCMES)

This repository and its branches contains code used in the work on XXXX

## Installation

In addition to the packages in `./requirements.txt`, this branch depends on an installation of [SUMO-RL](https://github.com/LucasAlegre/sumo-rl).

## Bugs and Issues

If you encounter any bugs or issues with running the code, please create an issue on this github repo or email the authors of the paper. Having these code be useful to future researchers is something the authors meaningfully care about.

## Basic Idea

The basic idea of MCMES is as follows:

1. Run a hyperparameter sweep for a reinforcement learning algorithm.
2. Select top `n` best sets of hyperparameters.
3. Train `m` policies using the reinforcement learning algorithm for each set of hyperparameters.
4. Observe behaviours that emerge from all `n`x`m` trained policies by rendering out an episode from all the policies.

## Using This Repository

This code is a lightly modified (and now slightly old) fork of https://github.com/DLR-RM/rl-baselines3-zoo, a simple library that performs hyperparameter tuning for RL learning algorithms found in https://github.com/DLR-RM/stable-baselines3, a popular and easy-to-use learning library. This repository modifies master by hooking in to PettingZoo environments via parameter sharing, using a method described in [this](https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b?gi=551aecde2d6f) tutorial, and adds a few small additional utilities for performing MCMES.

Alternate PettingZoo environments could be easily dropped into one of the branches, or the changes to our code could be fairly easily ported to the newest version of RL-Baselines3-Zoo without too much work should researchers wish to apply these methods to new environments.

### Run hyperparameter sweep

To perform an MCMES run on a distributed system, provide the direction to an SQL server directory via the `--storage` argument. An example would be:

```sh
python3 train.py --algo ppo -n 2000000 --optimization-log-path optimization_policies -optimize --study-name STUDY_NAME --storage mysql://root:dummy@99.999.9.99/study_name
```

Where:
- `-n 2000000` specifies the number of timesteps for training in the environment.
- `--optimization-log-path optimization_policies` saves all trained policies to the folder `./optimization_policies`.
- `--study-name STUDY_NAME` specifies the name of the results pool that the optimizer will use when sampling hyperparameters.

If running locally, it suffices to drop the `--storage` argument, in this case, all runs will be saved locally.

The experiments conducted in the paper utilize a small number of additional parameters for logging and noise sampling.
These parameters can be viewed in `./run_optimize.sh`.
In our script, we:

- use `CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=1` to select GPU N on the system, and dedicate 1 CPU thread per run.
- use `nohup` to catch all console output and save it to `./optimize_logs/optimize_x.out`.
- use `--sampler tpe` to use the TPE sampler to sample hyperparameters.
- use `--pruner median` to use the Median pruner to hedge against worse performing hyperparameters. This compares the result of each run with the median of all currently completed runs.

### Selecting Best Hyperparameters

Once a sufficient number of runs have been conducted, the top `N` best hyperparameters according to Optuna can be found using

```sh
python3 best_hyperparameters.py --study-name STUDY_NAME --storage mysql://root:dummy@99.999.9.99/$1 --save-n-best-hyperparameters N
```

The script used for the experiments can be viewed in `run_best_hyperparameters.sh`.
In our script, we:

- use `--print-n-best-trials 100` to print to console the evaluation results for the top 100 policies.

### Pruning Hyperparameters According to Performance

For all the policies obtained from the optimization runs, we define a "mature policy" as a policy wherein the evaluation score in the environment is more than a certain threshold, the value for this threshold is defined on line 90 of `./eval_hyperparameters.py`.
The script to save the policies into a folder `./mature_policies/` is

```sh
python3 eval_hyperparameters.py 0
```

where the number at the end corresponds to the top `n` best hyperparameters found using Optuna.
The exact script used for the experiments can be viewed in `run_evalXX.sh`, we utilize slightly different scripts for finding mature policies amongst the best hyperparameters by distributing the computation across multiple machines.
The arguments used here are similar to those defined [previously](#run-hyperparameter-sweep).

### Rendering Policies

Once all mature policies have been found, the indiviaul policy behaviours can be rendered out using

```sh
python3 render.py 0
```

where the number at the end corresponds to the top `n` best hyperparameters found using Optuna.
The exact script used for the experiments can be viewed in `run_renderX.sh`.
Again, we utilize slightly different scripts for rendering all mature policies for different hyperparameter sets across multiple machines.
The arguments used here are similar to those defined [previously](#run-hyperparameter-sweep).

### Debugging Environments

When applying MCMES to a new environment, it's often useful to see what initial behaviors were found during hyperparameter training to ensure no bugs were discovered in the environments (this is quite common in our experience), before completing the full MCMES search. To this end, during tuning, all the best policies for each run are saved in `./optimization_policies/`, and all saved policies can have gifs/videos generated using:

```sh
python3 render_optimization_policies.py
```

All rendered policies are then saved in `./optimization_gifs/`.
