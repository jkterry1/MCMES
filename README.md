# Monte Carlo Mature Emergence Search (MCMES)

This repository and its branches contains code used in the work on XXXX

## Basic Idea

The basic idea of MCMES is as follows:

1. Run a hyperparameter sweep for a reinforcement learning algorithm.
2. Select top `n` best sets of hyperparameters.
3. Train `m` policies using the reinforcement learning algorithm for each set of hyperparameters.
4. Observe behaviours that emerge from all `n`x`m` trained policies by rendering out an episode from all the policies.

## Using This Repository

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

When using a custom environment, it is possible to debug these environments using MCMES.
Debugging in this sense is the equivalent of finding ways in which policies can break or cause unintended behaviours.
During training, all policies are saved in `./optimization_policies/`, the behaviours for these mid-training policies can be rendered using

```sh
python3 render_optimization_policies.py
```

All rendered policies are then saved in `./optimization_gifs/`.

## Bugs and Issues

If you encounter any bugs or issues with running the code, please raise an issue on this github repo.
