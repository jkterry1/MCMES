# Monte Carlo Mature Emergence Search (MCMES)

This repository and its branches contains code used in the work on XXXX

## Using This Repository

### Basic Optimization Run

To perform an MCMES run on a distributed system, provide the direction to an SQL server directory via the `--storage` argument. An example would be:

```sh
python3 train.py --algo ppo -n 2000000 --optimization-log-path optimization_policies -optimize --study-name STUDY_NAME --storage mysql://root:dummy@99.999.9.99/study_name
```

The experiments conducted in the paper utilize a small number of additional parameters for logging and noise sampling.
These parameters can be viewed in `./run_optimize.sh`.

### Selecting Best Hyperparameters

Once a sufficient number of runs have been conducted, the top `N` best hyperparameters according to Optuna can be found using

```sh
python3 best_hyperparameters.py --study-name STUDY_NAME --storage mysql://root:dummy@99.999.9.99/$1 --save-n-best-hyperparameters N
```

The script used for the experiments can be viewed in `run_best_hyperparameters.sh`

### Pruning Hyperparameters According to Performance

For all the policies obtained from the optimization runs, we define a "mature policy" as a policy wherein the evaluation score in the environment is more than a certain threshold.
The script to save the policies into a folder `./mature_policies/` is

```sh
python3 eval_hyperparameters.py 0
```

where the number at the end corresponds to the top `n` best hyperparameters found using Optuna.
The exact script used for the experiments can be viewed in `run_evalXX.sh`, we utilize slightly different scripts for finding mature policies amongst the best hyperparameters by distributing the computation across multiple machines.

### Rendering Policies

Once all mature policies have been found, the indiviaul policy behaviours can be rendered out using

```sh
python3 render.py 0
```

where the number at the end corresponds to the top `n` best hyperparameters found using Optuna.
The exact script used for the experiments can be viewed in `run_renderX.sh`.
Again, we utilize slightly different scripts for rendering all mature policies for different hyperparameter sets across multiple machines.

### Debugging Environments

When using a custom environment, it is possible to debug these environments using MCMES.
During training, all policies are saved in `./optimization_policies/`, the behaviours for these mid-training policies can be rendered using

```sh
python3 render_optimization_policies.py
```

All rendered policies are then saved in `./optimization_gifs/`.
