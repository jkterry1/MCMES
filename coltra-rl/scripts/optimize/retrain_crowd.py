import optuna
from typarse import BaseParser

from optuna_crowd import objective

class Parser(BaseParser):
    env: str = "../builds/crowd-v5/crowd.x86_64"
    worker_id: int = 0
    n_trials: int = 10
    optuna_name: str = "egocentric"
    indices: list[str] = []

    _abbrev = {"env": "e", "worker_id": "w", "n_trials": "n", "optuna_name": "o", "indices": "i"}

    _help = {
        "env": "Path to the environment",
        "worker_id": "Worker ID to start from",
        "n_trials": "Number of trials",
        "optuna_name": "Name of the optuna study",
        "indices": "List of indices to train on",
    }


if __name__ == '__main__':
    args = Parser()

    print(args.indices)

    study = optuna.load_study(storage=f"sqlite:///{args.optuna_name}.db", study_name=args.optuna_name)
    trials = study.trials

    for idx in args.indices:
        trial = trials[int(idx)]
        print(f"Trial {idx}")
        for i in range(args.n_trials):
            print(f"Run {i}")
            objective(trial, i, args.worker_id, args.env)
