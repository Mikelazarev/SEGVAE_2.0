import numpy as np
from experiments import run_experiment_bio_1

if __name__ == '__main__':

    X = np.linspace(-2, 2, 100)[..., None]
    y_true = X + X * X
    run_experiment_bio_1(
        X, y_true,
        wandb_proj='RoboScientist',
        project_name='test_run'
    )
