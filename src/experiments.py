import wandb

import roboscientist.solver.vae_solver as rs_vae_solver
import roboscientist.equation.equation as rs_equation
import roboscientist.solver.vae_solver_lib.optimize_constants as rs_optimize_constants
import roboscientist.logger.wandb_logger as rs_logger
import equation_generator as rs_equation_generator
import torch

import os
import time
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error


def run_experiment(
        X,
        y_true,
        true_formula=None,
        functions=None,  # list, subset of ['sin', 'add', 'safe_log', 'safe_sqrt', 'cos', 'mul', 'sub']
        arities=None,
        free_variables=None,  # ['x1']
        wandb_proj='some_experiments',
        project_name='COLAB',
        constants=None,  # None or ['const']
        const_opt_method='bfgs',
        float_constants=None,
        epochs=100,
        train_size=40000,
        test_size=10000,
        n_formulas_to_sample=2000,
        max_formula_length=15,
        formula_predicate=None,
        device=torch.device('cuda'),
        latent=8,
        lstm_hidden_dim=128
):
    if functions is None:
        functions = ['sin', 'add', 'cos', 'mul']
    if arities is None:
        arities = {'cos': 1, 'sin': 1, 'add': 2, 'mul': 2,  'div': 2, 'sub': 2, 'pow': 2, 'safe_log': 1,
                   'safe_sqrt': 1, 'safe_exp': 1, 'safe_div': 2, 'safe_pow': 2}
    if free_variables is None:
        free_variables = ['x1']
    if constants is None:
        constants = []
    if float_constants is None:
        float_constants = []
        pretrain_float_token = []
    else:
        pretrain_float_token = ['float']
    if formula_predicate is None:
        formula_predicate = lambda func: True
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_dir = os.path.join(root_dir, 'logs/')
    os.makedirs(log_dir, exist_ok=True)
    train_file = os.path.join(log_dir, f'train_{str(time.time())}')
    val_file = os.path.join(log_dir, f'val_{str(time.time())}')
    rs_equation_generator.generate_pretrain_dataset(train_size, max_formula_length, train_file,
                                                    functions=functions, arities=arities,
                                                    all_tokens=functions + free_variables + constants + pretrain_float_token,
                                                    formula_predicate=formula_predicate)
    rs_equation_generator.generate_pretrain_dataset(test_size, max_formula_length, val_file,
                                                    functions=functions, arities=arities,
                                                    all_tokens=functions + free_variables + constants + pretrain_float_token,
                                                    formula_predicate=formula_predicate)

    vae_solver_params = rs_vae_solver.VAESolverParams(
        device=device,
        true_formula=true_formula,
        optimizable_constants=constants,
        float_constants=float_constants,
        formula_predicate=formula_predicate,
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        retrain_file=os.path.join(log_dir, f'retrain_1_{str(time.time())}'),
        file_to_sample=os.path.join(log_dir, f'sample_1_{str(time.time())}'),
        functions=functions,
        arities={'sin': 1, 'add': 2, 'sub': 2, 'safe_log': 1, 'cos': 1, 'mul': 2,
                 'safe_sqrt': 1, 'safe_exp': 1, 'safe_div': 2, 'safe_pow': 2},
        free_variables=free_variables,
        model_params={'token_embedding_dim': 128, 'hidden_dim': lstm_hidden_dim,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': latent, 'x_dim': len(free_variables)},
        is_condition=False,
        sample_from_logits=True,
        n_formulas_to_sample=n_formulas_to_sample,
        pretrain_train_file=train_file,
        pretrain_val_file=val_file,
        const_opt_method=const_opt_method,
        max_formula_length=max_formula_length,
    )

    if os.path.isfile(os.path.join(root_dir, 'wandb_key')):
        with open(os.path.join(root_dir, 'wandb_key')) as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
            os.environ["WANDB_DIR"] = '../logs'
        logger_init_conf = {key: str(item) for key, item in vae_solver_params._asdict().items()}
        logger = rs_logger.WandbLogger(wandb_proj,  project_name, logger_init_conf, mode='online')
    else:
        logger = None
        warnings.warn('Logging disabled! Please provide wandb_key at {}'.format(root_dir))
    vs = rs_vae_solver.VAESolver(logger, None, vae_solver_params)
    #vs = rs_vae_solver.VAESolver(logger, os.path.join(log_dir, f'checkpoint_no_const_9-12'), vae_solver_params)
    #vs.create_checkpoint(os.path.join(log_dir, f'checkpoint_no_const_latent_{latent}'))
    vs.solve((X, y_true), epochs=epochs)

    def final_log(top_k, mses, formulas):
        used = set()
        unique_pairs = [x for x in zip(mses, formulas) if x[1] not in used and (used.add(x[1]) or True)]
        mses, formulas = list(zip(*unique_pairs))

        top_indices = np.argsort(mses)[:top_k]
        top_formulas = np.array(formulas)[top_indices]
        top_mses = np.array(mses)[top_indices]
        data = []

        const = rs_optimize_constants.optimize_constants(true_formula, X, y_true, const_opt_method)
        yt = true_formula.func(X, const)
        if type(yt) is float or yt.shape == (1,) or yt.shape == (1, 1) or yt.shape == ():
            yt = np.repeat(np.array(yt).astype(np.float64),
                           X.reshape(-1, vs.params.model_params['x_dim']).shape[0]).reshape(-1, 1)

        for i, (f, m) in enumerate(zip(top_formulas, top_mses)):
            f = rs_equation.Equation(f.split())
            const = rs_optimize_constants.optimize_constants(f, X, y_true, const_opt_method)
            y_pred = f.func(X, const)
            if type(y_pred) is float or y_pred.shape == (1,) or y_pred.shape == (1, 1) or y_pred.shape == ():
                y_pred = np.repeat(np.array(y_pred).astype(np.float64),
                                   X.reshape(-1, vs.params.model_params['x_dim']).shape[0]).reshape(-1, 1)

            tm = mean_squared_error(y_pred, yt)
            success = True if tm < 1e-9 else False
            data.append([i + 1, f.repr(constants), m, tm, success])
        return data

    def final_log2(pareto_best_formulas):
        result = False
        complexity_best, pareto_best = [], []
        error_to_beat = 1e9
        pareto_best_formulas = dict(sorted(pareto_best_formulas.items()))

        const = rs_optimize_constants.optimize_constants(true_formula, X, y_true, const_opt_method)
        yt = true_formula.func(X, const)
        if type(yt) is float or yt.shape == (1,) or yt.shape == (1, 1) or yt.shape == ():
            yt = np.repeat(np.array(yt).astype(np.float64),
                           X.reshape(-1, vs.params.model_params['x_dim']).shape[0]).reshape(-1, 1)
        for complexity, (formula, error) in pareto_best_formulas.items():
            f = rs_equation.Equation(formula.split())
            const = rs_optimize_constants.optimize_constants(f, X, y_true, const_opt_method)
            y_pred = f.func(X, const)
            if type(y_pred) is float or y_pred.shape == (1,) or y_pred.shape == (1, 1) or y_pred.shape == ():
                y_pred = np.repeat(np.array(y_pred).astype(np.float64),
                                   X.reshape(-1, vs.params.model_params['x_dim']).shape[0]).reshape(-1, 1)

            tm = mean_squared_error(y_pred, yt)
            success = True if tm < 1e-9 else False
            complexity_best.append([complexity, f.repr(constants), error, tm, success])
            if error < error_to_beat:
                error_to_beat = error
                if success: result = True
                pareto_best.append([complexity, f.repr(constants), error, tm, success])

        return complexity_best, pareto_best, result
    complexity_best, pareto_best, result = final_log2(vs.stats.all_best_per_complexity)
    if logger is not None:
        wandb.log({
            'success': result,
            #'all_time_best': wandb.Table(data=final_log(10, vs.stats.all_best_mses, vs.stats.all_best_formulas),
            #                             columns=['rank', 'formula', 'mse', 'true_mse', 'success']),
            #'last_step_best': wandb.Table(data=final_log(10, vs.stats.last_n_best_mses, vs.stats.last_n_best_formulas),
            #                              columns=['rank', 'formula', 'mse', 'true_mse', 'success']),
            'complexity_best': wandb.Table(data=complexity_best,
                                           columns=['complexity', 'formula', 'mse', 'true_mse', 'success']),
            'pareto_best': wandb.Table(data=pareto_best,
                                       columns=['complexity', 'formula', 'mse', 'true_mse', 'success'])
        })

    return vs
