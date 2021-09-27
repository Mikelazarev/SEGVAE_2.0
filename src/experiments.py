import roboscientist.solver.vae_solver as rs_vae_solver
import roboscientist.equation.equation as rs_equation
import roboscientist.logger.wandb_logger as rs_logger
import equation_generator as rs_equation_generator
import torch

import os
import time
import warnings


def run_experiment(
        X,
        y_true,
        functions=None,  # list, subset of ['sin', 'add', 'safe_log', 'safe_sqrt', 'cos', 'mul', 'sub']
        arities=None,
        free_variables=None,  # ['x1']
        wandb_proj='some_experiments',
        project_name='COLAB',
        constants=None,  # None or ['const']
        const_opt_method='bfgs',
        float_constants=None,
        epochs=100,
        train_size=20000,
        test_size=10000,
        n_formulas_to_sample=2000,
        formula_predicate=None
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
    rs_equation_generator.generate_pretrain_dataset(train_size, 14, train_file, functions=functions, arities=arities,
                                                    all_tokens=functions + free_variables + constants + pretrain_float_token,
                                                    formula_predicate=formula_predicate)
    rs_equation_generator.generate_pretrain_dataset(test_size, 14, val_file, functions=functions, arities=arities,
                                                    all_tokens=functions + free_variables + constants + pretrain_float_token,
                                                    formula_predicate=formula_predicate)

    vae_solver_params = rs_vae_solver.VAESolverParams(
        device=torch.device('cuda'),
        true_formula=None,
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
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': len(free_variables)},
        is_condition=False,
        sample_from_logits=True,
        n_formulas_to_sample=n_formulas_to_sample,
        pretrain_train_file=train_file,
        pretrain_val_file=val_file,
        const_opt_method=const_opt_method,
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
    vs.create_checkpoint(os.path.join(log_dir, 'checkpoint_1'))
    vs.solve((X, y_true), epochs=epochs)
    return vs
