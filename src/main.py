import os
import pandas as pd
import argparse
from experiments import run_experiment
import roboscientist.equation.equation as rs_equation
import roboscientist.equation.operators as rs_operators


def get_offsprings(list_of_tokens, idx):
    if list_of_tokens[idx] in rs_operators.OPERATORS:
        open_nodes = rs_operators.OPERATORS[list_of_tokens[idx]].arity
    else:
        open_nodes = 0
    traversal = []
    for i, token in enumerate(list_of_tokens[idx + 1:]):
        if open_nodes == 0:
            break
        traversal.append(token)
        if token in rs_operators.OPERATORS:
            operator = rs_operators.OPERATORS[token]
            open_nodes += operator.arity - 1
        else:
            open_nodes -= 1

    return traversal


def predicate(list_of_tokens):
    for i, token in enumerate(list_of_tokens):
        offsprings = get_offsprings(list_of_tokens, i)
        if token == 'sin' or token == 'cos':
            if 'sin' in offsprings or 'cos' in offsprings:
                return False
    return True


NGUYEN = {
    '1': (['add', 'x1', 'add', 'mul', 'x1', 'x1', 'mul', 'mul', 'x1', 'x1', 'x1'], ['x1']),
    '2': (['mul', 'add', 'x1', '1.0', 'mul', 'x1', 'add', '1.0', 'mul', 'x1', 'x1'], ['x1']),
    '3': (['add', 'x1', 'mul', 'add', 'x1', 'mul', 'x1', 'x1', 'add', 'x1', 'mul', 'mul', 'x1', 'x1', 'x1'], ['x1']),
    '4': (['add', 'x1', 'add', 'mul', 'x1', 'x1', 'add', 'safe_pow', 'x1', '3.0',
          'add', 'safe_pow', 'x1', '4.0', 'add', 'safe_pow', 'x1', '5.0', 'safe_pow', 'x1', '6.0'], ['x1']),
    '5': (['sub', 'mul', 'sin', 'mul', 'x1', 'x1', 'cos', 'x1', '1.0'], ['x1']),
    '6': (['add', 'sin', 'x1', 'sin', 'add', 'x1', 'mul', 'x1', 'x1'], ['x1']),
    '7': (['add', 'safe_log', 'add', 'x1', '1.0', 'safe_log', 'add', 'mul', 'x1', 'x1', '1.0'], ['x1']),
    '8': (['safe_pow', 'x1', 'safe_div', '1.0', '2.0'], ['x1']),
    '9': (['add', 'sin', 'x1', 'sin', 'mul', 'x2', 'x2'], ['x1', 'x2']),
    '10': (['mul', '2.0', 'mul', 'sin', 'x1', 'cos', 'x2'], ['x1', 'x2']),
    '11': (['safe_pow', 'x1', 'x2'], ['x1', 'x2']),
    '12': (['add', 'sub', 'safe_pow', 'x1', '4.0', 'safe_pow', 'x1', '3.0',
           'sub', 'safe_div', 'mul', 'x2', 'x2', '2.0', 'x2'], ['x1', 'x2'])
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument("--eq", type=str)
    parser.add_argument("--noise", type=str)
    parser.add_argument("--latent", type=int)
    parser.add_argument("--hidden", type=int)
    args = parser.parse_args()

    filename = f'Nguyen-{args.eq}_n0.{args.noise:0>2}_d10.csv'
    true_func, free_variables = NGUYEN[args.eq]
    df = pd.read_csv(os.path.join(args.path, filename), sep=',', header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(X.shape)
    run_experiment(
        X.values, y.values,
        functions=['add', 'sub', 'mul', 'safe_div', 'sin', 'cos', 'safe_log', 'safe_pow'],
        free_variables=free_variables,
        wandb_proj='latent128',
        project_name=f'{filename[:-7]}_wo_const_latent_{args.latent}',
        constants=[],
        float_constants=None, # rs_operators.FLOAT_CONST,
        epochs=400,
        n_formulas_to_sample=5000,
        max_formula_length=25,
        formula_predicate=predicate,
        true_formula=rs_equation.Equation(true_func),
        latent=args.latent,
        lstm_hidden_dim=args.hidden,
        device='cuda'
    )

