import os
from utils import read_results
from estimator import TorchExpModel
import torch
import argparse
from pprint import pprint
import numpy as np
from scipy.optimize import minimize

MAIN_PATH = os.getenv("OpenLLM_OUTPUT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
    )
    parser.add_argument(
        "--target_index",
        type=int,
        default=0,
        help="Index of the target to optimize",
    )
    args = parser.parse_args()

    # Load data
    datamix, target = read_results(os.path.join(args.path, "out", "regmix_results.csv"))
    print("Datamix labels:")
    pprint(list(datamix.columns))
    print("\nTarget labels:")
    pprint(list(target.columns))
    input_dim = len(datamix.columns)
    output_dim = len(target.columns)

    # Load model
    model = TorchExpModel(input_dim=input_dim, output_dim=output_dim)
    state_dict = torch.load(
        os.path.join(args.path, "out/exp_fit", "torch_exp_model.pth"),
        map_location="cpu",
    )  # or "cuda"
    model.load_state_dict(state_dict)
    model.eval()

    # Constraint: sum(x) = 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # Bounds: x_i in [0, 1]
    bounds = [(0, 1)] * input_dim

    # Initial guess
    x0 = np.full(input_dim, 1.0 / input_dim)

    def objective(x):
        x_t = torch.tensor(x, dtype=torch.float32)
        out = model.forward(x_t)  # currently returns a vector
        out_scalar = out.sum()  # or out.mean(), or whatever makes sense
        out.shape
        return out_scalar.detach().cpu().numpy().item()

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        constraints=[constraints],
        bounds=bounds,
        options={"disp": True, "maxiter": 10000},
    )

    print("Optimal x:", res.x)
    print("y_pred:", res.fun)


# # WIP - find optimal datamix - optimization under constraint

# import numpy as np
# from scipy.optimize import minimize

# # Dimensions
# m = 3  # input dimension
# d = 4  # output dimension

# # Random model parameters
# np.random.seed(0)
# T = np.random.randn(d, m)
# k = np.abs(np.random.randn(d))  # ensure k > 0
# c = np.random.randn(d)


# # Objective: minimize sum(y_pred)
# def objective(x):
#     y_pred = c + k * np.exp(T @ x)
#     return np.sum(y_pred)


# # Constraint: sum(x) = 1
# constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

# # Bounds: x_i in [0, 1]
# bounds = [(0, 1)] * m

# # Initial guess
# x0 = np.full(m, 1.0 / m)

# # Optimize
# res = minimize(
#     objective,
#     x0,
#     method="SLSQP",
#     constraints=[constraints],
#     bounds=bounds,
#     options={"disp": True},
# )

# print("Optimal x:", res.x)
# print("Sum(y_pred):", res.fun)
