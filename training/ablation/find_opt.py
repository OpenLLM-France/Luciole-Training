# WIP - find optimal datamix - optimization under constraint

import numpy as np
from scipy.optimize import minimize

# Dimensions
m = 3  # input dimension
d = 4  # output dimension

# Random model parameters
np.random.seed(0)
T = np.random.randn(d, m)
k = np.abs(np.random.randn(d))  # ensure k > 0
c = np.random.randn(d)


# Objective: minimize sum(y_pred)
def objective(x):
    y_pred = c + k * np.exp(T @ x)
    return np.sum(y_pred)


# Constraint: sum(x) = 1
constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

# Bounds: x_i in [0, 1]
bounds = [(0, 1)] * m

# Initial guess
x0 = np.full(m, 1.0 / m)

# Optimize
res = minimize(
    objective,
    x0,
    method="SLSQP",
    constraints=[constraints],
    bounds=bounds,
    options={"disp": True},
)

print("Optimal x:", res.x)
print("Sum(y_pred):", res.fun)
