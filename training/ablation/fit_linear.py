import os
import argparse
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from utils import read_results

MAIN_PATH = os.getenv("OpenLLM_OUTPUT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
    )
    args = parser.parse_args()
    os.makedirs(os.path.join(args.path, "out/linear_fit"), exist_ok=True)

    datamix, target = read_results(os.path.join(args.path, "out", "regmix_results.csv"))

    datamix_labels = datamix.columns
    print(f"Datamix labels: {datamix_labels}")

    target_labels = target.columns
    print(f"Target labels: {target_labels}")

    x = torch.tensor(datamix.to_numpy(), dtype=torch.float32)
    y = torch.tensor(np.log(target.to_numpy()), dtype=torch.float32)

    N, input_dim = x.shape
    N, output_dim = y.shape

    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)

    # 6. Evaluate
    mse = mean_squared_error(y, y_pred)
    print(f"MSE: {mse}")

    W = model.coef_  # shape (n_features_y, n_features_x) in sklearn's convention
    b = model.intercept_  # shape (n_features_y,)

    absmax = abs(W).max()

    plt.figure()
    plt.imshow(
        W.T,
        aspect="auto",
        interpolation="nearest",
        cmap="seismic",
        vmin=-absmax,
        vmax=absmax,
    )
    plt.yticks(
        ticks=range(len(datamix_labels)),
        labels=datamix_labels,
        fontsize=8,
    )
    plt.xticks(
        ticks=range(len(target_labels)),
        labels=target_labels,
        fontsize=8,
        rotation=45,
        ha="right",
    )
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(args.path, "out/linear_fit", "model_weights.png"), dpi=300)

    plt.figure()
    plt.plot(y, y_pred, ".", label=target_labels)
    plt.legend()
    plt.xlabel("True target values")
    plt.ylabel("Predicted target values")
    plt.savefig(
        os.path.join(args.path, "out/linear_fit", "model_predictions.png"), dpi=300
    )
