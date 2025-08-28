import os
import argparse
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

MAIN_PATH = os.getenv("OpenLLM_OUTPUT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(MAIN_PATH, "ablations/train/regmix/common-pile_v2"),
    )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.dir, "out", "regmix_results.csv"))

    # FIT
    datamix = df.loc[:, df.columns.str.startswith("datamix:")]
    datamix_labels = datamix.columns
    print(f"Datamix labels: {datamix_labels}")
    target = df.loc[
        :,
        df.columns.str.startswith("target:")
        & ~df.columns.str.contains("_average")
        & ~df.columns.str.contains("all"),
    ]
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
    plt.savefig(os.path.join(args.dir, "out", "linear_model_weights.png"), dpi=300)

    plt.figure()
    plt.plot(y, y_pred, ".", label=target_labels)
    plt.legend()
    plt.xlabel("True target values")
    plt.ylabel("Predicted target values")
    plt.savefig(os.path.join(args.dir, "out", "linear_model_predictions.png"), dpi=300)
