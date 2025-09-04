import os
import argparse
import torch
from estimator import TorchExpEstimator
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
    os.makedirs(os.path.join(args.path, "out/exp_fit"), exist_ok=True)

    datamix, target = read_results(os.path.join(args.path, "out", "regmix_results.csv"))

    datamix_labels = datamix.columns
    print(f"Datamix labels: {datamix_labels}")

    target_labels = target.columns
    print(f"Target labels: {target_labels}")

    x = torch.tensor(datamix.to_numpy(), dtype=torch.float32)
    y = torch.tensor(np.log(target.to_numpy()), dtype=torch.float32)

    N, input_dim = x.shape
    N, output_dim = y.shape

    model = TorchExpEstimator(input_dim=input_dim, output_dim=output_dim)
    model.fit(x, y)

    y_pred = model.predict(x)

    # 6. Evaluate
    mse = mean_squared_error(y, y_pred)
    print(f"MSE: {mse}")

    torch.save(
        model.model.state_dict(),
        os.path.join(args.path, "out/exp_fit", "torch_exp_model.pth"),
    )

    # PLOT T
    fig = plt.figure(figsize=(6, 5))
    T_mat = model.model.T.detach().cpu().numpy()
    absmax = abs(T_mat).max()
    print(T_mat)
    plt.imshow(
        T_mat.transpose(),
        aspect="auto",
        interpolation="nearest",
        cmap="seismic",
        vmin=-absmax,
        vmax=absmax,
    )

    # Set ticks and labels
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

    plt.title("Learned T matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.path, "out/exp_fit", "learned_T_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.figure()
    plt.plot(y, y_pred, ".", label=target_labels)
    plt.legend()
    plt.xlabel("True target values")
    plt.ylabel("Predicted target values")
    plt.savefig(os.path.join(args.path, "out/exp_fit", "predictions.png"), dpi=300)
