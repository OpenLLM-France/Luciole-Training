import os
import argparse
import pandas as pd
import torch
from estimator import TorchExpEstimator
from sklearn.metrics import mean_squared_error
import numpy as np

MAIN_PATH = os.getenv("OpenLLM_OUTPUT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(MAIN_PATH, "ablations/train/regmix/common-pile"),
    )
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.dir, "regmix_results.csv"))

    # FIT
    datamix = df.loc[:, df.columns.str.startswith("datamix:")]
    datamix_labels = datamix.columns
    print(f"Datamix labels: {datamix_labels}")
    target = df.loc[:, df.columns.str.startswith("target:")]
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

    torch.save(model.model.state_dict(), os.path.join(args.dir, "torch_exp_model.pth"))
