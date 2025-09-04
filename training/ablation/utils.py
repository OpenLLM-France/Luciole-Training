import pandas as pd


def read_results(results_path):
    df = pd.read_csv(results_path)

    datamix = df.loc[:, df.columns.str.startswith("datamix:")]
    target = df.loc[
        :,
        df.columns.str.startswith("target:")
        & ~df.columns.str.contains("_average")
        & ~df.columns.str.contains("all"),
    ]
    return datamix, target
