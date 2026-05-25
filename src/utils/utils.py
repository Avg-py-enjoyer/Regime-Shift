import os
import pandas as pd
import matplotlib.pyplot as plt


def create_directories():

    folders = [

        "data",
        "data/raw",
        "data/processed",

        "reports",

        "notebooks"

    ]

    for folder in folders:

        os.makedirs(
            folder,
            exist_ok=True
        )

    print("Project directories created.")


def save_dataframe(df, path):

    directory = os.path.dirname(path)

    os.makedirs(
        directory,
        exist_ok=True
    )

    df.to_csv(path)

    print(f"Saved CSV: {path}")


def load_dataframe(path):

    return pd.read_csv(
        path,
        index_col=0,
        parse_dates=True
    )


def save_plot(filename):

    reports_dir = "reports"

    os.makedirs(
        reports_dir,
        exist_ok=True
    )

    filepath = os.path.join(
        reports_dir,
        filename
    )

    plt.savefig(

        filepath,

        dpi=300,

        bbox_inches="tight"

    )

    print(f"Saved Plot: {filepath}")