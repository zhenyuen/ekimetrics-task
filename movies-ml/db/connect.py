import pandas as pd
import pathlib
import os


from typing import Optional


def read_df(
    fname: str, sheet_name: str, index_col: Optional[list] = []
) -> pd.DataFrame:
    dir = pathlib.Path().resolve()
    path = os.path.join(dir, "db", fname)
    df = pd.read_excel(path, sheet_name=sheet_name, index_col=index_col)
    return df


def save_df(fname: str, sheet_name: str, df: pd.DataFrame) -> None:
    dir = pathlib.Path().resolve()
    path = os.path.join(dir, "db", fname)
    if not os.path.exists(path):
        with pd.ExcelWriter(path, mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(
            path, mode="a", if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name)


def get_all_monthly_updates() -> list[str]:
    dir = pathlib.Path().resolve()
    path = os.path.join(dir, "db/monthly")
    files = os.listdir(path)
    return [os.path.join(path, f) for f in files]
