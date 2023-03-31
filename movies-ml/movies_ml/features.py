import pandas as pd
from db.connect import read_df, save_df
from collections import defaultdict
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import GridSearchCV
import lightgbm as lgbm
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer


def lgbm_predict(X) -> np.array:
    lgbm = pickle.load(open("lgbmreg.sav", "rb"))
    return lreg.predict(X)


def train_lgbmreg() -> None:
    df = read_df("latest.xlsx", "engineered")
    X = df.select_dtypes(exclude=["object"])
    # X = df.drop(["searches_min", "searches_max"], axis=1)
    X = X.fillna(X.median())
    y = X.pop("BoxOffice")

    param_grid = {
        "n_estimators": [100, 80, 60, 55, 51, 45, 20],
        "max_depth": [0, 20, 40],
        "reg_lambda": [0.26, 0.25, 0.2, 0],
    }

    grid = GridSearchCV(
        lgbm.LGBMRegressor(), param_grid, refit=True, verbose=3, n_jobs=-1
    )
    regr_trans = TransformedTargetRegressor(
        regressor=grid,
        transformer=QuantileTransformer(
            n_quantiles=80, output_distribution="normal"
        ),
    )

    # fitting the model for grid search
    grid_result = regr_trans.fit(X, y)
    best_params = grid_result.regressor_.best_params_

    # using best params to create and fit model
    best_model = lgbm.LGBMRegressor(
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        reg_lambda=best_params["reg_lambda"],
    )
    regr_trans = TransformedTargetRegressor(
        regressor=best_model,
        transformer=QuantileTransformer(output_distribution="normal"),
    )
    regr_trans.fit(X, y)

    pickle.dump(regr_trans, open("lgbmreg.sav", "wb"))
    print("done")


def lreg_predict(X) -> np.array:
    lreg = pickle.load(open("lreg.sav", "rb"))
    return lreg.predict(X)


def train_linear_regression_model() -> None:
    df = read_df("latest.xlsx", "engineered")
    reg = linear_model.LinearRegression()
    X = df.select_dtypes(exclude=["object"])
    # X = df.drop(["searches_min", "searches_max"], axis=1)
    X = X.fillna(X.median())
    y = X.pop("BoxOffice")

    reg.fit(X, y)
    pickle.dump(reg, open("lreg.sav", "wb"))
    print("done")


def build_features() -> pd.DataFrame:
    df = read_df("latest.xlsx", "cleaned")
    df = df.set_index("Title")
    df = df.drop(
        [
            "Ratings",
            "Plot",
            "imdbID",
            "Response",
            "Released",
            "Awards",
            "Poster",
            "Year",
            "DVD",
            "Type",
        ],
        axis=1,
    )
    cat_dict = get_significant_cat_features(df)

    # One hot-encode cat features
    df_cat = df.select_dtypes(include=["object"])
    for f in cat_dict:
        for k in cat_dict[f]:
            df_cat[k] = df_cat[f].apply(
                lambda x: int(k in unique(x)) if isinstance(x, str) else 0
            )
        df_cat = df_cat.drop(f, axis=1)

    # Impute Rotten Tomatoes ratings
    df["Metascore"] = df["Metascore"].fillna(df["Metascore"].mean())
    reg1 = get_rgr_model_rotten_tomatoes(df)
    df["Rotten Tomatoes"] = df["Rotten Tomatoes"].fillna(
        pd.Series(
            reg1.predict(
                df[
                    [
                        "Runtime",
                        "imdbRating",
                        "log_imdbVotes",
                        "imdbVotes",
                        "Metascore",
                    ]
                ]
            ),
            index=df.index,
        )
    )

    # PCA to get agg. rating
    pca = get_pca_ratings(df)
    df["agg_rating"] = pd.DataFrame(
        pca.transform(df[["Metascore", "imdbRating", "Rotten Tomatoes"]]),
        index=df.index,
        columns=["agg_rating"],
    )

    # Add one hot categorical variables
    df = df.merge(df_cat, how="left", left_index=True, right_index=True)

    save_df("latest.xlsx", "engineered", df)


def get_rgr_model_rotten_tomatoes(
    df: pd.DataFrame,
) -> linear_model.LinearRegression:
    # Impute Rotten Tomatoes
    X = df[
        [
            "Runtime",
            "imdbRating",
            "log_imdbVotes",
            "imdbVotes",
            "Metascore",
            "Rotten Tomatoes",
        ]
    ]

    X = X.dropna()
    y = X.pop("Rotten Tomatoes")

    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    return reg


def get_pca_ratings(df: pd.DataFrame) -> PCA:
    df_ratings = df[["Metascore", "imdbRating", "Rotten Tomatoes"]]
    pca = PCA(n_components=1)
    df_ratings = pd.DataFrame(
        pca.fit_transform(df_ratings), index=df_ratings.index
    )
    df_ratings = df_ratings.rename({0: "agg_rating"}, axis=1)
    return pca


def unique(x):
    t = x.split(",")
    new = set([a.strip() for a in t])
    return new


def get_significant_cat_features(df: pd.DataFrame) -> dict:
    df_cat = df.select_dtypes(include=["object"])

    def get_unique(df, key):
        arr = set()
        for d in df[key].dropna().unique():
            arr = arr.union(unique(d))
        return list(arr)

    uniques = dict()

    for c in df_cat.columns:
        uniques[c] = get_unique(df_cat, c)

    d = defaultdict(dict)

    for c in df_cat.columns:
        for word in uniques[c]:
            d[c][word] = (
                df[c]
                .apply(
                    lambda x: int(word in unique(x))
                    if isinstance(x, str)
                    else pd.NA
                )
                .sum()
            )

    d = {(i, j): d[i][j] for i in d.keys() for j in d[i].keys()}

    mux = pd.MultiIndex.from_tuples(d.keys(), name=["Feature", "Cat"])
    df_uniques_count = pd.DataFrame(
        list(d.values()), index=mux, columns=["Count"]
    )

    df_counts = df_uniques_count.groupby("Feature").count()
    df_threshold = (
        df_uniques_count.groupby("Feature").sum() / df_counts * 2
    )  # Threshold

    merged_df = pd.merge(
        df_uniques_count.reset_index(), df_threshold, on="Feature", how="left"
    ).set_index(["Feature", "Cat"])
    filtered_df = merged_df.loc[merged_df["Count_x"] >= merged_df["Count_y"]]

    d = filtered_df["Count_x"].to_dict()

    nested_dict = defaultdict(dict)

    for k, v in d.keys():
        nested_dict[k][v] = d[(k, v)]

    return nested_dict


if __name__ == "__main__":
    build_features()
    # train_linear_regression_model()
    train_lgbmreg()
