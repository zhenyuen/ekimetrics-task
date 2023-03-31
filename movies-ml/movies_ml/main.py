#!/usr/bin/env python

from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from logging import getLogger, DEBUG, StreamHandler, Formatter
from typing import Union
from db.connect import read_df, save_df, get_all_monthly_updates
from time import sleep
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import random
import pandas as pd
from urllib.parse import quote
from requests import get
import numpy as np
import json
from features import (
    build_features,
    train_lgbmreg,
    train_linear_regression_model,
    lreg_predict,
    lgbm_predict,
)
import calendar
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
logger = getLogger("movies-ml")
logger.setLevel(DEBUG)
formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


@app.route("/")
def hello_world() -> None:
    return "<p>Hello, World!</p>"


@app.route("/lgbmreg_predict")
def lgbmreg_predict_ep() -> np.array:
    X = request.get_json()
    return lgbm_predict(X)


@app.route("/lreg_predict")
def lreg_predict_ep() -> np.array:
    X = request.get_json()
    return lreg_predict(X)


@app.route("/lreg_train")
def lreg_train_ep() -> None:
    train_linear_regression_model()
    return "<p>Training completed!</p>"


@app.route("/lgbmreg_train")
def lgbmreg_train_ep() -> None:
    train_lgbmreg()
    return "<p>Training completed!</p>"


@app.route("/build_features")
def build_features_ep() -> None:
    build_features()
    return


def update() -> None:
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")

    populate_data(date_str)
    df_agg_latest = compile_aggregates()
    print(df_agg_latest)
    clean_agg(df_agg_latest)


def clean_agg(df_agg: pd.DataFrame) -> pd.DataFrame:
    # drop columns with more than 50% null values

    threshold = 0.5
    df_agg = df_agg.replace("N/A", pd.NA)
    df_agg = df_agg.dropna(thresh=len(df_agg) * threshold, axis=1)

    # Process numerical data
    df2 = df_agg

    df2["Runtime"] = df2["Runtime"].apply(
        lambda x: int(x.split()[0]) if x is not pd.NA else np.NaN
    )
    df2["imdbVotes"] = df2["imdbVotes"].apply(
        lambda x: int(x.replace(",", "")) if x is not pd.NA else np.NaN
    )
    df2["BoxOffice"] = df2["BoxOffice"].apply(
        lambda x: int(x.replace(",", "")[1:]) if isinstance(x, str) else np.NaN
    )
    df2["Metascore"] = df2["Metascore"].apply(
        lambda x: int(x) / 100 if not np.isnan(x) else np.NaN
    )
    df2["imdbRating"] = df2["imdbRating"].apply(
        lambda x: int(x) / 10 if not np.isnan(x) else np.NaN
    )
    # Flatten ratings
    def flatten_ratings(row):
        x = {}
        row["Ratings"] = row["Ratings"].replace("'", '"')
        obj = json.loads(row["Ratings"])
        for s in obj:
            if s["Source"] is not None:
                x[s["Source"]] = s["Value"]

        if "Rotten Tomatoes" in x:
            row["Rotten Tomatoes"] = float(x["Rotten Tomatoes"][:-1]) / 100
        else:
            row["Rotten Tomatoes"] = np.NaN
        return row

    df2 = df2.apply(flatten_ratings, axis=1)
    df2["log_imdbVotes"] = np.log(df2["imdbVotes"])

    # Process datetime data
    df2["Year"] = pd.to_datetime(df2["Year"])
    df2["Released"] = pd.to_datetime(df2["Released"])
    df2["DVD"] = pd.to_datetime(df2["DVD"])
    save_df("latest.xlsx", "cleaned", df2)


def populate_data(date_str: str) -> None:
    fname = "movies.xlsx"
    titles = read_df(fname, "Sheet1").title.to_list()
    # titles = ['Black Christmas', 'London']
    timeframe = [get_monthly_timeframe()] * len(titles)

    logger.debug(timeframe[0])

    df_omdb_update = retrieve_omdb(titles)
    df_r_queries_update = retrieve_r_queries(titles, df_omdb_update, timeframe)
    df_r_searches_update = retrieve_r_searches(
        titles, df_r_queries_update, timeframe
    )
    df_searches_update = retrieve_searches(titles, timeframe)
    df_agg_update = compute_aggregate(
        titles, df_r_searches_update, df_searches_update, df_omdb_update
    )
    # print(df_agg_update.head())

    fname_update = f"./monthly/{date_str}.xlsx"
    save_df(fname_update, "omdb", df_omdb_update)
    save_df(fname_update, "searches", df_searches_update)
    save_df(fname_update, "related_searches", df_r_searches_update)
    save_df(fname_update, "agg", df_agg_update)


def compile_aggregates() -> pd.DataFrame:
    # titles = read_df(fname, "Sheet1").title.to_list()
    df_agg = read_df("movies.xlsx", "agg")

    months = 12
    for f in get_all_monthly_updates():
        df_agg_update = read_df(f, "agg")
        df_agg["searches_avg"] = (
            df_agg["searches_avg"] + df_agg_update["searches_avg"]
        ) / 2
        df_agg["searches_sum"] = (
            df_agg["searches_sum"] + df_agg_update["searches_sum"]
        )
        df_agg["searches_max"] = np.maximum(
            df_agg["searches_max"], df_agg_update["searches_max"]
        )
        df_agg["searches_min"] = np.minimum(
            df_agg["searches_min"], df_agg_update["searches_min"]
        )
        df_agg["searches_std"] = (
            df_agg["searches_avg"] * months + df_agg_update["searches_avg"]
        ) / (
            months + 1
        )  # Simplification
        months += 1

    df_agg = df_agg.set_index("Title")
    save_df("latest.xlsx", "agg", df_agg)
    return df_agg


def compute_aggregate(
    titles: list[str],
    df_r_s: pd.DataFrame,
    df_s: pd.DataFrame,
    df_omdb: pd.DataFrame,
) -> pd.DataFrame:
    df_r_s_sum = df_r_s.groupby("Title").sum()
    df_s_avg = df_s.groupby(level=0).mean()
    df_s_std = df_s.groupby(level=0).std()
    df_s_max = df_s.groupby(level=0).max()
    df_s_min = df_s.groupby(level=0).min()

    df_rolling_max = df_s.groupby(level=0, group_keys=False).apply(
        lambda x: x.rolling(window=4).sum()
    )
    df_rolling_max = df_rolling_max.groupby(level=0, group_keys=False).max()

    df1 = df_omdb.copy()
    df1 = df1.merge(
        df_s_avg["searches"], how="left", left_index=True, right_index=True
    )
    df1 = df1.rename(columns={"searches": "searches_avg"})
    df1 = df1.merge(
        df_s_std["searches"], how="left", left_index=True, right_index=True
    )
    df1 = df1.rename(columns={"searches": "searches_std"})
    df1 = df1.merge(
        df_s_max["searches"], how="left", left_index=True, right_index=True
    )
    df1 = df1.rename(columns={"searches": "searches_max"})
    df1 = df1.merge(
        df_s_min["searches"], how="left", left_index=True, right_index=True
    )
    df1 = df1.rename(columns={"searches": "searches_min"})
    df1 = df1.merge(
        df_rolling_max["searches"],
        how="left",
        left_index=True,
        right_index=True,
    )
    df1 = df1.rename(columns={"searches": "searches_rolling_max"})
    df1 = df1.merge(
        df_r_s_sum["searches"], how="left", left_index=True, right_index=True
    )
    df1 = df1.rename(columns={"searches": "searches_sum"})

    return df1


def get_date_one_year(date_string: str) -> str:
    date_obj = datetime.strptime(date_string, "%d %b %Y")
    new_date_obj = date_obj + relativedelta(years=1)

    return "{} {}".format(
        date_obj.strftime("%Y-%m-%d"), new_date_obj.strftime("%Y-%m-%d")
    )


def get_monthly_timeframe() -> str:
    date_obj = datetime.today()
    new_date_obj = date_obj - relativedelta(month=1)

    return "{} {}".format(
        new_date_obj.strftime("%Y-%m-%d"), date_obj.strftime("%Y-%m-%d")
    )


def retrieve_searches(titles: list[str], timeframe: str) -> pd.DataFrame:
    def retreive(title: list[str], timeframe: str, d: dict) -> dict:
        for t, tf in zip(title, timeframe):
            logger.debug(t)
            pytrend.build_payload(kw_list=[t], timeframe=tf)
            try:
                d[t] = pytrend.interest_over_time()
                sleep(
                    10 + random.randint(1, 5)
                )  # Time out to ensure google does not flag

            except ResponseError as e:
                logger.error("Error:", e)
                d[t] = None
                # Invalid request, usually caused by released date being earlier than 2004
                sleep(3)  # Time out to ensure google does not flag

    # timeframe = [
    #     get_date_one_year(released_date)
    #     for released_date in df_omdb["Released"]
    # ]

    requests_args = {
        "headers": {
            "Cookie": "__utma=10102256.1490800305.1679441551.1679844599.1679844599.2;"
            "__utmb=10102256.1.10.1679844599; __utmc=10102256; __utmd=1; __u"
            "tmt=1; __utmz=10102256.1679844599.1.1.utmcsr=trends.google.com|"
            "utmccn=(referral)|utmcmd=referral|utmcct=/; SIDCC=AFvIBn8l-M2TM"
            "4WYuiIwgDX_jWecoplbo1fngwm68Ot3wJWqsmjrEuK-QFCNqDx84KOy9MjyOU-4"
            "; __Secure-1PSIDCC=AFvIBn8GVxzd1Xva_gW-1-gir4mRsds0dU2dRsQjftSF"
            "Y5tEjO05NY3_KCgvjZdUbFPLBmP8jFo; __Secure-3PSIDCC=AFvIBn8sidF5kq"
            "3DqDKsREglAB0A3iCkc7SA7dpfGpsr_H39n299KWALkdpk1pSzgzMelA2mhv4; _"
            "ga_VWZPXDNJJB=GS1.1.1679848310.5.0.1679848310.0.0.0; _gat_gtag_UA"
            "_4401283=1; 1P_JAR=2023-3-26-16; NID=511=KpGyZzJSh9Dog7UZ3jafxppUx"
            "sTV74MK_aRXXz6f54jZ46s63DvQ7f9tD-Jrn5FxrfxdLMk_1jcJtOj2OfZMUKSkMow"
            "1-lw4tXclwxnIlZD6TZbWf3JyBfx61F2pZTFd0_OBMslqKZBVkyNGlwI34UmkSaHuU"
            "asmXXRfQLBuypawi4MYvA0JqnyjGgsHDA4LFnvSHnh1ofbKdblWkvfaR4fd5fmmnQh"
            "QDyO2DM2z0OPovNc_ygD1E8EgnpZaHaaf1zjb5faDePAD9NWIWNiucXyekOPoxnsd6z"
            "p0d0JClKG1hfsQjbUUwjLE7ae0CMVBaiF2WgTVWnOCS6aKJaMD2hpKdQJ4Iyb8IX_pq9"
            "7XkLd-g3N2r1SNLE7kcie8-YfmzccpR35q4SUsUPhYbkeCQ_MByZcovX5cmEMcX5augK"
            "mrljAH6ULxVLCcEeGH9iblIg4UJQfSzw; _ga=GA1.3.1490800305.1679441551; _"
            "gid=GA1.3.353626381.1679844591; AEC=AUEFqZd8mLhQBGS0fhetYNWeF2paE0jSO"
            "3V3yygv8lmQ2Bv081GFl0p85as; APISID=_7EJ-a5syz9zJgEU/AAT5Peub6PlkFkO5E"
            "; HSID=APaE7HbeXzljG9lfq; SAPISID=vWm--1fOhBOhKj1Y/AHX6EI69vaZFY39c4; "
            "SID=UggFQiXde0ut5rzblEwpNezi6Pl1se0M1QgafLtQw_SYfEHd1C-AAXa8S8RHuvp9W4y"
            "7Vw.; SSID=AIe3EdCRUnqT0ETZI; __Secure-1PAPISID=vWm--1fOhBOhKj1Y/AHX6E"
            "I69vaZFY39c4; __Secure-1PSID=UggFQiXde0ut5rzblEwpNezi6Pl1se0M1QgafLtQw_"
            "SYfEHdEXbXvug4To1QZZxuiOwwqw.; __Secure-3PAPISID=vWm--1fOhBOhKj1Y/AHX6EI"
            "69vaZFY39c4; __Secure-3PSID=UggFQiXde0ut5rzblEwpNezi6Pl1se0M1QgafLtQw_SY"
            "fEHdP7A18IAyN6mLGaE9sABhAg.; SEARCH_SAMESITE=CgQI7pcB; OTZ=6952293_52_52"
            "_123900_48_436380; OGPC=19031496-1:19031779-1:; CONSENT=PENDING+812; SOCS"
            "=CAISHAgBEhJnd3NfMjAyMjA4MTEtMF9SQzMaAmVuIAEaBgiA64WYBg; ANID=AHWqTUntC6"
            "syHP0EaAPEEzqXA9xCScNDLDS9xmd-i-cLlzSHT6x6n2LnkvG5STTi",
            "Accept": "application/json, text/plain, */*, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Host": "trends.google.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
            "Accept-Language": "en-GB,en;q=0.9",
            "Referer": "https://trends.google.com/trends/explore?date=all&q=The%20Age%20of%20Adaline&hl=en-GB",
            "Connection": "keep-alive",
        }
    }

    pytrend = TrendReq(requests_args=requests_args)
    d = dict()

    retreive(titles, timeframe, d)

    for k, v in d.items():
        if v is not None:
            d[k] = v.rename(columns={k: "searches"})

    df = pd.concat(d, axis=0)
    df = df.drop("isPartial", axis=1)
    df.index = df.index.set_names(["Title", "date"])
    return df


def retrieve_r_searches(
    titles: list[str], df_r_queries: pd.DataFrame, timeframe: str
) -> pd.DataFrame:
    def retrieve(title: list[str], timeframe: str, d: dict) -> dict:
        for t, tf in zip(title, timeframe):
            for query in df_r_queries.loc[t]["query"]:
                pytrend.build_payload(kw_list=[query], timeframe=tf)
                try:
                    d[t][query] = pytrend.interest_over_time()
                    sleep(
                        10 + random.randint(1, 5)
                    )  # Time out to ensure google does not flag

                except ResponseError as e:
                    logger.error("Error:", e)
                    d[t][query] = None
                    # Invalid request, usually caused by released date being earlier than 2004
                    sleep(3)

    requests_args = {
        "headers": {
            "Cookie": "__utma=10102256.1490800305.1679441551.1679844599.1679844599.2; __utmb=10102256.1.10.1679844599;"
            " __utmc=10102256; __utmd=1; __utmt=1; __utmz=10102256.1679844599.1.1.utmcsr=trends.google.com|u"
            "tmccn=(referral)|utmcmd=referral|utmcct=/; SIDCC=AFvIBn8l-M2TM4WYuiIwgDX_jWecoplbo1fngwm68Ot3wJ"
            "WqsmjrEuK-QFCNqDx84KOy9MjyOU-4; __Secure-1PSIDCC=AFvIBn8GVxzd1Xva_gW-1-gir4mRsds0dU2dRsQjftSFY5t"
            "EjO05NY3_KCgvjZdUbFPLBmP8jFo; __Secure-3PSIDCC=AFvIBn8sidF5kq3DqDKsREglAB0A3iCkc7SA7dpfGpsr_H39n2"
            "99KWALkdpk1pSzgzMelA2mhv4; _ga_VWZPXDNJJB=GS1.1.1679848310.5.0.1679848310.0.0.0; _gat_gtag_UA_44"
            "01283=1; 1P_JAR=2023-3-26-16; NID=511=KpGyZzJSh9Dog7UZ3jafxppUxsTV74MK_aRXXz6f54jZ46s63DvQ7f9tD-J"
            "rn5FxrfxdLMk_1jcJtOj2OfZMUKSkMow1-lw4tXclwxnIlZD6TZbWf3JyBfx61F2pZTFd0_OBMslqKZBVkyNGlwI34UmkSaHu"
            "UasmXXRfQLBuypawi4MYvA0JqnyjGgsHDA4LFnvSHnh1ofbKdblWkvfaR4fd5fmmnQhQDyO2DM2z0OPovNc_ygD1E8EgnpZaH"
            "aaf1zjb5faDePAD9NWIWNiucXyekOPoxnsd6zp0d0JClKG1hfsQjbUUwjLE7ae0CMVBaiF2WgTVWnOCS6aKJaMD2hpKdQJ4Iy"
            "b8IX_pq97XkLd-g3N2r1SNLE7kcie8-YfmzccpR35q4SUsUPhYbkeCQ_MByZcovX5cmEMcX5augKmrljAH6ULxVLCcEeGH9ib"
            "lIg4UJQfSzw; _ga=GA1.3.1490800305.1679441551; _gid=GA1.3.353626381.1679844591; AEC=AUEFqZd8mLhQBGS"
            "0fhetYNWeF2paE0jSO3V3yygv8lmQ2Bv081GFl0p85as; APISID=_7EJ-a5syz9zJgEU/AAT5Peub6PlkFkO5E; HSID=APa"
            "E7HbeXzljG9lfq; SAPISID=vWm--1fOhBOhKj1Y/AHX6EI69vaZFY39c4; SID=UggFQiXde0ut5rzblEwpNezi6Pl1se0M1"
            "QgafLtQw_SYfEHd1C-AAXa8S8RHuvp9W4y7Vw.; SSID=AIe3EdCRUnqT0ETZI; __Secure-1PAPISID=vWm--1fOhBOhKj1"
            "Y/AHX6EI69vaZFY39c4; __Secure-1PSID=UggFQiXde0ut5rzblEwpNezi6Pl1se0M1QgafLtQw_SYfEHdEXbXvug4To1QZ"
            "ZxuiOwwqw.; __Secure-3PAPISID=vWm--1fOhBOhKj1Y/AHX6EI69vaZFY39c4; __Secure-3PSID=UggFQiXde0ut5rzbl"
            "EwpNezi6Pl1se0M1QgafLtQw_SYfEHdP7A18IAyN6mLGaE9sABhAg.; SEARCH_SAMESITE=CgQI7pcB; OTZ=6952293_52_5"
            "2_123900_48_436380; OGPC=19031496-1:19031779-1:; CONSENT=PENDING+812; SOCS=CAISHAgBEhJnd3NfMjAyMjA4"
            "MTEtMF9SQzMaAmVuIAEaBgiA64WYBg; ANID=AHWqTUntC6syHP0EaAPEEzqXA9xCScNDLDS9xmd-i-cLlzSHT6x6n2LnkvG5STTi",
            "Accept": "application/json, text/plain, */*, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Host": "trends.google.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
            "Accept-Language": "en-GB,en;q=0.9",
            "Referer": "https://trends.google.com/trends/explore?date=all&q=The%20Age%20of%20Adaline&hl=en-GB",
            "Connection": "keep-alive",
        }
    }

    pytrend = TrendReq(requests_args=requests_args)
    d = defaultdict(dict)
    retrieve(titles, timeframe, d)

    indices = d.keys()
    dfs2 = []
    for k in indices:
        if d[k] is not None:
            dfi = []
            keys = d[k].keys()
            print(k, keys)
            for q in keys:
                if d[k][q] is not None:
                    dfi.append(d[k][q][q])
            if dfi:
                #             dfs2.append(pd.concat(dfi, axis=0, keys=keys))
                dfs2.append(pd.DataFrame(pd.concat(dfi, axis=0, keys=keys)))
            else:
                dfs2.append(None)

    df = pd.concat(dfs2, axis=0, keys=indices)
    df = df.rename(columns={0: "searches"})
    df = pd.DataFrame(df.loc[..., "searches"])
    df.index.names = ["Title", "Query", "date"]
    return df


def retrieve_r_queries(
    titles: list[str], df_omdb: pd.DataFrame, timeframe: str
) -> pd.DataFrame:
    def retreive(title: list[str], timeframe: str, d: dict) -> dict:
        for t, tf in zip(title, timeframe):
            logger.debug(t)
            pytrend.build_payload(kw_list=[t], timeframe=tf)
            try:
                d[t] = pytrend.related_queries()
                sleep(
                    10 + random.randint(1, 5)
                )  # Time out to ensure google does not flag

            except ResponseError as e:
                logger.error("Error:", e)
                d[t] = None
                # Invalid request, usually caused by released date being earlier than 2004
                sleep(3)  # Time out to ensure google does not flag

    # timeframe = [
    #     get_date_one_year(released_date)
    #     for released_date in df_omdb["Released"]
    # ]

    requests_args = {
        "headers": {
            "Cookie": "__utma=10102256.1490800305.1679441551.1679844599.1679844599.2;"
            "__utmb=10102256.1.10.1679844599; __utmc=10102256; __utmd=1; __u"
            "tmt=1; __utmz=10102256.1679844599.1.1.utmcsr=trends.google.com|"
            "utmccn=(referral)|utmcmd=referral|utmcct=/; SIDCC=AFvIBn8l-M2TM"
            "4WYuiIwgDX_jWecoplbo1fngwm68Ot3wJWqsmjrEuK-QFCNqDx84KOy9MjyOU-4"
            "; __Secure-1PSIDCC=AFvIBn8GVxzd1Xva_gW-1-gir4mRsds0dU2dRsQjftSF"
            "Y5tEjO05NY3_KCgvjZdUbFPLBmP8jFo; __Secure-3PSIDCC=AFvIBn8sidF5kq"
            "3DqDKsREglAB0A3iCkc7SA7dpfGpsr_H39n299KWALkdpk1pSzgzMelA2mhv4; _"
            "ga_VWZPXDNJJB=GS1.1.1679848310.5.0.1679848310.0.0.0; _gat_gtag_UA"
            "_4401283=1; 1P_JAR=2023-3-26-16; NID=511=KpGyZzJSh9Dog7UZ3jafxppUx"
            "sTV74MK_aRXXz6f54jZ46s63DvQ7f9tD-Jrn5FxrfxdLMk_1jcJtOj2OfZMUKSkMow"
            "1-lw4tXclwxnIlZD6TZbWf3JyBfx61F2pZTFd0_OBMslqKZBVkyNGlwI34UmkSaHuU"
            "asmXXRfQLBuypawi4MYvA0JqnyjGgsHDA4LFnvSHnh1ofbKdblWkvfaR4fd5fmmnQh"
            "QDyO2DM2z0OPovNc_ygD1E8EgnpZaHaaf1zjb5faDePAD9NWIWNiucXyekOPoxnsd6z"
            "p0d0JClKG1hfsQjbUUwjLE7ae0CMVBaiF2WgTVWnOCS6aKJaMD2hpKdQJ4Iyb8IX_pq9"
            "7XkLd-g3N2r1SNLE7kcie8-YfmzccpR35q4SUsUPhYbkeCQ_MByZcovX5cmEMcX5augK"
            "mrljAH6ULxVLCcEeGH9iblIg4UJQfSzw; _ga=GA1.3.1490800305.1679441551; _"
            "gid=GA1.3.353626381.1679844591; AEC=AUEFqZd8mLhQBGS0fhetYNWeF2paE0jSO"
            "3V3yygv8lmQ2Bv081GFl0p85as; APISID=_7EJ-a5syz9zJgEU/AAT5Peub6PlkFkO5E"
            "; HSID=APaE7HbeXzljG9lfq; SAPISID=vWm--1fOhBOhKj1Y/AHX6EI69vaZFY39c4; "
            "SID=UggFQiXde0ut5rzblEwpNezi6Pl1se0M1QgafLtQw_SYfEHd1C-AAXa8S8RHuvp9W4y"
            "7Vw.; SSID=AIe3EdCRUnqT0ETZI; __Secure-1PAPISID=vWm--1fOhBOhKj1Y/AHX6E"
            "I69vaZFY39c4; __Secure-1PSID=UggFQiXde0ut5rzblEwpNezi6Pl1se0M1QgafLtQw_"
            "SYfEHdEXbXvug4To1QZZxuiOwwqw.; __Secure-3PAPISID=vWm--1fOhBOhKj1Y/AHX6EI"
            "69vaZFY39c4; __Secure-3PSID=UggFQiXde0ut5rzblEwpNezi6Pl1se0M1QgafLtQw_SY"
            "fEHdP7A18IAyN6mLGaE9sABhAg.; SEARCH_SAMESITE=CgQI7pcB; OTZ=6952293_52_52"
            "_123900_48_436380; OGPC=19031496-1:19031779-1:; CONSENT=PENDING+812; SOCS"
            "=CAISHAgBEhJnd3NfMjAyMjA4MTEtMF9SQzMaAmVuIAEaBgiA64WYBg; ANID=AHWqTUntC6"
            "syHP0EaAPEEzqXA9xCScNDLDS9xmd-i-cLlzSHT6x6n2LnkvG5STTi",
            "Accept": "application/json, text/plain, */*, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Host": "trends.google.com",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
            "Accept-Language": "en-GB,en;q=0.9",
            "Referer": "https://trends.google.com/trends/explore?date=all&q=The%20Age%20of%20Adaline&hl=en-GB",
            "Connection": "keep-alive",
        }
    }

    pytrend = TrendReq(requests_args=requests_args)
    d = dict()

    retreive(titles, timeframe, d)

    indices = d.keys()
    dfs = []
    for k in indices:
        if d[k] is not None:
            key = list(d[k].keys())[0]
            t = d[k][key]["top"]
            dfs.append(
                t[t["value"] > 50]
            )  # Only take queries with scores above 50
        else:
            dfs.append(d[k])

    df = pd.concat(dfs, axis=0, keys=indices)
    df.index = df.index.set_names(["Title", "No"])
    return df


def retrieve_omdb(titles: list[str]) -> pd.DataFrame:
    # Example, http://www.omdbapi.com/?i=tt3896198&apikey=833f1e37
    api_key = os.getenv("OMDB_API_KEY")
    titles_encoded = [quote(t) for t in titles]
    urls = [
        "http://www.omdbapi.com/?t={}&apikey={}".format(t, api_key)
        for t in titles_encoded
    ]
    res = [get(url).json() for url in urls]
    df = pd.DataFrame.from_dict(res)
    df = df.set_index("Title")
    return df


def dummy():
    logger.info("Scheduler ok")


sched = BackgroundScheduler(daemon=True)
sched.add_job(update, "cron", day=calendar.monthrange(date.today().year, date.today().month)[1])
sched.add_job(dummy, "interval", seconds=60)

if __name__ == "__main__":
    sched.start()
    app.run(host="127.0.0.1", port=8080, debug=True)
