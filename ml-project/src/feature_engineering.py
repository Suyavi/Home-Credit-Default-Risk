"""Home Credit Level-2 style feature engineering (application + multi-table aggregates)."""

from __future__ import annotations

import gc
import numpy as np
import pandas as pd

from data_preprocessing import RawTables

# Columns used for frequency encoding (train+test joint counts; no target leakage).
_FREQUENCY_ENCODE_CANDIDATES = (
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_TYPE_SUITE",
    "CODE_GENDER",
    "WALLSMATERIAL_MODE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "EMERGENCYSTATE_MODE",
    "NAME_CONTRACT_TYPE",
)


def _sanitize_col_suffix(name: str, max_len: int = 32) -> str:
    s = "".join(c if c.isalnum() else "_" for c in str(name))
    return s[:max_len].strip("_") or "X"


def add_frequency_encoding_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> None:
    """Map categoricals to global frequency (train+test). Improves tree splits on rare groups."""
    cols = [c for c in _FREQUENCY_ENCODE_CANDIDATES if c in app_train.columns and c in app_test.columns]
    for col in cols:
        vc = pd.concat([app_train[col], app_test[col]], ignore_index=True).astype(str).value_counts()
        app_train[f"{col}_FREQ"] = app_train[col].astype(str).map(vc).fillna(0).astype(np.float64)
        app_test[f"{col}_FREQ"] = app_test[col].astype(str).map(vc).fillna(0).astype(np.float64)


def add_ext_source_rank_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> None:
    """Percentile ranks of external scores (train+test joint); scale-free, no label leakage."""
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if col not in app_train.columns or col not in app_test.columns:
            continue
        comb = pd.concat([app_train[col], app_test[col]], ignore_index=True)
        r = comb.rank(pct=True, method="average").to_numpy()
        n_tr = len(app_train)
        app_train[f"{col}_PCTRANK"] = r[:n_tr].astype(np.float64)
        app_test[f"{col}_PCTRANK"] = r[n_tr:].astype(np.float64)


def add_group_relative_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> None:
    """Leakage-safe group-relative monetary features from train+test pooled statistics."""
    group_cols = ("ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_EDUCATION_TYPE")
    value_cols = ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY")
    for g in group_cols:
        if g not in app_train.columns or g not in app_test.columns:
            continue
        for v in value_cols:
            if v not in app_train.columns or v not in app_test.columns:
                continue
            comb = pd.concat([app_train[[g, v]], app_test[[g, v]]], ignore_index=True)
            med = comb.groupby(g)[v].median()
            tr_med = app_train[g].map(med).fillna(comb[v].median())
            te_med = app_test[g].map(med).fillna(comb[v].median())
            app_train[f"{v}_BY_{g}_MED_RATIO"] = app_train[v] / (tr_med + 1.0)
            app_test[f"{v}_BY_{g}_MED_RATIO"] = app_test[v] / (te_med + 1.0)


def add_missingness_profile_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> None:
    """Robust null-structure features; useful for Home Credit data quality signals."""
    all_cols = [c for c in app_train.columns if c in app_test.columns]
    tr_null = app_train[all_cols].isnull()
    te_null = app_test[all_cols].isnull()
    app_train["APP_MISSING_COUNT"] = tr_null.sum(axis=1).astype(np.float64)
    app_test["APP_MISSING_COUNT"] = te_null.sum(axis=1).astype(np.float64)
    denom = float(len(all_cols)) if all_cols else 1.0
    app_train["APP_MISSING_RATIO"] = app_train["APP_MISSING_COUNT"] / denom
    app_test["APP_MISSING_RATIO"] = app_test["APP_MISSING_COUNT"] / denom

    key_cols = [c for c in all_cols if c.startswith("EXT_SOURCE_") or "AMT_REQ_CREDIT_BUREAU" in c]
    if key_cols:
        app_train["KEY_MISSING_COUNT"] = app_train[key_cols].isnull().sum(axis=1).astype(np.float64)
        app_test["KEY_MISSING_COUNT"] = app_test[key_cols].isnull().sum(axis=1).astype(np.float64)


def _defragment(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce pandas block fragmentation after many column inserts."""
    return df.copy()


def create_level2_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_before = df.shape[1]

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
    df["GOODS_PRICE_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
    df["ID_PUBLISH_YEARS"] = -df["DAYS_ID_PUBLISH"] / 365.25

    df["IS_RETIRED"] = (df["EMPLOYMENT_YEARS"] > 100).astype(int)
    df["EMPLOYMENT_YEARS"] = df["EMPLOYMENT_YEARS"].clip(upper=50)
    df["EMP_BIRTH_RATIO"] = df["EMPLOYMENT_YEARS"] / (df["AGE_YEARS"] + 1)
    df["WORK_START_AGE"] = df["AGE_YEARS"] - df["EMPLOYMENT_YEARS"]

    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_MAX"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
    df["EXT_SOURCE_MIN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    df["EXT_SOURCE_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_MISSING_COUNT"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isnull().sum(axis=1)

    doc_cols = [col for col in df.columns if "FLAG_DOCUMENT" in col]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    df["PHONE_REACHABLE"] = (
        df["FLAG_MOBIL"] + df["FLAG_EMP_PHONE"] + df["FLAG_WORK_PHONE"] + df["FLAG_PHONE"]
    ).astype(float)

    for col in df.columns:
        if "AMT_REQ_CREDIT_BUREAU" in col:
            df[col] = df[col].fillna(0)
    df["TOTAL_ENQUIRIES"] = df[[c for c in df.columns if "AMT_REQ_CREDIT_BUREAU" in c]].sum(axis=1)

    df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)
    df["HAS_CAR"] = df["FLAG_OWN_CAR"].map({"Y": 1, "N": 0})
    df["HAS_REALTY"] = df["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0})

    df["EXT_SOURCE_1_2"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
    df["EXT_SOURCE_1_3"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_2_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_WEIGHTED"] = df["EXT_SOURCE_1"] * 0.3 + df["EXT_SOURCE_2"] * 0.4 + df["EXT_SOURCE_3"] * 0.3

    df["EXT_SOURCE_2_SQ"] = df["EXT_SOURCE_2"] ** 2
    df["EXT_SOURCE_2_CUBE"] = df["EXT_SOURCE_2"] ** 3
    df["EXT_SOURCE_2_SQRT"] = np.sqrt(df["EXT_SOURCE_2"].fillna(0))

    df["AGE_INCOME"] = df["AGE_YEARS"] * df["AMT_INCOME_TOTAL"]
    df["AGE_CREDIT"] = df["AGE_YEARS"] * df["AMT_CREDIT"]
    df["AGE_EMPLOYED"] = df["AGE_YEARS"] * df["EMPLOYMENT_YEARS"]

    df["CREDIT_EXT_SOURCE_2"] = df["AMT_CREDIT"] * df["EXT_SOURCE_2"].fillna(0)
    df["INCOME_EXT_SOURCE_2"] = df["AMT_INCOME_TOTAL"] * df["EXT_SOURCE_2"].fillna(0)

    df["ANNUITY_INCOME_EMPLOYED"] = df["ANNUITY_INCOME_RATIO"] * df["EMPLOYMENT_YEARS"]
    df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["CREDIT_GOODS_RATIO_ADJ"] = df["CREDIT_GOODS_DIFF"] / (df["AMT_INCOME_TOTAL"] + 1)

    df["INCOME_STABILITY"] = df["AMT_INCOME_TOTAL"] * df["EMPLOYMENT_YEARS"] / (df["AGE_YEARS"] + 1)
    df["CREDIT_BURDEN"] = (df["AMT_CREDIT"] + df["AMT_ANNUITY"]) / (df["AMT_INCOME_TOTAL"] + 1)

    df["RISK_SCORE_1"] = (
        df["CREDIT_INCOME_RATIO"] * 0.3
        + df["ANNUITY_INCOME_RATIO"] * 0.3
        + (1 - df["EXT_SOURCE_MEAN"].fillna(0.5)) * 0.4
    )

    df["RISK_SCORE_2"] = (
        (df["AGE_YEARS"] < 30).astype(int) * 0.2
        + (df["EMPLOYMENT_YEARS"] < 2).astype(int) * 0.3
        + (df["TOTAL_ENQUIRIES"] > 3).astype(int) * 0.5
    )

    df["INCOME_BIN"] = pd.cut(df["AMT_INCOME_TOTAL"], bins=10, labels=False, duplicates="drop")
    df["CREDIT_BIN"] = pd.cut(df["AMT_CREDIT"], bins=10, labels=False, duplicates="drop")
    df["AGE_BIN"] = pd.cut(df["AGE_YEARS"], bins=[0, 25, 35, 45, 55, 100], labels=False)

    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        dlp = df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan)
        df["PHONE_CHANGE_YEARS"] = (-dlp.clip(upper=0)) / 365.25
        df["PHONE_CHANGED_LT_90D"] = (df["DAYS_LAST_PHONE_CHANGE"] > -90).astype(int)
        df["PHONE_CHANGED_LT_180D"] = (df["DAYS_LAST_PHONE_CHANGE"] > -180).astype(int)

    if "HOUR_APPR_PROCESS_START" in df.columns:
        h = df["HOUR_APPR_PROCESS_START"].fillna(12).astype(float) % 24
        df["HOUR_APPR_SIN"] = np.sin(2 * np.pi * h / 24.0)
        df["HOUR_APPR_COS"] = np.cos(2 * np.pi * h / 24.0)

    if "REGION_RATING_CLIENT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["REGION_RATING_X_INCOME"] = df["REGION_RATING_CLIENT"].fillna(3) * df["AMT_INCOME_TOTAL"]
    if "REGION_RATING_CLIENT_W_CITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["REGION_RATING_WCITY_X_CREDIT"] = df["REGION_RATING_CLIENT_W_CITY"].fillna(3) * df["AMT_CREDIT"]

    if "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns and "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns:
        df["SOCIAL_DEF_RATE_30"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
    if "OBS_60_CNT_SOCIAL_CIRCLE" in df.columns and "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns:
        df["SOCIAL_DEF_RATE_60"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] / (df["OBS_60_CNT_SOCIAL_CIRCLE"] + 1)

    if "AMT_REQ_CREDIT_BUREAU_TOTAL" not in df.columns and "TOTAL_ENQUIRIES" in df.columns:
        df["ENQUIRIES_PER_YEAR"] = df["TOTAL_ENQUIRIES"] / (df["AGE_YEARS"] + 0.1)

    if "REGION_POPULATION_RELATIVE" in df.columns:
        df["REGION_POP_X_CREDIT"] = df["REGION_POPULATION_RELATIVE"].fillna(0) * df["AMT_CREDIT"]
        df["REGION_POP_X_INCOME"] = df["REGION_POPULATION_RELATIVE"].fillna(0) * df["AMT_INCOME_TOTAL"]

    if "LIVINGAREA_AVG" in df.columns and "NONLIVINGAREA_AVG" in df.columns:
        df["TOTAL_LIVING_AREA"] = df["LIVINGAREA_AVG"].fillna(0) + df["NONLIVINGAREA_AVG"].fillna(0)

    df = _defragment(df)
    print(f"Created {df.shape[1] - n_before} Level 2 application features")
    return df


def aggregate_bureau_advanced(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
        {
            "MONTHS_BALANCE": ["min", "max", "mean", "size"],
            "STATUS": [
                lambda x: (x == "C").sum(),
                lambda x: (x == "0").sum(),
                lambda x: (x.isin(["1", "2", "3", "4", "5"])).sum(),
            ],
        }
    )
    bb_agg.columns = ["BB_" + "_".join(col).upper() for col in bb_agg.columns]
    bb_agg.reset_index(inplace=True)
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    bureau["IS_RECENT"] = (bureau["DAYS_CREDIT"] > -365).astype(int)

    agg_dict = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "CREDIT_DAY_OVERDUE": ["max", "mean", "sum"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "mean", "sum"],
        "DAYS_CREDIT_UPDATE": ["min", "max", "mean"],
        "AMT_ANNUITY": ["max", "mean"],
    }

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(agg_dict)
    bureau_agg.columns = ["BUREAU_" + "_".join(col).upper() for col in bureau_agg.columns]

    bureau_recent = bureau[bureau["IS_RECENT"] == 1].groupby("SK_ID_CURR").agg(
        {"AMT_CREDIT_SUM": ["mean", "sum"], "AMT_CREDIT_SUM_DEBT": ["mean", "sum"], "CREDIT_DAY_OVERDUE": ["max", "mean"]}
    )
    bureau_recent.columns = ["BUREAU_RECENT_" + "_".join(col).upper() for col in bureau_recent.columns]

    bureau_counts = pd.DataFrame()
    bureau_counts["BUREAU_COUNT"] = bureau.groupby("SK_ID_CURR").size()
    bureau_counts["BUREAU_ACTIVE_COUNT"] = bureau[bureau["CREDIT_ACTIVE"] == "Active"].groupby("SK_ID_CURR").size()
    bureau_counts["BUREAU_CLOSED_COUNT"] = bureau[bureau["CREDIT_ACTIVE"] == "Closed"].groupby("SK_ID_CURR").size()
    bureau_counts["BUREAU_RECENT_COUNT"] = bureau[bureau["IS_RECENT"] == 1].groupby("SK_ID_CURR").size()
    bureau_counts["BUREAU_CREDIT_TYPES"] = bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique()

    bureau_agg = bureau_agg.join(bureau_recent, how="left")
    bureau_agg = bureau_agg.join(bureau_counts, how="left")
    bureau_agg = bureau_agg.fillna(0)

    bureau_agg["BUREAU_CREDIT_UTILIZATION"] = bureau_agg["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] / (
        bureau_agg["BUREAU_AMT_CREDIT_SUM_SUM"] + 1
    )
    bureau_agg["BUREAU_ACTIVE_RATIO"] = bureau_agg["BUREAU_ACTIVE_COUNT"] / (bureau_agg["BUREAU_COUNT"] + 1)
    bureau_agg["BUREAU_RECENT_RATIO"] = bureau_agg["BUREAU_RECENT_COUNT"] / (bureau_agg["BUREAU_COUNT"] + 1)
    bureau_agg["BUREAU_AVG_DEBT_PER_CREDIT"] = bureau_agg["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] / (
        bureau_agg["BUREAU_COUNT"] + 1
    )

    if "DAYS_CREDIT" in bureau.columns and "DAYS_CREDIT_ENDDATE" in bureau.columns:
        bureau["CREDIT_SPAN_EST"] = (bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]).replace([np.inf, -np.inf], np.nan)
        span_agg = bureau.groupby("SK_ID_CURR")["CREDIT_SPAN_EST"].agg(["mean", "max", "sum"])
        span_agg.columns = ["BUREAU_CREDIT_SPAN_" + c.upper() for c in span_agg.columns]
        bureau_agg = bureau_agg.join(span_agg, how="left").fillna(0)

    overdue_lines = bureau[bureau["CREDIT_DAY_OVERDUE"] > 0].groupby("SK_ID_CURR").size()
    bureau_agg["BUREAU_LINES_WITH_OVERDUE"] = overdue_lines.reindex(bureau_agg.index, fill_value=0).fillna(0)

    if "CREDIT_ACTIVE" in bureau.columns:
        sold = bureau[bureau["CREDIT_ACTIVE"] == "Sold"].groupby("SK_ID_CURR").size()
        bureau_agg["BUREAU_SOLD_COUNT"] = sold.reindex(bureau_agg.index, fill_value=0).fillna(0)

    bureau["DEBT_OVER_SUM"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1.0)
    debt_ratio = bureau.groupby("SK_ID_CURR")["DEBT_OVER_SUM"].agg(["mean", "max"])
    debt_ratio.columns = ["BUREAU_DEBT_RATIO_MEAN", "BUREAU_DEBT_RATIO_MAX"]
    bureau_agg = bureau_agg.join(debt_ratio, how="left").fillna(0)

    top_types = bureau["CREDIT_TYPE"].value_counts().head(10).index
    for ct in top_types:
        suf = _sanitize_col_suffix(ct)
        ct_count = bureau[bureau["CREDIT_TYPE"] == ct].groupby("SK_ID_CURR").size()
        bureau_agg[f"BUREAU_CT_CNT_{suf}"] = ct_count.reindex(bureau_agg.index, fill_value=0).fillna(0)

    del bureau, bb_agg, bureau_recent
    gc.collect()
    return bureau_agg


def aggregate_bureau_balance_by_client(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """Client-level bureau_balance stats (distinct from per-bureau-line aggregates)."""
    key = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
    bb = bureau_balance.merge(key, on="SK_ID_BUREAU", how="left")
    bb = bb.dropna(subset=["SK_ID_CURR"])
    bb["SK_ID_CURR"] = bb["SK_ID_CURR"].astype(int)
    bb["STATUS_BAD"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(np.int8)
    bb["STATUS_GOOD_CLOSED"] = (bb["STATUS"] == "C").astype(np.int8)
    bb["STATUS_ACTIVE"] = (bb["STATUS"] == "0").astype(np.int8)

    agg_dict = {
        "MONTHS_BALANCE": ["min", "max", "mean", "std"],
        "STATUS_BAD": ["sum", "mean"],
        "STATUS_GOOD_CLOSED": ["sum", "mean"],
        "STATUS_ACTIVE": ["sum", "mean"],
    }
    bb_agg = bb.groupby("SK_ID_CURR").agg(agg_dict)
    bb_agg.columns = ["BBCURR_" + "_".join(col).upper() for col in bb_agg.columns]
    n_rows = bb.groupby("SK_ID_CURR").size().rename("BBCURR_N_STATEMENT_LINES")
    bb_agg = bb_agg.join(n_rows, how="left").fillna(0)
    bb_agg["BBCURR_BAD_SHARE"] = bb_agg["BBCURR_STATUS_BAD_SUM"] / (bb_agg["BBCURR_N_STATEMENT_LINES"] + 1)

    recent_win = bb["MONTHS_BALANCE"] >= -6
    if recent_win.any():
        sub = bb.loc[recent_win]
        bad_recent = sub["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(np.int32).groupby(sub["SK_ID_CURR"]).sum()
        bb_agg["BBCURR_LAST6M_BAD_LINES"] = bad_recent.reindex(bb_agg.index, fill_value=0).fillna(0)
        n_recent = sub.groupby("SK_ID_CURR").size()
        bb_agg["BBCURR_LAST6M_LINE_COUNT"] = n_recent.reindex(bb_agg.index, fill_value=0).fillna(0)
        bb_agg["BBCURR_LAST6M_BAD_RATE"] = bb_agg["BBCURR_LAST6M_BAD_LINES"] / (bb_agg["BBCURR_LAST6M_LINE_COUNT"] + 1)

    del bb, key
    gc.collect()
    return bb_agg


def _prev_first_last_application_features(prev: pd.DataFrame) -> pd.DataFrame:
    """Most recent vs earliest previous-application row per client (sorted by DAYS_DECISION)."""
    if "DAYS_DECISION" not in prev.columns:
        return pd.DataFrame()
    p = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"])
    g = p.groupby("SK_ID_CURR", sort=False)
    first = g.first()
    last = g.last()
    out = pd.DataFrame(index=first.index)
    for col in ("AMT_CREDIT", "AMT_APPLICATION", "AMT_ANNUITY", "CNT_PAYMENT", "DAYS_DECISION"):
        if col in first.columns:
            out[f"PFIRST_{col}"] = first[col].values
        if col in last.columns:
            out[f"PLAST_{col}"] = last[col].values
    if "DAYS_DECISION" in first.columns and "DAYS_DECISION" in last.columns:
        out["PREV_DECISION_SPAN"] = (last["DAYS_DECISION"] - first["DAYS_DECISION"]).values
    if "NAME_CONTRACT_STATUS" in last.columns:
        st = last["NAME_CONTRACT_STATUS"]
        out["PLAST_WAS_REFUSED"] = (st == "Refused").fillna(False).astype(np.int8).values
        out["PLAST_WAS_APPROVED"] = (st == "Approved").fillna(False).astype(np.int8).values
    if "AMT_CREDIT" in first.columns and "AMT_CREDIT" in last.columns:
        out["PREV_FIRST_LAST_CREDIT_DELTA"] = (last["AMT_CREDIT"] - first["AMT_CREDIT"]).values
    return out


def aggregate_previous_apps_advanced(prev: pd.DataFrame) -> pd.DataFrame:
    prev = prev.copy()
    prev_order_feats = _prev_first_last_application_features(prev)
    prev["IS_RECENT"] = (prev["DAYS_DECISION"] > -730).astype(int)
    prev["IS_VERY_RECENT"] = (prev["DAYS_DECISION"] > -365).astype(int)

    if "AMT_CREDIT" in prev.columns and "AMT_GOODS_PRICE" in prev.columns:
        prev["GOODS_OVER_CREDIT"] = prev["AMT_GOODS_PRICE"] / (prev["AMT_CREDIT"] + 1.0)

    agg_dict = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean", "sum"],
        "AMT_CREDIT": ["min", "max", "mean", "sum"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean", "sum"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum", "max"],
        "DAYS_FIRST_DRAWING": ["min", "max", "mean"],
        "DAYS_FIRST_DUE": ["min", "max", "mean"],
        "DAYS_LAST_DUE": ["min", "max", "mean"],
        "DAYS_TERMINATION": ["min", "max", "mean"],
    }
    if "GOODS_OVER_CREDIT" in prev.columns:
        agg_dict["GOODS_OVER_CREDIT"] = ["mean", "max", "min"]

    prev_agg = prev.groupby("SK_ID_CURR").agg(agg_dict)
    prev_agg.columns = ["PREV_" + "_".join(col).upper() for col in prev_agg.columns]

    prev_recent = prev[prev["IS_RECENT"] == 1].groupby("SK_ID_CURR").agg(
        {"AMT_CREDIT": ["mean", "sum"], "AMT_APPLICATION": ["mean", "sum"]}
    )
    prev_recent.columns = ["PREV_RECENT_" + "_".join(col).upper() for col in prev_recent.columns]

    prev_counts = pd.DataFrame()
    prev_counts["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size()
    prev_counts["PREV_APPROVED_COUNT"] = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].groupby("SK_ID_CURR").size()
    prev_counts["PREV_REFUSED_COUNT"] = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].groupby("SK_ID_CURR").size()
    prev_counts["PREV_CANCELED_COUNT"] = prev[prev["NAME_CONTRACT_STATUS"] == "Canceled"].groupby("SK_ID_CURR").size()
    prev_counts["PREV_RECENT_COUNT"] = prev[prev["IS_RECENT"] == 1].groupby("SK_ID_CURR").size()
    prev_counts["PREV_VERY_RECENT_COUNT"] = prev[prev["IS_VERY_RECENT"] == 1].groupby("SK_ID_CURR").size()

    unused = prev[prev["NAME_CONTRACT_STATUS"] == "Unused offer"].groupby("SK_ID_CURR").size()
    prev_counts["PREV_UNUSED_OFFER_COUNT"] = unused.reindex(prev_counts.index, fill_value=0).fillna(0)

    if "NAME_CONTRACT_TYPE" in prev.columns:
        prev_counts["PREV_CONTRACT_TYPE_NUNIQUE"] = prev.groupby("SK_ID_CURR")["NAME_CONTRACT_TYPE"].nunique()
    if "PRODUCT_COMBINATION" in prev.columns:
        prev_counts["PREV_PRODUCT_COMBO_NUNIQUE"] = prev.groupby("SK_ID_CURR")["PRODUCT_COMBINATION"].nunique()
    if "NAME_YIELD_GROUP" in prev.columns:
        prev_counts["PREV_YIELD_GROUP_NUNIQUE"] = prev.groupby("SK_ID_CURR")["NAME_YIELD_GROUP"].nunique()

    if "FLAG_LAST_APPL_PER_CONTRACT" in prev.columns:
        last_y = (prev["FLAG_LAST_APPL_PER_CONTRACT"] == "Y").groupby(prev["SK_ID_CURR"]).sum().rename(
            "PREV_LAST_APPL_Y_COUNT"
        )
        prev_counts = prev_counts.join(last_y, how="left")

    for col in ("RATE_DOWN_PAYMENT", "RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED"):
        if col in prev.columns:
            g = prev.groupby("SK_ID_CURR")[col].agg(["mean", "max", "min"])
            g.columns = [f"PREV_{col.upper()}_{x.upper()}" for x in g.columns]
            prev_counts = prev_counts.join(g, how="left")

    top_contracts = (
        prev["NAME_CONTRACT_TYPE"].value_counts().head(8).index if "NAME_CONTRACT_TYPE" in prev.columns else []
    )
    for ct in top_contracts:
        suf = _sanitize_col_suffix(ct)
        c = prev[prev["NAME_CONTRACT_TYPE"] == ct].groupby("SK_ID_CURR").size()
        prev_counts[f"PREV_NTYPE_{suf}"] = c.reindex(prev_counts.index, fill_value=0).fillna(0)

    prev_counts = prev_counts.fillna(0)

    prev_agg = prev_agg.join(prev_recent, how="left")
    prev_agg = prev_agg.join(prev_counts, how="left")
    if len(prev_order_feats) > 0:
        prev_agg = prev_agg.join(prev_order_feats, how="left")
    prev_agg = prev_agg.fillna(0)

    prev_agg["PREV_APPROVAL_RATE"] = prev_agg["PREV_APPROVED_COUNT"] / (prev_agg["PREV_COUNT"] + 1)
    prev_agg["PREV_REFUSAL_RATE"] = prev_agg["PREV_REFUSED_COUNT"] / (prev_agg["PREV_COUNT"] + 1)
    prev_agg["PREV_APP_CREDIT_RATIO"] = prev_agg["PREV_AMT_APPLICATION_MEAN"] / (prev_agg["PREV_AMT_CREDIT_MEAN"] + 1)
    prev_agg["PREV_RECENT_RATIO"] = prev_agg["PREV_RECENT_COUNT"] / (prev_agg["PREV_COUNT"] + 1)
    if "PREV_UNUSED_OFFER_COUNT" in prev_agg.columns:
        prev_agg["PREV_UNUSED_RATE"] = prev_agg["PREV_UNUSED_OFFER_COUNT"] / (prev_agg["PREV_COUNT"] + 1)
    else:
        prev_agg["PREV_UNUSED_RATE"] = 0.0
    if "PREV_VERY_RECENT_COUNT" in prev_agg.columns:
        prev_agg["PREV_VERY_RECENT_RATIO"] = prev_agg["PREV_VERY_RECENT_COUNT"] / (prev_agg["PREV_COUNT"] + 1)
    else:
        prev_agg["PREV_VERY_RECENT_RATIO"] = 0.0

    del prev, prev_recent
    gc.collect()
    return prev_agg


def aggregate_installments_advanced(inst: pd.DataFrame) -> pd.DataFrame:
    inst = inst.copy()
    inst["PAYMENT_DIFF"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + 1)
    inst["DAYS_LATE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["IS_LATE"] = (inst["DAYS_LATE"] > 0).astype(int)
    inst["IS_UNDERPAID"] = (inst["PAYMENT_DIFF"] < -0.01).astype(int)
    inst["IS_OVERPAID"] = (inst["PAYMENT_DIFF"] > 0.01).astype(int)
    inst["IS_RECENT"] = (inst["DAYS_INSTALMENT"] > -365).astype(int)

    agg_dict = {
        "NUM_INSTALMENT_NUMBER": ["max", "mean", "min"],
        "AMT_INSTALMENT": ["min", "max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "PAYMENT_DIFF": ["min", "max", "mean", "sum", "std"],
        "PAYMENT_RATIO": ["min", "max", "mean", "std"],
        "DAYS_LATE": ["max", "mean", "std", "sum"],
        "IS_LATE": ["sum", "mean"],
        "IS_UNDERPAID": ["sum", "mean"],
        "IS_OVERPAID": ["sum", "mean"],
    }
    if "DAYS_ENTRY_PAYMENT" in inst.columns:
        agg_dict["DAYS_ENTRY_PAYMENT"] = ["min", "max", "mean", "std"]
    if "NUM_INSTALMENT_VERSION" in inst.columns:
        agg_dict["NUM_INSTALMENT_VERSION"] = ["max", "mean", "nunique"]

    inst_agg = inst.groupby("SK_ID_CURR").agg(agg_dict)
    inst_agg.columns = ["INST_" + "_".join(col).upper() for col in inst_agg.columns]

    inst_recent = inst[inst["IS_RECENT"] == 1].groupby("SK_ID_CURR").agg(
        {"IS_LATE": ["sum", "mean"], "PAYMENT_DIFF": ["mean", "min"]}
    )
    inst_recent.columns = ["INST_RECENT_" + "_".join(col).upper() for col in inst_recent.columns]

    inst_agg = inst_agg.join(inst_recent, how="left")
    inst_agg = inst_agg.fillna(0)

    inst_agg["INST_LATE_RATE"] = inst_agg["INST_IS_LATE_SUM"] / (inst_agg["INST_NUM_INSTALMENT_NUMBER_MAX"] + 1)
    inst_agg["INST_PAYMENT_CONSISTENCY"] = 1 / (inst_agg["INST_PAYMENT_RATIO_STD"] + 0.01)
    inst_agg["INST_UNDERPAID_RATE"] = inst_agg["INST_IS_UNDERPAID_SUM"] / (
        inst_agg["INST_NUM_INSTALMENT_NUMBER_MAX"] + 1
    )
    if "SK_ID_PREV" in inst.columns:
        late_prev = inst.loc[inst["IS_LATE"] == 1].groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
        inst_agg["INST_N_PREV_WITH_LATE"] = late_prev.reindex(inst_agg.index, fill_value=0).fillna(0)
    inst_agg.replace([np.inf, -np.inf], 0, inplace=True)

    del inst, inst_recent
    gc.collect()
    return inst_agg


def aggregate_credit_card_advanced(cc: pd.DataFrame) -> pd.DataFrame:
    cc = cc.copy()
    cc["BALANCE_LIMIT_RATIO"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["MIN_PAYMENT_RATIO"] = cc["AMT_PAYMENT_CURRENT"] / (cc["AMT_INST_MIN_REGULARITY"] + 1)
    cc["DRAWINGS_RATIO"] = cc["AMT_DRAWINGS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["ATM_DRAWINGS_RATIO"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["POS_DRAWINGS_RATIO"] = cc["AMT_DRAWINGS_POS_CURRENT"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)

    agg_dict = {
        "AMT_BALANCE": ["min", "max", "mean", "sum", "std"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["min", "max", "mean"],
        "AMT_DRAWINGS_CURRENT": ["max", "mean", "sum"],
        "AMT_PAYMENT_CURRENT": ["min", "max", "mean", "sum"],
        "CNT_DRAWINGS_CURRENT": ["max", "mean", "sum"],
        "SK_DPD": ["max", "mean", "sum"],
        "SK_DPD_DEF": ["max", "mean", "sum"],
        "BALANCE_LIMIT_RATIO": ["min", "max", "mean", "std"],
        "MIN_PAYMENT_RATIO": ["min", "max", "mean"],
        "DRAWINGS_RATIO": ["max", "mean"],
        "ATM_DRAWINGS_RATIO": ["max", "mean"],
        "POS_DRAWINGS_RATIO": ["max", "mean"],
    }

    cc_agg = cc.groupby("SK_ID_CURR").agg(agg_dict)
    cc_agg.columns = ["CC_" + "_".join(col).upper() for col in cc_agg.columns]

    cc_counts = cc.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique().to_frame("CC_COUNT")
    cc_agg = cc_agg.join(cc_counts, how="left")

    high_util = cc[cc["BALANCE_LIMIT_RATIO"] > 0.8].groupby("SK_ID_CURR").size()
    cc_agg["CC_HIGH_UTIL_COUNT"] = high_util.reindex(cc_agg.index, fill_value=0).fillna(0)

    dpd_pos = cc[(cc["SK_DPD"] > 0) | (cc["SK_DPD_DEF"] > 0)].groupby("SK_ID_CURR").size()
    cc_agg["CC_ROWS_WITH_DPD"] = dpd_pos.reindex(cc_agg.index, fill_value=0).fillna(0)

    cc_agg = cc_agg.fillna(0)

    del cc, cc_counts
    gc.collect()
    return cc_agg


def aggregate_pos_cash_advanced(pos: pd.DataFrame) -> pd.DataFrame:
    pos = pos.copy()
    pos["HAS_DPD"] = ((pos["SK_DPD"] > 0) | (pos["SK_DPD_DEF"] > 0)).astype(int)

    agg_dict = {
        "MONTHS_BALANCE": ["max", "mean", "min", "size"],
        "CNT_INSTALMENT": ["max", "mean", "sum"],
        "CNT_INSTALMENT_FUTURE": ["max", "mean", "sum"],
        "SK_DPD": ["max", "mean", "sum"],
        "SK_DPD_DEF": ["max", "mean", "sum"],
        "HAS_DPD": ["sum", "mean"],
    }

    pos_agg = pos.groupby("SK_ID_CURR").agg(agg_dict)
    pos_agg.columns = ["POS_" + "_".join(col).upper() for col in pos_agg.columns]

    if "NAME_CONTRACT_STATUS" in pos.columns:
        active_rows = pos[pos["NAME_CONTRACT_STATUS"] == "Active"].groupby("SK_ID_CURR").size()
        pos_agg["POS_ACTIVE_ROW_COUNT"] = active_rows.reindex(pos_agg.index, fill_value=0).fillna(0)

    n_prev = pos.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
    pos_agg["POS_N_PREV_CONTRACTS"] = n_prev.reindex(pos_agg.index, fill_value=0).fillna(0)

    pos_agg["POS_COMPLETION_RATIO"] = (
        pos_agg["POS_CNT_INSTALMENT_MEAN"] - pos_agg["POS_CNT_INSTALMENT_FUTURE_MEAN"]
    ) / (pos_agg["POS_CNT_INSTALMENT_MEAN"] + 1)
    pos_agg["POS_FUTURE_OVER_TOTAL"] = pos_agg["POS_CNT_INSTALMENT_FUTURE_SUM"] / (
        pos_agg["POS_CNT_INSTALMENT_SUM"] + 1
    )
    pos_agg.replace([np.inf, -np.inf], 0, inplace=True)

    del pos
    gc.collect()
    return pos_agg


def create_cross_table_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "BUREAU_AMT_CREDIT_SUM_SUM" in df.columns:
        df["BUREAU_APP_CREDIT_RATIO"] = df["BUREAU_AMT_CREDIT_SUM_SUM"] / (df["AMT_CREDIT"] + 1)
        df["BUREAU_APP_INCOME_RATIO"] = df["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] / (df["AMT_INCOME_TOTAL"] + 1)

    if "PREV_AMT_CREDIT_SUM" in df.columns:
        df["PREV_APP_CREDIT_RATIO"] = df["PREV_AMT_CREDIT_SUM"] / (df["AMT_CREDIT"] + 1)

    if "CC_AMT_BALANCE_SUM" in df.columns:
        df["CC_APP_INCOME_RATIO"] = df["CC_AMT_BALANCE_SUM"] / (df["AMT_INCOME_TOTAL"] + 1)

    debt_cols = []
    if "BUREAU_AMT_CREDIT_SUM_DEBT_SUM" in df.columns:
        debt_cols.append("BUREAU_AMT_CREDIT_SUM_DEBT_SUM")
    if "CC_AMT_BALANCE_SUM" in df.columns:
        debt_cols.append("CC_AMT_BALANCE_SUM")

    if debt_cols:
        df["TOTAL_DEBT"] = df[debt_cols].sum(axis=1)
        df["TOTAL_DEBT_INCOME_RATIO"] = df["TOTAL_DEBT"] / (df["AMT_INCOME_TOTAL"] + 1)

    if "PREV_REFUSED_COUNT" in df.columns and "PREV_COUNT" in df.columns:
        df["PREV_REFUSED_X_INCOME"] = df["PREV_REFUSED_COUNT"] * df["AMT_INCOME_TOTAL"] / (
            df["PREV_COUNT"] + 1
        )

    if "INST_IS_LATE_SUM" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["INST_LATE_SUM_OVER_INCOME"] = df["INST_IS_LATE_SUM"] / (df["AMT_INCOME_TOTAL"] + 1)

    if "BBCURR_STATUS_BAD_SUM" in df.columns and "BUREAU_COUNT" in df.columns:
        df["BBCURR_BAD_PER_BUREAU_LINE"] = df["BBCURR_STATUS_BAD_SUM"] / (df["BUREAU_COUNT"] + 1)

    if "PREV_AMT_CREDIT_SUM" in df.columns and "BUREAU_AMT_CREDIT_SUM_SUM" in df.columns:
        df["PREV_OVER_BUREAU_CREDIT"] = df["PREV_AMT_CREDIT_SUM"] / (df["BUREAU_AMT_CREDIT_SUM_SUM"] + 1)

    if "EXT_SOURCE_MEAN" in df.columns and "BUREAU_CREDIT_UTILIZATION" in df.columns:
        df["EXT_MEAN_X_BUREAU_UTIL"] = df["EXT_SOURCE_MEAN"].fillna(0.5) * df["BUREAU_CREDIT_UTILIZATION"]

    if "EXT_SOURCE_2" in df.columns and "BUREAU_DEBT_RATIO_MEAN" in df.columns:
        df["EXT2_X_BUREAU_DEBT_RATIO"] = df["EXT_SOURCE_2"].fillna(0) * df["BUREAU_DEBT_RATIO_MEAN"]

    if "PLAST_AMT_CREDIT" in df.columns and "AMT_CREDIT" in df.columns:
        df["LAST_PREV_CREDIT_OVER_APP"] = df["PLAST_AMT_CREDIT"] / (df["AMT_CREDIT"] + 1)

    if "INST_N_PREV_WITH_LATE" in df.columns and "PREV_COUNT" in df.columns:
        df["LATE_PREV_CONTRACTS_OVER_HIST"] = df["INST_N_PREV_WITH_LATE"] / (df["PREV_COUNT"] + 1)

    return df


def merge_external_features(
    app: pd.DataFrame,
    bureau_feats: pd.DataFrame,
    prev_feats: pd.DataFrame,
    inst_feats: pd.DataFrame,
    cc_feats: pd.DataFrame,
    pos_feats: pd.DataFrame,
    bureau_balance_client_feats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = app.merge(bureau_feats, left_on="SK_ID_CURR", right_index=True, how="left")
    df = df.merge(prev_feats, left_on="SK_ID_CURR", right_index=True, how="left")
    df = df.merge(inst_feats, left_on="SK_ID_CURR", right_index=True, how="left")
    df = df.merge(cc_feats, left_on="SK_ID_CURR", right_index=True, how="left")
    df = df.merge(pos_feats, left_on="SK_ID_CURR", right_index=True, how="left")
    if bureau_balance_client_feats is not None and len(bureau_balance_client_feats) > 0:
        df = df.merge(
            bureau_balance_client_feats,
            left_on="SK_ID_CURR",
            right_index=True,
            how="left",
        )
    return df


def build_enriched_train_test(tables: RawTables) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full Level-2 feature pipeline: application features, table aggregates, merges, cross-table.
    """
    app_train = create_level2_application_features(tables.application_train.copy())
    app_test = create_level2_application_features(tables.application_test.copy())
    add_frequency_encoding_features(app_train, app_test)
    add_ext_source_rank_features(app_train, app_test)
    add_group_relative_features(app_train, app_test)
    add_missingness_profile_features(app_train, app_test)

    bureau_feats = aggregate_bureau_advanced(tables.bureau.copy(), tables.bureau_balance.copy())
    bureau_bb_curr = aggregate_bureau_balance_by_client(tables.bureau.copy(), tables.bureau_balance.copy())
    prev_feats = aggregate_previous_apps_advanced(tables.previous_application.copy())
    inst_feats = aggregate_installments_advanced(tables.installments_payments.copy())
    cc_feats = aggregate_credit_card_advanced(tables.credit_card_balance.copy())
    pos_feats = aggregate_pos_cash_advanced(tables.pos_cash_balance.copy())

    app_train = merge_external_features(
        app_train, bureau_feats, prev_feats, inst_feats, cc_feats, pos_feats, bureau_bb_curr
    )
    app_test = merge_external_features(
        app_test, bureau_feats, prev_feats, inst_feats, cc_feats, pos_feats, bureau_bb_curr
    )

    app_train = create_cross_table_features(app_train)
    app_test = create_cross_table_features(app_test)
    app_train = _defragment(app_train)
    app_test = _defragment(app_test)

    del bureau_feats, bureau_bb_curr, prev_feats, inst_feats, cc_feats, pos_feats
    gc.collect()

    print(f"Final: Train {app_train.shape}, Test {app_test.shape}")
    return app_train, app_test
