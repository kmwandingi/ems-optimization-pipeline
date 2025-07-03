import pandas as pd
import numpy as np
from pathlib import Path
import joblib, warnings
from typing import Tuple

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import PredefinedSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)
from pandas.tseries.holiday import USFederalHolidayCalendar

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -------------------------------------------------------------------------------
# STEP 1 – DATA PREPARATION
# -------------------------------------------------------------------------------

def data_prep_step(
    parquet_dir: str = "processed_data",
    default_threshold: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads per-building *_processed_data.parquet files, then builds:
      - daily_df with 7-day rolling, *prior-day* peak_hour (sin/cos), target device_used
      - hourly_df with device_on_at_hour, day_cumulative_usage, circular & raw features,
        plus peak_usage_ratio & time_since_last_usage
    """
    parquet_dir = Path(parquet_dir)
    files = list(parquet_dir.glob("*_processed_data.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    daily_records, hourly_records = [], []

    def rolling7(df, col):
        return df[col].shift(1).rolling(7, min_periods=1).mean()

    for fp in files:
        df = pd.read_parquet(fp)

        if "utc_timestamp" in df.columns:
            df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
            df = df.set_index("utc_timestamp")
        df.index = (
            df.index.tz_localize(None)
                    .tz_localize("UTC")
                    .tz_convert("Europe/Berlin")
        )

        building = fp.stem.replace("_processed_data", "")
        required = {
            "pv_forecast",
            "DE_temperature",
            "DE_radiation_direct_horizontal",
            "DE_radiation_diffuse_horizontal"
        }
        if not required.issubset(df.columns):
            print(f"⚠️ Skipping {building}: missing weather/pv columns")
            continue

        device_cols = [
            c for c in df.columns
            if "_residential" in c
               and all(x not in c for x in ("pv","import","export","grid"))
        ]
        if not device_cols:
            print(f"⚠️ Skipping {building}: no device columns")
            continue

        # HOURLY base
        hr = df[device_cols].copy()
        hr["temperature"]     = df["DE_temperature"]
        hr["solar_radiation"] = (
            df["DE_radiation_direct_horizontal"]
            + df["DE_radiation_diffuse_horizontal"]
        )
        hr["pv_forecast"]     = df["pv_forecast"]
        hr["hour"]            = hr.index.hour
        hr["day_of_week"]     = hr.index.dayofweek
        hr["is_weekend"]      = (hr["day_of_week"] >= 5).astype(int)

        # DAILY aggregates
        daily_usage = df[device_cols].resample("D").sum()
        daily_temp  = df["DE_temperature"].resample("D").mean()
        daily_rad   = (
            df["DE_radiation_direct_horizontal"]
            + df["DE_radiation_diffuse_horizontal"]
        ).resample("D").mean()
        daily_pv    = df["pv_forecast"].resample("D").mean()

        dd = daily_usage.copy()
        dd["temperature"]     = daily_temp
        dd["solar_radiation"] = daily_rad
        dd["pv_forecast"]     = daily_pv
        dd["day_of_week"]     = dd.index.dayofweek

        for dev in device_cols:
            dd[f"{dev}_rolling_7d"] = rolling7(dd, dev)

        # per-device
        for dev in device_cols:
            parts = dev.split("_")
            device_type = (
                "washing_machine"
                if parts[-2:] == ["washing","machine"]
                else parts[-1]
            )

            pos_usages = hr[dev][hr[dev] > 0]
            dyn_thr = np.quantile(pos_usages, 0.25) if len(pos_usages)>0 else default_threshold

            tmp = hr[[dev]].rename(columns={dev:"usage"})
            tmp["date"] = tmp.index.date
            peak_per_day = tmp.groupby("date")["usage"] \
                              .apply(lambda s: s.idxmax().hour if s.notna().any() else np.nan)
            peak_per_day_prior = peak_per_day.shift(1)

            # DAILY rows
            for ts, row in dd.iterrows():
                used_flag = int(row[dev] > dyn_thr)
                ph = peak_per_day_prior.get(ts.date(), np.nan)
                sin_h = np.sin(2*np.pi*ph/24) if not np.isnan(ph) else np.nan
                cos_h = np.cos(2*np.pi*ph/24) if not np.isnan(ph) else np.nan

                daily_records.append({
                    "building":          building,
                    "date":              ts,
                    "device":            dev,
                    "plain_device_type": device_type,
                    "temperature":       row["temperature"],
                    "solar_radiation":   row["solar_radiation"],
                    "pv_forecast":       row["pv_forecast"],
                    "day_of_week":       row["day_of_week"],
                    "rolling_7d_usage":  row[f"{dev}_rolling_7d"],
                    "actual_usage_kWh":  row[dev],
                    "device_used":       used_flag,
                    "peak_hour":         ph,
                    "peak_hour_sin":     sin_h,
                    "peak_hour_cos":     cos_h
                })

            # HOURLY rows
            df_dev = hr[[dev,"temperature","solar_radiation","pv_forecast",
                         "hour","day_of_week","is_weekend"]].copy()
            df_dev.rename(columns={dev:"actual_usage_kWh"}, inplace=True)
            df_dev["device_on_at_hour"] = (df_dev["actual_usage_kWh"] > dyn_thr).astype(int)
            df_dev["day_cumulative_usage"] = (
                df_dev.groupby(df_dev.index.date)["actual_usage_kWh"]
                      .apply(lambda s: s.shift().fillna(0).cumsum())
            )
            df_dev["prev_hour_on"] = df_dev["device_on_at_hour"].shift().fillna(0).astype(int)
            df_dev["hour_sin"]     = np.sin(2*np.pi*df_dev["hour"]/24)
            df_dev["hour_cos"]     = np.cos(2*np.pi*df_dev["hour"]/24)
            df_dev["device"]       = dev
            df_dev["plain_device_type"] = device_type
            df_dev["building"]     = building
            df_dev["datetime"]     = df_dev.index
            df_dev["date"]         = df_dev["datetime"].dt.date

            # ─── NEW FEATURES ───
            daily_max = (
                df_dev.groupby(df_dev["date"])["actual_usage_kWh"]
                     .transform("max")
                     .replace(0, np.nan)
            )
            df_dev["peak_usage_ratio"]     = df_dev["actual_usage_kWh"] / daily_max
            last_usage_ts = df_dev["datetime"].where(
                df_dev["actual_usage_kWh"]>0
            ).ffill()
            df_dev["time_since_last_usage"] = (
                (df_dev["datetime"] - last_usage_ts)
                 .dt.total_seconds()
                 .div(3600)
                 .fillna(0)
            )

            hourly_records.extend(df_dev.to_dict("records"))

    daily_df  = pd.DataFrame(daily_records).dropna(subset=["actual_usage_kWh"])
    hourly_df = pd.DataFrame(hourly_records).dropna(subset=["actual_usage_kWh"])
    return daily_df, hourly_df


# -------------------------------------------------------------------------------
# STEP 2 – FEATURE ENGINEERING
# -------------------------------------------------------------------------------

def feature_engineering_step(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build X_day, y_day, X_hr exactly as in your pipeline."""
    # unify
    daily_df["date"]      = pd.to_datetime(daily_df["date"]).dt.date
    hourly_df["date"]     = pd.to_datetime(hourly_df["date"]).dt.date
    hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])

    # calendar
    daily_df["month"]      = pd.to_datetime(daily_df["date"]).dt.month
    daily_df["is_weekend"] = daily_df["day_of_week"].isin([5,6]).astype(int)
    hourly_df["month"]     = pd.to_datetime(hourly_df["date"]).dt.month
    daily_df["dow_pv"]     = daily_df["day_of_week"] * daily_df["pv_forecast"]
    hourly_df["dow_pv"]    = hourly_df["day_of_week"] * hourly_df["pv_forecast"]

    # rolling & lags
    hourly_df.sort_values(["device","datetime"], inplace=True)
    hourly_df["rolling_24h_usage"] = (
        hourly_df.groupby("device")["actual_usage_kWh"]
                 .transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
                 .fillna(0)
    )
    daily_df["rolling_3d_usage"] = (
        daily_df.groupby("device")["rolling_7d_usage"]
                .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
                .fillna(0)
    )
    daily_df["rolling_14d_usage"] = (
        daily_df.groupby("device")["rolling_7d_usage"]
                .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
                .fillna(0)
    )
    daily_df["lag_7d_usage"] = (
        daily_df.groupby("device")["rolling_7d_usage"]
                .shift(7).fillna(0)
    )

    # interactions
    daily_df["temp_month"] = daily_df["temperature"] * daily_df["month"]
    hourly_df["temp_hour"] = hourly_df["temperature"]  * hourly_df["hour"]

    # holidays
    hols = USFederalHolidayCalendar().holidays(
        start=min(daily_df["date"]), end=max(daily_df["date"])
    )
    daily_df["is_holiday"] = daily_df["date"].isin(hols).astype(int)

    # one-hot building & device_type for daily
    daily_df = pd.get_dummies(daily_df, columns=["building"], drop_first=True)
    dt = pd.get_dummies(
        daily_df["plain_device_type"], prefix="device_type", drop_first=True
    )
    daily_df = pd.concat([daily_df, dt], axis=1)

    # OHE hours for hourly
    hr_ohe = pd.get_dummies(hourly_df["hour"].astype(int), prefix="hour")
    hourly_df = pd.concat([hourly_df, hr_ohe], axis=1)
    # recalc circular hour
    hourly_df["hour_sin"] = np.sin(2*np.pi*hourly_df["hour"]/24)
    hourly_df["hour_cos"] = np.cos(2*np.pi*hourly_df["hour"]/24)

    # feature lists
    feat_cols_day = [
        "temperature","solar_radiation","pv_forecast",
        "day_of_week","rolling_7d_usage","rolling_3d_usage",
        "rolling_14d_usage","lag_7d_usage","peak_hour_sin","peak_hour_cos",
        "temp_month","dow_pv","is_holiday","is_weekend"
    ] + [c for c in daily_df.columns if c.startswith(("building_","device_type_"))]

    feat_cols_hr = [
        "temperature","solar_radiation","pv_forecast",
        "hour","day_of_week","month","is_weekend",
        "rolling_24h_usage","temp_hour","dow_pv",
        "day_cumulative_usage","prev_hour_on",
        "hour_sin","hour_cos"
    ] + list(hr_ohe.columns) + ["building","plain_device_type"]

    # cast numeric only
    cat_cols_hr = ["day_of_week","is_weekend","prev_hour_on","plain_device_type","building"]
    num_feats_hr = [c for c in feat_cols_hr if c not in cat_cols_hr]
    hourly_df[num_feats_hr] = hourly_df[num_feats_hr].astype("float32")

    # convert cats to string
    for c in cat_cols_hr:
        if c in hourly_df.columns:
            hourly_df[c] = hourly_df[c].astype(str)

    # split target
    X_day = daily_df[feat_cols_day + ["date","plain_device_type"]]
    y_day = daily_df["device_used"].astype("uint8")
    X_hr  = hourly_df[feat_cols_hr + ["date","datetime","actual_usage_kWh","device_on_at_hour"]]

    return X_day, y_day, X_hr


# -------------------------------------------------------------------------------
# STEP 3 – DAILY MODEL TRAINING + CALIBRATION
# -------------------------------------------------------------------------------

def train_daily_step(
    X_day: pd.DataFrame,
    y_day: pd.Series
) -> CalibratedClassifierCV:
    """Train LightGBM + pick best between Platt (sigmoid) and isotonic."""
    stats = pd.DataFrame({
        "total": X_day.groupby("plain_device_type").size(),
        "on":    y_day.groupby(X_day["plain_device_type"]).sum()
    })
    stats["ratio"] = stats["on"] / stats["total"]
    known_cont = {"freezer","refrigerator"}
    new = stats[stats["ratio"] > 0.8].index.difference(known_cont)
    all_cont = known_cont.union(new)
    mask = X_day["plain_device_type"].isin(all_cont)
    Xf, yf = X_day[~mask], y_day[~mask]

    # time‐based 5-fold
    dates = sorted(Xf["date"].unique())
    b = int(np.ceil(len(dates)/5))
    fold = np.zeros(len(Xf), dtype=int)
    for i in range(5):
        fold[Xf["date"].isin(dates[i*b:(i+1)*b])] = i
    ps = PredefinedSplit(fold)

    # raw LightGBM
    params = dict(
        objective="binary",
        metric="auc",
        class_weight="balanced",
        learning_rate=0.07,
        num_leaves=48,
        reg_alpha=0.6,
        reg_lambda=0.2,
        n_estimators=1000,
        random_state=42,
    )
    Xnum = Xf.drop(columns=["date","plain_device_type"])
    cv_idx = fold != fold.max()
    tmp = lgb.LGBMClassifier(**params).fit(
        Xnum.iloc[cv_idx], yf.iloc[cv_idx],
        eval_set=[(Xnum.iloc[~cv_idx], yf.iloc[~cv_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    params["n_estimators"] = tmp.best_iteration_
    model_raw = lgb.LGBMClassifier(**params).fit(Xnum, yf)

    # calibration
    cal_iso = CalibratedClassifierCV(model_raw, method="isotonic", cv=ps).fit(Xnum, yf)
    cal_pl  = CalibratedClassifierCV(model_raw, method="sigmoid",  cv=ps).fit(Xnum, yf)

    def cal_err(y,p):
        fp, mp = calibration_curve(y, p, n_bins=10)
        return np.mean((fp-mp)**2)

    err_iso = cal_err(yf, cal_iso.predict_proba(Xnum)[:,1])
    err_pl  = cal_err(yf, cal_pl.predict_proba(Xnum)[:,1])
    best_cal   = "isotonic" if err_iso < err_pl else "sigmoid"
    best_model = {"isotonic": cal_iso, "sigmoid": cal_pl}[best_cal]
    joblib.dump(best_model, f"artifacts/day_model_{best_cal[:2]}.pkl")
    return best_model


# -------------------------------------------------------------------------------
# STEP 4 – HOURLY MODEL TRAINING
# -------------------------------------------------------------------------------

def train_hourly_step(X_hr):
    """
    Train CatBoost on/off-per-hour model using only non-leaky features.
    """
    Xu = X_hr.copy()
    yu = Xu["device_on_at_hour"].astype("int8")

    # inverse-prevalence weights
    pos_rate = Xu.groupby("plain_device_type")["device_on_at_hour"].mean()
    w_train  = Xu.apply(
        lambda r: 1/pos_rate[r["plain_device_type"]] if r["device_on_at_hour"] else 1,
        axis=1
    ).values

    # features & cat list
    hour_feat   = [c for c in X_hr.columns if c not in ("date","datetime","actual_usage_kWh","device_on_at_hour")]
    cat_features= ["day_of_week","is_weekend","prev_hour_on","plain_device_type","building"]

    # split by date
    dates = sorted(Xu["date"].unique())
    split = int(0.8*len(dates))
    tr_idx = Xu["date"].isin(dates[:split])
    te_idx = Xu["date"].isin(dates[split:])

    # Pool expects cats as str or int
    for c in cat_features:
        Xu[c] = Xu[c].astype(str)

    train_pool = Pool(
        data=Xu.loc[tr_idx, hour_feat],
        label=yu.loc[tr_idx],
        weight=w_train[tr_idx],
        cat_features=cat_features
    )
    test_pool = Pool(
        data=Xu.loc[te_idx, hour_feat],
        label=yu.loc[te_idx],
        cat_features=cat_features
    )

    cat_params = dict(
        depth=6, iterations=800, learning_rate=0.05,
        l2_leaf_reg=3, random_state=42,
        loss_function="Logloss", auto_class_weights="Balanced",
        task_type="GPU", verbose=100
    )

    print("Training hourly CatBoost …")
    cat_clf = CatBoostClassifier(**cat_params).fit(train_pool, eval_set=test_pool)

    joblib.dump(hour_feat, "artifacts/hour_feat.pkl")
    cat_clf.save_model("artifacts/cat_hourly_model.cb")
    return cat_clf



# -------------------------------------------------------------------------------
# STEP 5 – EVALUATION & VISUALIZATION
# -------------------------------------------------------------------------------

import os

def evaluation_step(
    calibrated_daily_model,
    catboost_hourly_model,
    X_day: pd.DataFrame,
    y_day: pd.Series,
    X_hr: pd.DataFrame,
    output_dir: str = "artifacts/evaluation"
) -> None:
    """Compute daily & hourly metrics and save 4-panel device plots."""
    os.makedirs(output_dir, exist_ok=True)

    # --- daily metrics ---
    daily_features = X_day.drop(columns=["date","plain_device_type"])
    probs_day = calibrated_daily_model.predict_proba(daily_features)[:,1]
    print(f"Daily ROC AUC: {roc_auc_score(y_day, probs_day):.4f}")

    # --- hourly metrics ---
    # (reuse exactly the same feature‐selection logic you used in training)
    feat_names = catboost_hourly_model.feature_names_
    cat_idxs   = catboost_hourly_model.get_cat_feature_indices()
    cat_names  = [feat_names[i] for i in cat_idxs]

    # evaluate on the same hold-out you used before
    dates      = sorted(X_hr["date"].unique())
    split      = int(0.8 * len(dates))
    test_mask  = X_hr["date"].isin(dates[split:])
    X_test     = X_hr[test_mask].copy()
    y_test     = X_test["device_on_at_hour"]

    # assemble Pool exactly as in training
    X_test_feat = X_test[feat_names].copy()
    for c in feat_names:
        if c in cat_names:
            X_test_feat[c] = X_test_feat[c].astype(str)
        else:
            X_test_feat[c] = pd.to_numeric(X_test_feat[c], errors="raise")

    test_pool = Pool(
        data=X_test_feat,
        label=y_test,
        cat_features=cat_names
    )
    y_pred = catboost_hourly_model.predict_proba(test_pool)[:,1]

    print("Hourly ROC AUC:", roc_auc_score(y_test, y_pred))
    print("Hourly PR  AUC:", average_precision_score(y_test, y_pred))

    # now per‐device 4-panel plots
    TEST_DF = X_test.copy()
    TEST_DF["p_pred"] = y_pred

    CONTINUOUS = {"freezer","refrigerator","pump"}
    for dev in sorted(TEST_DF["plain_device_type"].unique()):
        if dev in CONTINUOUS:
            continue
        dev_df = TEST_DF[TEST_DF["plain_device_type"] == dev]
        _plot_device_metrics_and_save(dev_df,
                                      catboost_hourly_model,
                                      os.path.join(output_dir, f"{dev}.png"))


def _plot_device_metrics_and_save(dev_df, MODEL, save_path: str):
    """Same 4-panel logic, but _save_ the figure instead of plt.show()."""
    feat_names = MODEL.feature_names_
    cat_idxs   = MODEL.get_cat_feature_indices()
    cat_names  = [feat_names[i] for i in cat_idxs]

    # build X & y
    X = dev_df[feat_names].copy()
    for c in feat_names:
        if c in cat_names:
            X[c] = X[c].astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="raise")
    y_true = dev_df["device_on_at_hour"].values
    pool   = Pool(data=X, label=y_true, cat_features=cat_names)
    y_pred = MODEL.predict_proba(pool)[:,1]

    # metrics
    fpr, tpr, _   = roc_curve(y_true, y_pred)
    roc_auc       = roc_auc_score(y_true, y_pred)
    prec, reca, _ = precision_recall_curve(y_true, y_pred)
    ap            = average_precision_score(y_true, y_pred)

    # calibration curves
    mean_pred, frac_pos = {}, {}
    for method in ("raw","sigmoid","isotonic"):
        if method == "raw":
            mp, fp = calibration_curve(y_true, y_pred, n_bins=10, strategy="quantile")
        else:
            cal = CalibratedClassifierCV(MODEL, method=method, cv="prefit")
            cal.fit(X, y_true)
            pcal = cal.predict_proba(X)[:,1]
            mp, fp = calibration_curve(y_true, pcal, n_bins=10, strategy="quantile")
        mean_pred[method], frac_pos[method] = mp, fp

    # empirical vs predicted PMF
    emp_counts = dev_df.groupby("hour")["device_on_at_hour"]\
                       .sum().reindex(range(24), fill_value=0)
    emp_pmf    = emp_counts / emp_counts.sum() if emp_counts.sum() else np.zeros(24)
    pred_mass  = dev_df.groupby("hour")["p_pred"]\
                       .sum().reindex(range(24), fill_value=0)
    pred_pmf   = pred_mass / pred_mass.sum() if pred_mass.sum() else np.zeros(24)

    # plot
    fig = plt.figure(figsize=(12,3.5))
    gs  = GridSpec(1,4, width_ratios=[1.2,1.2,1.2,1.4], wspace=0.4)

    # ROC
    ax = fig.add_subplot(gs[0])
    ax.plot(fpr, tpr, lw=2); ax.plot([0,1],[0,1],"--",color="gray")
    ax.set(title=f"ROC AUC={roc_auc:.3f}", xlabel="FPR", ylabel="TPR")

    # PR
    ax = fig.add_subplot(gs[1])
    ax.plot(reca, prec, lw=2)
    ax.set(xlim=(0,1), ylim=(0,1),
           title=f"PR AP={ap:.3f}", xlabel="Recall", ylabel="Precision")

    # Calibration
    ax = fig.add_subplot(gs[2])
    ax.plot(mean_pred["raw"],    frac_pos["raw"],    "o-", label="Raw")
    ax.plot(mean_pred["sigmoid"],frac_pos["sigmoid"],"s-", label="Platt")
    ax.plot(mean_pred["isotonic"],frac_pos["isotonic"],"^-", label="Isotonic")
    ax.plot([0,1],[0,1],"--",color="gray")
    ax.set(title="Calibration", xlabel="Mean pred P", ylabel="Freq positives")
    ax.legend(fontsize=7, loc="lower right")

    # PMF
    ax = fig.add_subplot(gs[3])
    hrs = np.arange(24)
    ax.plot(hrs, pred_pmf, "o-", label="Pred PMF")
    ax.plot(hrs, emp_pmf,  "x--", label="Emp PMF")
    ax.set(xticks=range(0,24,3),
           title="Hourly distribution", xlabel="Hour", ylabel="Probability")
    ax.legend(fontsize=7)

    fig.suptitle(f"Device type: {dev_df['plain_device_type'].iloc[0]}", y=1.02)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
