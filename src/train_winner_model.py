from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fastf1
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RaceRecord:
    year: int
    round_number: int
    event_name: str
    event_date: pd.Timestamp
    driver: str
    driver_number: str
    team: str
    grid_position: float
    finish_position: float
    points: float
    won: int


def _safe_float(value) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fetch_race_records(start_year: int, end_year: int) -> pd.DataFrame:
    records: list[RaceRecord] = []

    for year in range(start_year, end_year + 1):
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule["EventFormat"].notna()]

        for _, event in races.iterrows():
            round_number = int(event["RoundNumber"])
            event_name = str(event["EventName"])
            event_date = pd.to_datetime(event["EventDate"])

            try:
                session = fastf1.get_session(year, round_number, "R")
                session.load(
                    laps=False,
                    telemetry=False,
                    weather=False,
                    messages=False,
                )
            except Exception as exc:
                print(f"Skipping {year} round {round_number} ({event_name}): {exc}")
                continue

            results = session.results
            if results is None or results.empty:
                continue

            for _, row in results.iterrows():
                finish_position = _safe_float(row.get("Position"))
                points = _safe_float(row.get("Points"))

                if np.isnan(finish_position):
                    continue

                records.append(
                    RaceRecord(
                        year=year,
                        round_number=round_number,
                        event_name=event_name,
                        event_date=event_date,
                        driver=str(row.get("Abbreviation", "UNK")),
                        driver_number=str(row.get("DriverNumber", "")),
                        team=str(row.get("TeamName", "Unknown")),
                        grid_position=_safe_float(row.get("GridPosition")),
                        finish_position=finish_position,
                        points=points if not np.isnan(points) else 0.0,
                        won=1 if finish_position == 1.0 else 0,
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        raise RuntimeError("No race records fetched. Check FastF1 connectivity and year range.")

    return df.sort_values(["event_date", "round_number", "driver"]).reset_index(drop=True)


def build_features(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    work = df.copy()
    work = work.sort_values(["driver", "event_date", "round_number"]).reset_index(drop=True)

    grouped_driver = work.groupby("driver", group_keys=False)
    grouped_team = work.groupby("team", group_keys=False)

    work["driver_avg_finish_last_n"] = grouped_driver["finish_position"].transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )
    work["driver_avg_grid_last_n"] = grouped_driver["grid_position"].transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )
    work["driver_avg_points_last_n"] = grouped_driver["points"].transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )

    work["team_avg_points_last_n"] = grouped_team["points"].transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )
    work["team_avg_finish_last_n"] = grouped_team["finish_position"].transform(
        lambda s: s.shift(1).rolling(rolling_window, min_periods=1).mean()
    )

    first_events = work.groupby("event_name")["event_date"].transform("min") == work["event_date"]
    work = work[~first_events | work["driver_avg_finish_last_n"].notna()].copy()

    return work.sort_values(["event_date", "round_number", "driver"]).reset_index(drop=True)


def train_and_evaluate(df: pd.DataFrame, artifacts_dir: Path) -> None:
    feature_cols_num = [
        "grid_position",
        "driver_avg_finish_last_n",
        "driver_avg_grid_last_n",
        "driver_avg_points_last_n",
        "team_avg_points_last_n",
        "team_avg_finish_last_n",
        "round_number",
        "year",
    ]
    feature_cols_cat = ["driver", "team"]

    df = df.sort_values(["event_date", "round_number"]).reset_index(drop=True)
    unique_events = df[["year", "round_number"]].drop_duplicates().reset_index(drop=True)

    split_index = int(len(unique_events) * 0.8)
    train_events = unique_events.iloc[:split_index]
    test_events = unique_events.iloc[split_index:]

    train_keys = set(zip(train_events["year"], train_events["round_number"]))
    test_keys = set(zip(test_events["year"], test_events["round_number"]))

    train_mask = df[["year", "round_number"]].apply(tuple, axis=1).isin(train_keys)
    test_mask = df[["year", "round_number"]].apply(tuple, axis=1).isin(test_keys)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("Not enough race events for train/test split. Increase year range.")

    X_train = train_df[feature_cols_num + feature_cols_cat]
    y_train = train_df["won"]
    X_test = test_df[feature_cols_num + feature_cols_cat]
    y_test = test_df["won"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                feature_cols_num,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                feature_cols_cat,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print("Validation metrics")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Log-loss: {ll:.4f}")
    print(f"Brier score: {brier:.4f}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "winner_model.joblib")

    with (artifacts_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump({"numeric": feature_cols_num, "categorical": feature_cols_cat}, f, indent=2)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an F1 winner prediction model using FastF1.")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--cache-dir", type=Path, default=Path(".fastf1_cache"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    fastf1.Cache.enable_cache(str(args.cache_dir))

    raw = fetch_race_records(args.start_year, args.end_year)
    dataset = build_features(raw)
    train_and_evaluate(dataset, args.artifacts_dir)


if __name__ == "__main__":
    main()
