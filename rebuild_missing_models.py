#!/usr/bin/env python3
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.preprocessing.utils.target import TargetType
from src.services.league_sync import _model_parameters


APP_ROOT = Path("/home/administrator/ProphitBet")
SOURCE_INDEX = Path(
    "/home/administrator/ProphitBet-backups/"
    "storage-before-encoding-fix-20260810/leagues/model_index.pkl"
)
MODEL_ROOT = APP_ROOT / "storage/leagues"


def classifier_path(league_id: str, model_id: str) -> Path:
    return MODEL_ROOT / league_id / "models" / model_id / "classifier.pkl"


def training_frame(raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    target_type = config.get("target_type")
    if target_type == TargetType.HALF_FULL:
        clean = raw.dropna().reset_index(drop=True)
        if len(clean) < 100:
            raise RuntimeError(
                f"{config['league_id']} 半全场有效样本不足：{len(clean)}"
            )
    else:
        clean = raw.drop(columns=["HTR"], errors="ignore")
        clean = clean.dropna().reset_index(drop=True)

    history_tuning = config.get("train", {}).get("history_weight_tuning", {})
    history_years = history_tuning.get("history_years")
    if history_years is not None:
        dates = pd.to_datetime(clean["Date"])
        cutoff = dates.max() - pd.DateOffset(years=int(history_years))
        clean = clean.loc[dates >= cutoff].reset_index(drop=True)
    return clean


def main() -> None:
    if APP_ROOT.resolve() != APP_ROOT or not SOURCE_INDEX.is_file():
        raise RuntimeError("Recovery paths are not in the expected deployment")

    with SOURCE_INDEX.open("rb") as handle:
        source_index = pickle.load(handle)

    expected = sum(len(models) for models in source_index.values())
    if expected != 45:
        raise RuntimeError(f"Expected 45 source model configs, found {expected}")

    league_db = LeagueDatabase()
    rebuilt = 0
    for league_id in sorted(source_index):
        raw = league_db.load_league(league_id)
        if raw is None or raw.empty:
            raise RuntimeError(f"Missing league dataset: {league_id}")

        for model_id, config in sorted(source_index[league_id].items()):
            path = classifier_path(league_id, model_id)
            if path.is_file():
                print(f"[已有] {league_id}/{model_id}", flush=True)
                continue

            fit_data = training_frame(raw, config)
            model = config["cls"](**_model_parameters(config))
            print(
                f"[训练] {league_id}/{model_id}: "
                f"{type(model).__name__}, {len(fit_data)} 样本",
                flush=True,
            )
            model.fit(fit_data)

            new_config = model.get_default_model_config()
            new_config["train"] = dict(config.get("train", {}))
            new_config["train"]["recovered_missing_classifier"] = {
                "at": datetime.now(timezone.utc).isoformat(),
                "samples": len(fit_data),
                "source": str(SOURCE_INDEX),
            }
            ModelDatabase(league_id).save_model(model, new_config)
            rebuilt += 1
            print(f"[完成] {league_id}/{model_id}", flush=True)

    physical = list(MODEL_ROOT.rglob("classifier.pkl"))
    if len(physical) != expected:
        raise RuntimeError(
            f"Expected {expected} classifier files after recovery, found {len(physical)}"
        )
    print(f"重建完成：新增 {rebuilt} 个模型，当前共 {len(physical)} 个。", flush=True)


if __name__ == "__main__":
    main()
