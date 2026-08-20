import os
import numpy as np
import pandas as pd

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.randomforest import RandomForest
from src.models.trainer import Trainer
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.target import TargetType, construct_targets


LEAGUES = ["英超", "西甲", "德甲", "意甲", "法甲"]
TASKS = [
    ("胜平负", TargetType.RESULT),
    ("大小球", TargetType.OVER_UNDER),
    ("比分", TargetType.SCORE),
]

PARAM_GRID = [
    {"max_depth": 4, "min_samples_leaf": 20, "min_samples_split": 40, "max_features": None, "class_weight": False},
    {"max_depth": 6, "min_samples_leaf": 10, "min_samples_split": 20, "max_features": None, "class_weight": False},
    {"max_depth": 7, "min_samples_leaf": 15, "min_samples_split": 30, "max_features": "sqrt", "class_weight": False},
    {"max_depth": 9, "min_samples_leaf": 7, "min_samples_split": 14, "max_features": "sqrt", "class_weight": False},
    {"max_depth": 10, "min_samples_leaf": 5, "min_samples_split": 10, "max_features": "sqrt", "class_weight": True},
]


def build_model(league_id, model_id, target_type, params, n_estimators):
    return RandomForest(
        league_id=league_id,
        model_id=model_id,
        target_type=target_type,
        calibrate_probabilities=False,
        n_estimators=n_estimators,
        criterion="gini",
        **params,
    )


def top_k_accuracy(model, df, target_type, k):
    y_true = construct_targets(df, target_type)
    probabilities = model.predict_proba(df)
    classes = np.asarray(model.classifier.classes_)
    k = min(k, probabilities.shape[1])
    top_indices = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    top_classes = classes[top_indices]
    return round(float(np.mean(np.any(top_classes == y_true[:, None], axis=1))), 3)


def majority_baseline(train_df, test_df, target_type):
    train_y = construct_targets(train_df, target_type)
    test_y = construct_targets(test_df, target_type)
    classes, counts = np.unique(train_y, return_counts=True)
    majority = classes[np.argmax(counts)]
    return round(float(np.mean(test_y == majority)), 3)


def tune_one(league_id, task_name, target_type, all_reports):
    league_db = LeagueDatabase()
    dataset = league_db.load_league(league_id).dropna().reset_index(drop=True)
    train_validation, test_df = train_test_split(dataset, 20.0)
    train_df, validation_df = train_test_split(train_validation, 25.0)
    model_id = f"{league_id}{task_name}模型"

    best = None
    print(f"\n[{league_id} / {task_name}] 训练={len(train_df)} 验证={len(validation_df)} 测试={len(test_df)}")
    for candidate_id, params in enumerate(PARAM_GRID, start=1):
        candidate = build_model(
            league_id, model_id, target_type, params, n_estimators=400
        )
        candidate.fit(train_df=train_df)
        validation_metrics = candidate.evaluate(validation_df).iloc[0]
        validation_accuracy = float(validation_metrics["Accuracy"])
        print(f"  参数{candidate_id}: 验证准确率={validation_accuracy:.3f} {params}")
        if best is None or validation_accuracy > best["accuracy"]:
            best = {
                "candidate_id": candidate_id,
                "accuracy": validation_accuracy,
                "params": params,
            }

    final_model = build_model(
        league_id, model_id, target_type, best["params"], n_estimators=1000
    )
    final_model, metrics = Trainer().train(
        model=final_model,
        train_df=train_validation,
        eval_df=test_df,
        check_nan=True,
    )
    test_accuracy = float(metrics[metrics["data"] == "eval"]["Accuracy"].iloc[0])
    top3 = top_k_accuracy(final_model, test_df, target_type, 3)
    top5 = top_k_accuracy(final_model, test_df, target_type, 5)
    baseline = majority_baseline(train_validation, test_df, target_type)

    config = final_model.get_default_model_config()
    config["train"] = {
        "eval_samples_size": 20.0,
        "results": {"fit": metrics},
        "tuning": {
            "method": "60%训练 / 20%验证选参 / 20%最近比赛测试",
            "selected_candidate": best["candidate_id"],
            "validation_accuracy": best["accuracy"],
            "test_accuracy": test_accuracy,
            "top3_accuracy": top3,
            "top5_accuracy": top5,
            "majority_baseline": baseline,
        },
    }
    ModelDatabase(league_id).save_model(final_model, config)

    report = {
        "联赛": league_id,
        "模型": model_id,
        "任务": task_name,
        "有效样本": len(dataset),
        "验证准确率": best["accuracy"],
        "最近比赛测试准确率": test_accuracy,
        "Top3命中率": top3,
        "Top5命中率": top5,
        "简单基线": baseline,
        "参数组": best["candidate_id"],
    }
    all_reports.append(report)
    print(
        f"  [最终] 测试={test_accuracy:.3f} Top3={top3:.3f} "
        f"Top5={top5:.3f} 基线={baseline:.3f}"
    )


if __name__ == "__main__":
    reports = []
    for league_id in LEAGUES:
        for task_name, target_type in TASKS:
            tune_one(league_id, task_name, target_type, reports)

    tune_one("英超", "半全场", TargetType.HALF_FULL, reports)
    report_df = pd.DataFrame(reports)
    os.makedirs("storage/reports", exist_ok=True)
    report_df.to_csv("storage/reports/模型调教报告.csv", index=False)
    print("\n" + report_df.to_string(index=False))
    print("\n全部模型调教完成。")
