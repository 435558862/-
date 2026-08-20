from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.randomforest import RandomForest
from src.models.trainer import Trainer
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.target import TargetType
from scripts.rename_league import rename_model


LEAGUES = ["英超", "西甲", "德甲", "意甲", "法甲"]


def normalize_existing_model_names(league_id: str) -> None:
    model_db = ModelDatabase(league_id)
    renames = {
        "默认胜平负模型": f"{league_id}胜平负模型",
        "默认大小球模型": f"{league_id}大小球模型",
    }
    for old_id, new_id in renames.items():
        if model_db.model_exists(old_id) and not model_db.model_exists(new_id):
            rename_model(league_id, old_id, new_id)
            model_db = ModelDatabase(league_id)


def train_score_model(league_id: str) -> None:
    model_id = f"{league_id}比分模型"
    model_db = ModelDatabase(league_id)
    if model_db.model_exists(model_id):
        print(f"[模型已存在] {model_id}")
        return

    league_db = LeagueDatabase()
    raw_dataset = league_db.load_league(league_id)
    # Only the Premier League currently uses the half-time target. A partially
    # downloaded HTR column in another league must not discard its historical rows.
    if league_id != "英超" and "HTR" in raw_dataset.columns:
        raw_dataset = raw_dataset.drop(columns=["HTR"])
        league_db.save_league(raw_dataset, league_db.index[league_id])
        print(f"[数据修复] 已移除 {league_id} 的不完整半场字段")
    dataset = raw_dataset.dropna().reset_index(drop=True)
    train_df, eval_df = train_test_split(df=dataset, test_size=20.0)
    german_params = {
        "n_estimators": 1000,
        "min_samples_leaf": 15,
        "min_samples_split": 30,
        "max_features": "sqrt",
        "max_depth": 7,
    }
    default_params = {
        "n_estimators": 1000,
        "min_samples_leaf": 10,
        "min_samples_split": 20,
        "max_features": None,
        "max_depth": 6,
    }
    score_params = german_params if league_id == "德甲" else default_params
    model = RandomForest(
        league_id=league_id,
        model_id=model_id,
        target_type=TargetType.SCORE,
        calibrate_probabilities=False,
        n_estimators=score_params["n_estimators"],
        criterion="gini",
        min_samples_leaf=score_params["min_samples_leaf"],
        min_samples_split=score_params["min_samples_split"],
        max_features=score_params["max_features"],
        max_depth=score_params["max_depth"],
        class_weight=False,
    )
    print(f"\n[训练] {model_id}，有效样本 {len(dataset)}")
    model, metrics = Trainer().train(
        model=model,
        train_df=train_df,
        eval_df=eval_df,
        check_nan=True,
    )
    config = model.get_default_model_config()
    config["train"] = {"eval_samples_size": 20.0, "results": {"fit": metrics}}
    model_db.save_model(model=model, model_config=config)
    print(f"[完成] {model_id}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    for current_league in LEAGUES:
        normalize_existing_model_names(current_league)
        train_score_model(current_league)
    print("\n五大联赛胜平负、大小球和比分模型均已准备完成。")
