from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.randomforest import RandomForest
from src.models.trainer import Trainer
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.target import TargetType


BIG_FIVE = {
    ("England", "Premier-League"): "英超",
    ("Spain", "La-Liga"): "西甲",
    ("Germany", "Bundesliga-1"): "德甲",
    ("Italy", "Serie-A"): "意甲",
    ("France", "Ligue-1"): "法甲",
}

STATS_COLUMNS = [
    "HW", "AW", "HL", "AL", "HGF", "AGF", "HAGF", "HGA", "AGA", "HAGA",
    "HGD", "AGD", "HAGD", "HWGD", "AWGD", "HAWGD", "HLGD", "ALGD", "HALGD",
    "HW%", "HL%", "AW%", "AL%", "HSTF", "ASTF", "HCF", "ACF",
]


def ensure_leagues() -> None:
    league_db = LeagueDatabase()
    available = {(league.country, league.name): league for league in league_db.leagues}

    for key, league_id in BIG_FIVE.items():
        if league_db.league_exists(league_id):
            dataset = league_db.load_league(league_id)
            print(f"[数据已存在] {league_id}: {len(dataset)} 场")
            continue

        print(f"\n[下载数据] {league_id}")
        league = available[key].clone(
            start_year=2005,
            league_id=league_id,
            match_history_window=4,
            goal_diff_margin=2,
            stats_columns=STATS_COLUMNS,
        )
        dataset = league_db.create_league(league)
        if dataset is None or dataset.empty:
            raise RuntimeError(f"{league_id} 数据下载失败")
        print(f"[下载完成] {league_id}: {len(dataset)} 场")


def train_model(league_id: str, model_id: str, target_type: TargetType) -> None:
    league_db = LeagueDatabase()
    dataset = league_db.load_league(league_id=league_id).dropna().reset_index(drop=True)
    model_db = ModelDatabase(league_id=league_id)

    if model_db.model_exists(model_id=model_id):
        print(f"[模型已存在] {league_id} / {model_id}")
        return

    print(f"\n[开始训练] {league_id} / {model_id}，有效样本 {len(dataset)}")
    train_df, eval_df = train_test_split(df=dataset, test_size=20.0)
    model = RandomForest(
        league_id=league_id,
        model_id=model_id,
        target_type=target_type,
        calibrate_probabilities=False,
        n_estimators=300,
        criterion="gini",
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        max_depth=12,
        class_weight=True,
    )
    model, metrics = Trainer().train(
        model=model,
        train_df=train_df,
        eval_df=eval_df,
        check_nan=True,
    )
    config = model.get_default_model_config()
    config["train"] = {"eval_samples_size": 20.0, "results": {"fit": metrics}}
    model_db.save_model(model=model, model_config=config)
    print(f"[训练完成] {league_id} / {model_id}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    ensure_leagues()
    for current_league_id in BIG_FIVE.values():
        train_model(current_league_id, "默认胜平负模型", TargetType.RESULT)
        train_model(current_league_id, "默认大小球模型", TargetType.OVER_UNDER)
    print("\n五大联赛数据与模型全部准备完成。")
