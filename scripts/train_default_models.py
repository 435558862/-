from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.models.classifiers.randomforest import RandomForest
from src.models.trainer import Trainer
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.target import TargetType


def train_model(league_id: str, model_id: str, target_type: TargetType) -> None:
    league_db = LeagueDatabase()
    dataset = league_db.load_league(league_id=league_id).dropna().reset_index(drop=True)
    model_db = ModelDatabase(league_id=league_id)

    if model_db.model_exists(model_id=model_id):
        print(f"模型已存在，跳过：{model_id}")
        return

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
    config["train"] = {
        "eval_samples_size": 20.0,
        "results": {"fit": metrics},
    }
    model_db.save_model(model=model, model_config=config)
    print(f"\n模型训练完成：{model_id}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    league_ids = LeagueDatabase().get_league_ids()
    if not league_ids:
        raise RuntimeError("没有找到已创建的联赛。")

    selected_league = league_ids[0]
    print(f"使用联赛：{selected_league}")
    train_model(selected_league, "默认胜平负模型", TargetType.RESULT)
    train_model(selected_league, "默认大小球模型", TargetType.OVER_UNDER)
