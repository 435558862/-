import os
import pickle
import sys


def rename_league(old_id: str, new_id: str) -> None:
    league_index_path = "storage/leagues/league_index.pkl"
    model_index_path = "storage/leagues/model_index.pkl"
    old_league_dir = f"storage/leagues/{old_id}"
    new_league_dir = f"storage/leagues/{new_id}"
    old_model_dir = f"storage/models/{old_id}"
    new_model_dir = f"storage/models/{new_id}"

    with open(league_index_path, "rb") as file:
        league_index = pickle.load(file)
    with open(model_index_path, "rb") as file:
        model_index = pickle.load(file)

    if old_id not in league_index:
        raise RuntimeError(f"找不到联赛：{old_id}")
    if new_id in league_index or os.path.exists(new_league_dir) or os.path.exists(new_model_dir):
        raise RuntimeError(f"新名称已经存在：{new_id}")

    league = league_index.pop(old_id)
    league._league_id = new_id
    league_index[new_id] = league

    if old_id in model_index:
        configs = model_index.pop(old_id)
        for config in configs.values():
            config["league_id"] = new_id
        model_index[new_id] = configs

    os.rename(old_league_dir, new_league_dir)
    if os.path.exists(old_model_dir):
        os.rename(old_model_dir, new_model_dir)

    with open(league_index_path, "wb") as file:
        pickle.dump(league_index, file)
    with open(model_index_path, "wb") as file:
        pickle.dump(model_index, file)

    print(f"联赛已改名：{old_id} -> {new_id}")


def rename_model(league_id: str, old_id: str, new_id: str) -> None:
    model_index_path = "storage/leagues/model_index.pkl"
    old_dir = f"storage/leagues/{league_id}/models/{old_id}"
    new_dir = f"storage/leagues/{league_id}/models/{new_id}"
    with open(model_index_path, "rb") as file:
        model_index = pickle.load(file)
    models = model_index[league_id]
    if old_id not in models:
        return
    if new_id in models or os.path.exists(new_dir):
        raise RuntimeError(f"模型名称已经存在：{new_id}")
    config = models.pop(old_id)
    config["model_id"] = new_id
    models[new_id] = config
    os.rename(old_dir, new_dir)
    with open(model_index_path, "wb") as file:
        pickle.dump(model_index, file)
    print(f"模型已改名：{old_id} -> {new_id}")


if __name__ == "__main__":
    if sys.argv[1] == "--model":
        rename_model(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        rename_league(sys.argv[1], sys.argv[2])
