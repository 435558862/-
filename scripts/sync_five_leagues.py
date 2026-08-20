"""Manually synchronize completed matches for every prediction league."""

from src.services.league_sync import sync_five_leagues


if __name__ == '__main__':
    results = sync_five_leagues()
    for league, item in results.items():
        print(
            f'{league}: 新增={item.get("added", 0)} '
            f'最新={item.get("latest_date", "-")} '
            f'重训模型={item.get("models_retrained", 0)}'
        )
