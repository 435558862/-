from pathlib import Path
import shutil

from src.database.league import LeagueDatabase


LEAGUES = ['西甲', '德甲', '意甲', '法甲']


if __name__ == '__main__':
    league_db = LeagueDatabase()
    backup_root = Path('storage/backups/before-half-time-refresh')
    backup_root.mkdir(parents=True, exist_ok=True)

    for league_id in LEAGUES:
        current = league_db.load_league(league_id)
        if 'HTR' in current.columns and current['HTR'].notna().sum() >= int(len(current) * 0.95):
            print(f'[{league_id}] 半场数据已完整，跳过。', flush=True)
            continue

        source_path = Path(f'storage/leagues/{league_id}/data/dataset.csv')
        backup_path = backup_root / f'{league_id}-dataset.csv'
        if not backup_path.exists():
            shutil.copy2(source_path, backup_path)

        league = league_db.index[league_id]
        print(f'[{league_id}] 正在从官方配置的数据源重新下载历史比赛…', flush=True)
        dataset = league_db._download_league(league=league, start_year=league.start_year)
        if dataset is None or dataset.empty:
            raise RuntimeError(f'{league_id} 下载失败，原数据仍保留在 {backup_path}')
        if 'HTR' not in dataset.columns or dataset['HTR'].notna().sum() < int(len(dataset) * 0.90):
            raise RuntimeError(f'{league_id} 半场数据不完整，拒绝覆盖原数据。')

        league_db.save_league(dataset, league)
        print(
            f'[{league_id}] 完成：{len(dataset)} 场，'
            f'半场结果 {dataset["HTR"].notna().sum()} 场。',
            flush=True,
        )
