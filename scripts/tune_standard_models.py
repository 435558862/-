import os

import pandas as pd

from src.preprocessing.utils.target import TargetType
from scripts.tune_advanced_models import LEAGUES, tune_one


if __name__ == '__main__':
    reports = []
    for league_id in LEAGUES:
        reports.append(tune_one(league_id, '胜平负', TargetType.RESULT))
        reports.append(tune_one(league_id, '大小球', TargetType.OVER_UNDER))

    report_df = pd.DataFrame([row for row in reports if row is not None])
    os.makedirs('storage/reports', exist_ok=True)
    report_df.to_csv('storage/reports/胜平负大小球算法对比报告.csv', index=False)
    print('\n' + report_df.to_string(index=False), flush=True)
