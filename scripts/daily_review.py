"""Scheduled one-shot review for completed Sporttery predictions."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.services.daily_learning import review_and_learn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--full-backfill',
        action='store_true',
        help='补同步一年的官方赛果；日常定时任务无需开启。',
    )
    args = parser.parse_args()

    log_path = PROJECT_ROOT / 'storage' / 'jingcai' / 'daily_review.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    try:
        result = review_and_learn(full_backfill=args.full_backfill)
    except Exception:
        logging.exception('定时赛果复盘失败。')
        return 1

    logging.info(
        '复盘完成：新增结算 %s 场，等待官方赛果 %s 场，累计复盘 %s 场。',
        result.get('newly_settled', 0),
        result.get('pending_samples', 0),
        result.get('settled_samples', 0),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
