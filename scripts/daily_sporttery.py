"""Command-line entry point for the daily Sporttery report."""

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.daily_sporttery import run_daily_sporttery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-browser', action='store_true')
    parser.add_argument('--output-root', default='storage/jingcai')
    args = parser.parse_args()
    log_path = Path(args.output_root) / 'daily_sporttery.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    predictions, skipped = run_daily_sporttery(
        output_root=Path(args.output_root),
        headless=not args.show_browser,
    )
    print(f'已生成 {len(predictions)} 场预测；另有 {len(skipped)} 场未覆盖。')


if __name__ == '__main__':
    main()
