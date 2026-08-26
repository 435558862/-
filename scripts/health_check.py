"""Read-only release health check suitable for customer support."""

import importlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.version import PRODUCT_TITLE

REQUIRED_IMPORTS = (
    'numpy', 'pandas', 'scipy', 'sklearn', 'PySide6', 'matplotlib',
    'openpyxl', 'requests', 'selenium', 'xgboost',
)


def run() -> tuple[list[str], list[str]]:
    ok, problems = [], []
    ok.append(f'产品：{PRODUCT_TITLE}')
    ok.append(f'系统：{platform.system()} {platform.release()}')
    ok.append(f'Python：{platform.python_version()}')
    if sys.version_info[:2] != (3, 11):
        problems.append('需要Python 3.11')
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as error:
            problems.append(f'依赖不可用：{name}（{error}）')
    for relative in ('storage/leagues', 'storage/network', 'storage/graphics'):
        path = ROOT / relative
        (ok if path.exists() else problems).append(
            f'数据目录：{relative}' if path.exists() else f'缺少目录：{relative}'
        )
    storage = ROOT / 'storage'
    try:
        probe = storage / '.write-test'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        ok.append('storage可写')
    except OSError as error:
        problems.append(f'storage不可写：{error}')
    model_count = len(list((storage / 'leagues').glob('*/models/*/classifier.pkl')))
    (ok if model_count else problems).append(
        f'已发现模型：{model_count}个' if model_count else '未发现模型文件'
    )
    key_present = bool(os.environ.get('API_FOOTBALL_KEY')) or (
        storage / 'network' / '.api_football_key'
    ).exists()
    ok.append(f'阵容接口：{"已配置" if key_present else "未配置（阵容功能降级）"}')
    return ok, problems


if __name__ == '__main__':
    passed, failed = run()
    print(json.dumps({'通过': passed, '问题': failed}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failed else 0)
