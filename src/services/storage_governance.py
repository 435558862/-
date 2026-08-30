"""Conservative retention for reproducible football prediction data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = PROJECT_ROOT / 'storage'


def _older_than(path: Path, cutoff: datetime) -> bool:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime) < cutoff
    except OSError:
        return False


def _candidates(now: datetime) -> Iterable[tuple[Path, str]]:
    """Yield only disposable files; never yield predictions or learning data."""
    thirty_days = now - timedelta(days=30)
    one_day = now - timedelta(days=1)
    disposable_roots = (
        STORAGE_ROOT / 'cache',
        STORAGE_ROOT / 'tmp',
    )
    for root in disposable_roots:
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file() and _older_than(path, one_day):
                    yield path, '可再生缓存/临时文件超过1天'

    lineup_root = STORAGE_ROOT / 'jingcai' / 'lineups'
    if lineup_root.exists():
        for path in lineup_root.glob('catalog-*.json'):
            if _older_than(path, thirty_days):
                yield path, '阵容接口目录超过30天'

    for path in STORAGE_ROOT.rglob('*.log') if STORAGE_ROOT.exists() else ():
        if _older_than(path, thirty_days):
            yield path, '运行日志超过30天'
    for pattern in ('*.tmp', '*.part'):
        for path in STORAGE_ROOT.rglob(pattern) if STORAGE_ROOT.exists() else ():
            if _older_than(path, one_day):
                yield path, '中断产生的临时文件超过1天'


def run_storage_governance(*, dry_run: bool = True, now: datetime | None = None) -> dict:
    """Report or remove disposable data while retaining every audit asset."""
    current = now or datetime.now()
    seen: set[Path] = set()
    candidates = []
    removed = []
    errors = []
    for path, reason in _candidates(current):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            resolved.relative_to(STORAGE_ROOT.resolve())
        except ValueError:
            errors.append(f'越界目标已拒绝：{resolved}')
            continue
        item = {'path': str(resolved), 'reason': reason, 'bytes': path.stat().st_size}
        candidates.append(item)
        if not dry_run:
            try:
                path.unlink()
                removed.append(item)
            except OSError as error:
                errors.append(f'{resolved}: {error}')
    return {
        'dry_run': dry_run,
        'candidate_files': len(candidates),
        'candidate_bytes': sum(item['bytes'] for item in candidates),
        'removed_files': len(removed),
        'removed_bytes': sum(item['bytes'] for item in removed),
        'items': candidates,
        'errors': errors,
        'retained': [
            '官方赛果', '盘口历史', '每日推荐快照', '冻结预测',
            '训练样本', '模型文件', '阵容确认历史',
        ],
    }
