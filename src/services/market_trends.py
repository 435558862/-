"""Leakage-safe market trend analysis built from official snapshots."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional


OUTCOMES = ('H', 'D', 'A')
OUTCOME_LABELS = {'H': '胜', 'D': '平', 'A': '负'}


def _number(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def implied_probabilities(had: dict) -> Optional[dict]:
    """Return margin-normalized H/D/A probabilities."""
    inverse = {}
    for outcome in OUTCOMES:
        odds = _number((had or {}).get(outcome))
        if odds is None or odds <= 0:
            return None
        inverse[outcome] = 1.0 / odds
    total = sum(inverse.values())
    if total <= 0:
        return None
    return {outcome: inverse[outcome] / total for outcome in OUTCOMES}


def _timestamp(value) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(value or '').replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None


def _ttg_probabilities(ttg: dict):
    try:
        inverse = [1.0 / float(ttg[f's{goals}']) for goals in range(8)]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None, None
    total = sum(inverse)
    if total <= 0:
        return None, None
    under = sum(inverse[:3]) / total
    return 1.0 - under, under


def live_snapshot_from_match(raw: dict, captured_at: str) -> Optional[dict]:
    """Convert one live official fixture row to the stored snapshot shape."""
    match_id = str(raw.get('matchId') or '').strip()
    had = raw.get('had') or {}
    converted_had = {
        'H': _number(had.get('h')),
        'D': _number(had.get('d')),
        'A': _number(had.get('a')),
    }
    if not match_id or implied_probabilities(converted_had) is None:
        return None
    hhad = raw.get('hhad') or {}
    ttg = raw.get('ttg') or {}
    snapshot = {
        'match_id': match_id,
        'captured_at': captured_at,
        'had_update': datetime.now().strftime('%H:%M:%S'),
        'had': converted_had,
    }
    raw_line = hhad.get('goalLine')
    if raw_line in (None, ''):
        raw_line = hhad.get('goalLineValue')
    line = _number(raw_line)
    if line is not None:
        snapshot['hhad'] = {
            'line': line,
            'H': _number(hhad.get('h')),
            'D': _number(hhad.get('d')),
            'A': _number(hhad.get('a')),
        }
    if all(_number(ttg.get(f's{goals}')) is not None for goals in range(8)):
        snapshot['ttg'] = {
            f's{goals}': _number(ttg.get(f's{goals}')) for goals in range(8)
        }
    return snapshot


def build_trend_rows(match_id, series: Dict[str, List[dict]],
                     hours: Optional[int] = None) -> List[dict]:
    """Build chronological chart rows and always retain the opening baseline."""
    source = sorted(
        list(series.get(str(match_id), []) or []),
        key=lambda row: str(row.get('captured_at') or ''),
    )
    opening = source[0] if source else None
    if hours and source:
        latest = _timestamp(source[-1].get('captured_at'))
        if latest is not None:
            cutoff = latest - timedelta(hours=int(hours))
            filtered = [
                row for row in source
                if (stamp := _timestamp(row.get('captured_at'))) is not None
                and stamp >= cutoff
            ]
            # The first value recorded from the official feed is our honest
            # opening baseline. Keep it in every range so a 6/12/24-hour view
            # cannot accidentally compare two late-market points instead.
            if filtered and opening not in filtered:
                filtered.insert(0, opening)
            source = filtered

    result = []
    for snapshot in source:
        probabilities = implied_probabilities(snapshot.get('had') or {})
        if probabilities is None:
            continue
        over, under = _ttg_probabilities(snapshot.get('ttg') or {})
        hhad = snapshot.get('hhad') or {}
        raw_label = str(snapshot.get('market_update')
                        or snapshot.get('had_update')
                        or snapshot.get('captured_at') or '')
        is_opening = snapshot is opening
        result.append({
            'captured_at': str(snapshot.get('captured_at') or ''),
            'label': f'初盘（首次记录） {raw_label}' if is_opening else raw_label,
            'is_opening': is_opening,
            'had_H': _number((snapshot.get('had') or {}).get('H')),
            'had_D': _number((snapshot.get('had') or {}).get('D')),
            'had_A': _number((snapshot.get('had') or {}).get('A')),
            'H': probabilities['H'], 'D': probabilities['D'],
            'A': probabilities['A'],
            'hhad_line': _number(hhad.get('line')),
            'hhad_H': _number(hhad.get('H')),
            'hhad_D': _number(hhad.get('D')),
            'hhad_A': _number(hhad.get('A')),
            'over': over, 'under': under,
        })
    return result


def _reversal_count(rows: Iterable[dict]) -> int:
    favorites = [max(OUTCOMES, key=lambda key: row[key]) for row in rows]
    return sum(left != right for left, right in zip(favorites, favorites[1:]))


def summarize_trend(rows: List[dict]) -> dict:
    """Return one compact, honest conclusion for a market trend."""
    if not rows:
        return {
            'direction': '无数据', 'strength': '不足', 'stability': '无法判断',
            'handicap': '无让球快照', 'total_goals': '无大小球快照',
            'conclusion': '暂无可用赔率快照', 'observations': 0,
        }
    latest = rows[-1]
    latest_key = max(OUTCOMES, key=lambda key: latest[key])
    direction = OUTCOME_LABELS[latest_key]
    movement_key, movement = latest_key, 0.0
    if len(rows) >= 2:
        changes = {key: latest[key] - rows[0][key] for key in OUTCOMES}
        movement_key = max(OUTCOMES, key=changes.get)
        movement = changes[movement_key]
    strength = '强' if movement >= 0.03 else '中' if movement >= 0.015 else '弱'
    reversals = _reversal_count(rows)
    stability = '反复' if reversals >= 2 else '稳定' if len(rows) >= 2 else '待积累'

    first_line = next((row['hhad_line'] for row in rows
                       if row['hhad_line'] is not None), None)
    last_line = next((row['hhad_line'] for row in reversed(rows)
                      if row['hhad_line'] is not None), None)
    if last_line is None:
        handicap = '无让球快照'
    elif first_line != last_line:
        handicap = f'让球 {first_line:g}→{last_line:g}'
    else:
        handicap = f'让球 {last_line:g} 稳定'

    first_over = next((row['over'] for row in rows if row['over'] is not None), None)
    last_over = next((row['over'] for row in reversed(rows)
                      if row['over'] is not None), None)
    if last_over is None:
        total_goals = '无大小球快照'
    elif first_over is not None and last_over - first_over >= 0.02:
        total_goals = '大球升温'
    elif first_over is not None and first_over - last_over >= 0.02:
        total_goals = '小球升温'
    else:
        total_goals = '大小球稳定'

    flow = OUTCOME_LABELS[movement_key] if movement >= 0.005 else '不明确'
    if stability == '反复':
        conclusion = f'盘口反复，当前偏{direction}，等待临场确认'
    elif flow == '不明确':
        conclusion = f'市场当前偏{direction}，但流向不明确'
    else:
        conclusion = f'市场当前偏{direction}，价格流向{flow}（{strength}）'
    return {
        'direction': direction, 'flow': flow, 'strength': strength,
        'stability': stability, 'handicap': handicap,
        'total_goals': total_goals, 'conclusion': conclusion,
        'observations': len(rows),
        'latest_probabilities': {key: latest[key] for key in OUTCOMES},
    }
