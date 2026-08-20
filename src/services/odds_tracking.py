"""Append-only official odds snapshots for later drift analysis.

Each daily sync stores one compact observation per match: the latest HAD and
HHAD fixed bonuses plus their official update timestamps. Over days this
builds a chronological odds series per match, letting future work measure
open-to-close drift (our first recorded odds act as the opening baseline,
because the official feed only publishes current values).

The recorder is strictly side-effect-light: it must never raise into the
prediction pipeline.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

HISTORY_PATH = Path('storage/jingcai/odds_history.jsonl')


def _float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_observation(raw: dict) -> Optional[dict]:
    """Build one compact odds observation from a raw official match row."""
    match_id = raw.get('matchId')
    if match_id in (None, ''):
        return None
    had = raw.get('had') or {}
    home_odds = _float(had.get('h'))
    draw_odds = _float(had.get('d'))
    away_odds = _float(had.get('a'))
    if None in (home_odds, draw_odds, away_odds):
        return None
    had_update = f"{had.get('updateDate', '')} {had.get('updateTime', '')}".strip()
    hhad = raw.get('hhad') or {}
    line = _float(hhad.get('goalLine') or hhad.get('goalLineValue'))
    observation = {
        'match_id': str(match_id),
        'match_num': str(raw.get('matchNumStr') or ''),
        'league': str(raw.get('leagueAllName') or raw.get('leagueAbbName') or ''),
        'home': str(raw.get('homeTeamAllName') or ''),
        'away': str(raw.get('awayTeamAllName') or ''),
        'had': {'H': home_odds, 'D': draw_odds, 'A': away_odds},
        'had_update': had_update,
    }
    if line is not None:
        observation['hhad'] = {
            'line': line,
            'H': _float(hhad.get('h')),
            'D': _float(hhad.get('d')),
            'A': _float(hhad.get('a')),
        }
        observation['hhad_update'] = (
            f"{hhad.get('updateDate', '')} {hhad.get('updateTime', '')}".strip()
        )
    ttg = raw.get('ttg') or {}
    ttg_odds = {f's{i}': _float(ttg.get(f's{i}')) for i in range(8)}
    if all(value is not None for value in ttg_odds.values()):
        observation['ttg'] = ttg_odds
        observation['ttg_update'] = (
            f"{ttg.get('updateDate', '')} {ttg.get('updateTime', '')}".strip()
        )
    return observation


def _latest_key(observation: dict) -> tuple:
    hhad = observation.get('hhad') or {}
    ttg = observation.get('ttg') or {}
    return (
        observation['had']['H'], observation['had']['D'], observation['had']['A'],
        observation.get('had_update', ''),
        hhad.get('line'), hhad.get('H'), hhad.get('D'), hhad.get('A'),
        ttg.get('s0'), ttg.get('s1'), ttg.get('s2'), ttg.get('s3'),
        ttg.get('s4'), ttg.get('s5'), ttg.get('s6'), ttg.get('s7'),
    )


def record_odds_snapshots(
        matches: List[dict],
        path: Path = HISTORY_PATH,
        captured_at: Optional[str] = None,
) -> int:
    """Append fresh odds observations; skip rows identical to the last stored.

    Returns the number of appended observations. Never raises.
    """
    try:
        observations = []
        for raw in matches or []:
            observation = _extract_observation(raw)
            if observation is not None:
                observations.append(observation)
        if not observations:
            return 0

        previous: Dict[str, tuple] = {}
        if path.exists() and path.stat().st_size > 0:
            with path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        stored = json.loads(line)
                    except ValueError:
                        continue
                    previous[str(stored.get('match_id', ''))] = _latest_key(stored)

        captured_at = captured_at or datetime.now(timezone.utc).isoformat(
            timespec='seconds',
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        appended = 0
        with path.open('a', encoding='utf-8') as handle:
            for observation in observations:
                if previous.get(observation['match_id']) == _latest_key(observation):
                    continue
                observation = {'captured_at': captured_at, **observation}
                handle.write(json.dumps(observation, ensure_ascii=False) + '\n')
                appended += 1
        return appended
    except Exception:
        logging.exception('赔率快照记录失败，忽略。')
        return 0


def read_odds_series(path: Path = HISTORY_PATH) -> Dict[str, List[dict]]:
    """Return chronological odds observations grouped by match id."""
    series: Dict[str, List[dict]] = {}
    if not path.exists():
        return series
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                stored = json.loads(line)
            except ValueError:
                continue
            series.setdefault(str(stored.get('match_id', '')), []).append(stored)
    for rows in series.values():
        rows.sort(key=lambda row: str(row.get('captured_at', '')))
    return series


def drift_summary(match_id: str, path: Path = HISTORY_PATH) -> Optional[dict]:
    """Summarize first-vs-latest recorded HAD odds for one match."""
    rows = read_odds_series(path).get(str(match_id), [])
    if not rows:
        return None
    first, last = rows[0], rows[-1]
    return {
        'observations': len(rows),
        'first_captured_at': first.get('captured_at', ''),
        'last_captured_at': last.get('captured_at', ''),
        'first_had': first['had'],
        'last_had': last['had'],
        'drift': {
            key: round(last['had'][key] - first['had'][key], 3)
            for key in ('H', 'D', 'A')
        },
    }


def format_match_drift(match_id, path: Path = HISTORY_PATH,
                       series: Optional[Dict[str, List[dict]]] = None) -> str:
    """Render first-vs-latest recorded HAD drift as a compact Chinese label."""
    if series is None:
        series = read_odds_series(path)
    rows = series.get(str(match_id), [])
    if not rows:
        return ''
    if len(rows) == 1:
        return '首次记录'
    labels = {'H': '主', 'D': '平', 'A': '负'}
    first, last = rows[0]['had'], rows[-1]['had']
    parts = []
    for key in ('H', 'D', 'A'):
        difference = round(last[key] - first[key], 2)
        if difference > 0:
            parts.append(f'{labels[key]}↑{difference:g}')
        elif difference < 0:
            parts.append(f'{labels[key]}↓{abs(difference):g}')
    return '赔率平稳' if not parts else '·'.join(parts)


def intent_for_match(match_id, odds, path: Path = HISTORY_PATH,
                     series: Optional[Dict[str, List[dict]]] = None) -> str:
    """Combine current odds with recorded first-to-last drift into one label."""
    if series is None:
        series = read_odds_series(path)
    rows = series.get(str(match_id), [])
    drift = None
    if len(rows) >= 2:
        first, last = rows[0]['had'], rows[-1]['had']
        drift = {key: last[key] - first[key] for key in ('H', 'D', 'A')}
    return market_intent_label(odds, drift)


def market_intent_label(odds, drift=None) -> str:
    """Translate official odds level plus recorded drift into a plain label."""
    if not odds:
        return ''
    try:
        inverse = {key: 1.0 / float(odds[key]) for key in ('H', 'D', 'A')}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return ''
    total = sum(inverse.values())
    if total <= 0:
        return ''
    probabilities = {key: value / total for key, value in inverse.items()}
    favorite = max(probabilities, key=probabilities.__getitem__)
    max_probability = probabilities[favorite]
    labels = {'H': '主', 'D': '平', 'A': '客'}
    if max_probability >= 0.60:
        level = f'市场明确看好{labels[favorite]}'
    elif max_probability >= 0.50:
        level = f'市场偏向{labels[favorite]}'
    else:
        level = '市场三方格局'

    if not drift:
        return f'{level}·暂无变动'
    shortened = [
        key for key in ('H', 'D', 'A') if (drift.get(key) or 0) <= -0.05
    ]
    if not shortened:
        if all(abs(drift.get(key) or 0) < 0.03 for key in ('H', 'D', 'A')):
            return f'{level}·暂无变动'
        return f'{level}·赔率微调'
    target = labels[shortened[0]]
    if max_probability >= 0.50:
        if shortened[0] == favorite:
            return f'{level}·变动也挺热门'
        return f'{level}·变动向{target}，防冷'
    return f'{level}·变动向{target}'


def snapshot_had_direction(match_id, path: Path = HISTORY_PATH,
                           series: Optional[Dict[str, List[dict]]] = None) -> str:
    """Return the 胜平负 direction the latest snapshot odds favor.

    Uses the most recent recorded official HAD odds for the match and picks
    the outcome with the highest implied probability.
    """
    if series is None:
        series = read_odds_series(path)
    rows = series.get(str(match_id), [])
    if not rows:
        return ''
    had = rows[-1].get('had') or {}
    try:
        inverse = {key: 1.0 / float(had[key]) for key in ('H', 'D', 'A')}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return ''
    total = sum(inverse.values())
    if total <= 0:
        return ''
    probabilities = {key: value / total for key, value in inverse.items()}
    favorite = max(probabilities, key=probabilities.__getitem__)
    labels = {'H': '主胜', 'D': '平局', 'A': '客胜'}
    return f'{labels[favorite]}（{probabilities[favorite]:.1%}）'


def odds_early_warning(match_id, model_pick: str = '',
                       path: Path = HISTORY_PATH,
                       series: Optional[Dict[str, List[dict]]] = None) -> str:
    """Early-warning from snapshot odds direction vs the model pick.

    Uses the latest recorded official HAD odds to derive the market's favored
    胜平负 direction and its implied draw probability. Flags:
      * 防平       when the market gives the draw 30%+ (the 胜平负 model
                   almost never picks a draw, so this is a cold-draw warning);
      * 盘口反向   when the market direction differs from the model pick;
      * 平赔走低   when draw odds have shortened across recorded snapshots.
    """
    if series is None:
        series = read_odds_series(path)
    rows = series.get(str(match_id), [])
    if not rows:
        return '无快照'
    had = rows[-1].get('had') or {}
    try:
        inverse = {key: 1.0 / float(had[key]) for key in ('H', 'D', 'A')}
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return '无快照'
    total = sum(inverse.values())
    if total <= 0:
        return '无快照'
    probabilities = {key: value / total for key, value in inverse.items()}
    favorite = max(probabilities, key=probabilities.__getitem__)
    labels = {'H': '主胜', 'D': '平局', 'A': '客胜'}
    short = {'主胜': '胜', '平局': '平', '客胜': '负'}
    direction = labels[favorite]
    base = f'{direction} {probabilities[favorite]:.0%}'

    warnings = []
    draw_probability = probabilities['D']
    if draw_probability >= 0.30:
        warnings.append(f'防平（盘口平{draw_probability:.0%}）')
    if model_pick and short[direction] != model_pick:
        warnings.append(f'盘口{direction}，与模型{model_pick}相反')
    if len(rows) >= 2:
        try:
            draw_drift = float(rows[-1]['had']['D']) - float(rows[0]['had']['D'])
        except (KeyError, TypeError, ValueError):
            draw_drift = 0.0
        if draw_drift <= -0.05:
            warnings.append(f'平赔走低{abs(draw_drift):.2f}，防平')
    if not warnings:
        return base
    return f'{base}｜' + '；'.join(warnings)


def _ttg_over_under(ttg: dict):
    """Return (under, over) probabilities for 2.5 from official ttg odds."""
    try:
        values = [float(ttg[f's{i}']) for i in range(8)]
    except (KeyError, TypeError, ValueError):
        return None
    if any(value <= 1.0 for value in values):
        return None
    inverse = [1.0 / value for value in values]
    total = sum(inverse)
    if total <= 0:
        return None
    under = sum(inverse[:3]) / total
    return under, 1.0 - under


def format_handicap_drift(match_id, path: Path = HISTORY_PATH,
                          series: Optional[Dict[str, List[dict]]] = None) -> str:
    """Render first-vs-latest handicap line/water drift as a Chinese label."""
    if series is None:
        series = read_odds_series(path)
    rows = [row for row in series.get(str(match_id), []) if row.get('hhad')]
    if not rows:
        return ''
    first, last = rows[0]['hhad'], rows[-1]['hhad']
    first_line = _float(first.get('line'))
    last_line = _float(last.get('line'))
    label = f'让球线 {last_line:g}' if last_line is not None else ''
    if first_line is not None and last_line is not None and first_line != last_line:
        label = f'让球线 {first_line:g}→{last_line:g}'
    if len(rows) == 1:
        return f'{label}·首次记录'
    labels = {'H': '让胜', 'D': '让平', 'A': '让负'}
    parts = []
    for key in ('H', 'D', 'A'):
        old = _float(first.get(key))
        new = _float(last.get(key))
        if old is None or new is None:
            continue
        difference = round(new - old, 2)
        if difference >= 0.03:
            parts.append(f'{labels[key]}↑{difference:g}')
        elif difference <= -0.03:
            parts.append(f'{labels[key]}↓{abs(difference):g}')
    return label + ('·' + '·'.join(parts) if parts else '·水位平稳')


def format_ou_drift(match_id, path: Path = HISTORY_PATH,
                    series: Optional[Dict[str, List[dict]]] = None) -> str:
    """Render first-vs-latest official total-goals market drift (O/U 2.5)."""
    if series is None:
        series = read_odds_series(path)
    rows = [row for row in series.get(str(match_id), []) if row.get('ttg')]
    if not rows:
        return ''
    first = _ttg_over_under(rows[0]['ttg'])
    last = _ttg_over_under(rows[-1]['ttg'])
    if first is None or last is None:
        return ''
    first_under, first_over = first
    last_under, last_over = last
    if len(rows) == 1 or (
        abs(first_under - last_under) < 0.005
        and abs(first_over - last_over) < 0.005
    ):
        return f'大小球盘口 小{last_under:.1%}'
    return f'大小球盘口 小{first_under:.1%}→{last_under:.1%}'


def situation_summary(match_id, odds: Optional[dict] = None,
                      path: Path = HISTORY_PATH,
                      series: Optional[Dict[str, List[dict]]] = None) -> str:
    """Combine HAD drift, handicap line, O/U drift and bookmaker intent."""
    if series is None:
        series = read_odds_series(path)
    parts = []
    had = format_match_drift(match_id, series=series)
    if had:
        parts.append(f'胜平负{had}')
    handicap = format_handicap_drift(match_id, series=series)
    if handicap:
        parts.append(handicap)
    ou = format_ou_drift(match_id, series=series)
    if ou:
        parts.append(ou)
    if odds:
        intent = intent_for_match(match_id, odds, series=series)
        if intent:
            parts.append(intent)
    return '；'.join(parts)
