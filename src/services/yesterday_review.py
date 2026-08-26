"""Compact, cache-only details for yesterday's settled Sporttery predictions."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REPORT_ROOT = Path('storage/jingcai/reports')
SETTLED_PATH = Path('storage/jingcai/learning/settled_predictions.csv')

DETAIL_COLUMNS = [
    '赛事编号', '比赛时间', '联赛', '主队', '客队', '完场比分', '命中项目',
    '胜负', '让球（首/次）', '大小球', '半全场（首/次）',
    '比分（首/次1/次2/冷/进）', '胜负模型',
    '模拟Top3比分', '模拟胜负', '模拟让球', '模拟总进球',
    '模拟半全场', '模拟可信度', '蒙特风险', '模拟数据状态',
    '模拟模型来源',
]

METRIC_LABELS = {
    'result': '胜负',
    'handicap': '让球首选',
    'over_under': '大小球',
    'half_full': '半全场首选',
    'score': '比分5选',
}


def _normalize_match_id(value) -> str:
    text = str(value or '').strip()
    return text[:-2] if text.endswith('.0') and text[:-2].isdigit() else text


def _text(*values) -> str:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text and text.casefold() != 'nan':
            return text
    return ''


def _number(*values) -> Optional[float]:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _parse_score(value) -> Optional[tuple[int, int]]:
    text = _text(value).replace('：', ':').replace('-', ':')
    parts = [part.strip() for part in text.split(':')]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1])


def _score_text(value) -> str:
    score = _parse_score(value)
    return f'{score[0]}-{score[1]}' if score is not None else ''


def _outcome(home_goals: int, away_goals: int) -> str:
    return '胜' if home_goals > away_goals else '平' if home_goals == away_goals else '负'


def _hit_text(hit: bool) -> str:
    return '命中' if hit else '未中'


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _prediction_rows(settled: pd.DataFrame, report_root: Path) -> dict[str, dict]:
    """Load only the source reports referenced by the requested settled rows."""
    paths: list[Path] = []
    seen = set()
    for _, row in settled.iterrows():
        source = Path(_text(row.get('source_report'))).name
        prediction_day = _text(row.get('prediction_date'))[:10]
        candidates = []
        if source:
            candidates.append(report_root / source)
        if prediction_day:
            candidates.append(report_root / f'{prediction_day}-竞彩预测.csv')
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                paths.append(candidate)

    result: dict[str, dict] = {}
    for path in paths:
        report = _safe_read_csv(path)
        if report.empty or '比赛ID' not in report.columns:
            continue
        for _, row in report.iterrows():
            match_id = _normalize_match_id(row.get('比赛ID'))
            if match_id:
                result[match_id] = row.to_dict()
    return result


def _value(settled: pd.Series, prediction: dict, *columns):
    for column in columns:
        if column in settled:
            value = settled.get(column)
            if _text(value):
                return value
        if column in prediction:
            value = prediction.get(column)
            if _text(value):
                return value
    return None


def _metric(hit: bool, valid: bool) -> dict:
    return {'hit': int(bool(hit)) if valid else 0, 'valid': int(bool(valid))}


def _build_detail(settled: pd.Series, prediction: dict) -> tuple[dict, dict]:
    home_goals = _number(settled.get('home_goals'))
    away_goals = _number(settled.get('away_goals'))
    if home_goals is None or away_goals is None:
        actual_score = _parse_score(settled.get('actual_score'))
        if actual_score is None:
            raise ValueError('missing final score')
        home_goals, away_goals = actual_score
    home_goals, away_goals = int(home_goals), int(away_goals)
    actual_score_text = f'{home_goals}-{away_goals}'
    actual_result = _text(settled.get('actual_result_label')) or _outcome(
        home_goals, away_goals,
    )

    hit_items: list[str] = []
    predicted_result = _text(_value(
        settled, prediction, 'predicted_result', '胜平负首选',
    ))
    result_valid = bool(predicted_result)
    result_hit = result_valid and predicted_result == actual_result
    if result_hit:
        hit_items.append('胜负')
    result_display = (
        f'{predicted_result} → {actual_result}（{_hit_text(result_hit)}）'
        if result_valid else f'— → {actual_result}'
    )

    handicap_line = _number(_value(
        settled, prediction, 'handicap_line', '官方让球数',
    ))
    handicap_first = _text(_value(
        settled, prediction, 'predicted_handicap', '让球首选',
    ))
    handicap_second = _text(_value(
        settled, prediction, 'predicted_handicap_second', '让球次选',
    ))
    handicap_valid = handicap_line is not None and bool(handicap_first)
    actual_handicap = ''
    handicap_first_hit = False
    handicap_second_hit = False
    if handicap_line is not None:
        adjusted = home_goals + handicap_line - away_goals
        actual_handicap = '胜' if adjusted > 1e-9 else '平' if abs(adjusted) <= 1e-9 else '负'
        handicap_first_hit = bool(handicap_first) and handicap_first == actual_handicap
        handicap_second_hit = bool(handicap_second) and handicap_second == actual_handicap
    if handicap_first_hit:
        hit_items.append('让球首选')
    elif handicap_second_hit:
        hit_items.append('让球次选')
    if handicap_line is None:
        handicap_display = '未开盘'
    else:
        line_text = f'{handicap_line:+g}'
        picks = f'首{handicap_first or "—"}'
        if handicap_second:
            picks += f'/次{handicap_second}'
        status = '首中' if handicap_first_hit else '次中' if handicap_second_hit else '未中'
        handicap_display = f'[{line_text}] {picks} → 让{actual_handicap}（{status}）'

    predicted_ou = _text(_value(
        settled, prediction, 'predicted_over_under', '大小球首选',
    ))
    actual_ou = '大于2.5球' if home_goals + away_goals > 2 else '小于2.5球'
    ou_valid = bool(predicted_ou)
    ou_hit = ou_valid and predicted_ou == actual_ou
    if ou_hit:
        hit_items.append('大小球')
    short_ou = {'大于2.5球': '大', '小于2.5球': '小'}
    ou_display = (
        f'{short_ou.get(predicted_ou, predicted_ou)} → '
        f'{short_ou[actual_ou]}（{_hit_text(ou_hit)}）'
        if ou_valid else f'— → {short_ou[actual_ou]}'
    )

    half_score = _parse_score(_value(
        settled, prediction, 'official_half_score',
    ))
    actual_half_full = ''
    if half_score is not None:
        actual_half_full = _outcome(*half_score) + actual_result
    half_first = _text(_value(
        settled, prediction, 'predicted_half_full', '半全场首选',
    ))
    half_second = _text(_value(
        settled, prediction, 'predicted_half_full_second', '半全场次选',
    ))
    half_valid = bool(actual_half_full and half_first)
    half_first_hit = half_valid and half_first == actual_half_full
    half_second_hit = bool(actual_half_full and half_second) and half_second == actual_half_full
    if half_first_hit:
        hit_items.append('半全场首选')
    elif half_second_hit:
        hit_items.append('半全场次选')
    if not actual_half_full:
        half_display = '半场赛果缺失'
    else:
        picks = f'首{half_first or "—"}'
        if half_second:
            picks += f'/次{half_second}'
        status = '首中' if half_first_hit else '次中' if half_second_hit else '未中'
        half_display = f'{picks} → {actual_half_full}（{status}）'

    score_fields = [
        ('首', ('predicted_score', '首选比分')),
        ('次1', ('predicted_score_second', '次选比分')),
        ('次2', ('predicted_score_third', '第三比分')),
        ('冷', ('predicted_score_upset', '比分爆冷', '爆冷比分')),
        ('进', ('predicted_score_aggressive', '大小球进取比分')),
    ]
    scores: list[tuple[str, str]] = []
    seen_scores = set()
    for label, columns in score_fields:
        score = _score_text(_value(settled, prediction, *columns))
        if score and score not in seen_scores:
            scores.append((label, score))
            seen_scores.add(score)
    score_hit_source = next(
        (label for label, score in scores if score == actual_score_text), '',
    )
    score_valid = bool(scores)
    score_hit = bool(score_hit_source)
    if score_hit:
        hit_items.append(f'比分{score_hit_source}')
    score_picks = '/'.join(score for _, score in scores) or '—'
    score_status = f'{score_hit_source}中' if score_hit else '未中'
    score_display = f'{score_picks} → {actual_score_text}（{score_status}）'

    monte_top3 = _text(_value(settled, prediction, '模拟Top3比分'))
    monte_result = _text(_value(settled, prediction, '模拟胜负'))
    monte_handicap = _text(_value(settled, prediction, '模拟让球'))
    monte_total = _text(_value(settled, prediction, '模拟总进球'))
    monte_half_full = _text(_value(settled, prediction, '模拟半全场'))
    monte_confidence = _text(_value(settled, prediction, '模拟可信度'))
    monte_risk = _text(_value(settled, prediction, '蒙特风险'))
    has_raw_monte = any((
        monte_top3, monte_result, monte_handicap, monte_total,
        monte_half_full, monte_confidence, monte_risk,
    ))
    if not has_raw_monte:
        monte_top3 = _text(_value(settled, prediction, '最可能比分Top3'))
        result_probability = _number(_value(
            settled, prediction, '胜平负首选概率',
        ))
        monte_result = predicted_result
        if result_probability is not None:
            monte_result = f'{monte_result} {result_probability:.1%}'
        handicap_probability = _number(_value(
            settled, prediction, '让球首选概率',
        ))
        monte_handicap = handicap_first
        if handicap_probability is not None:
            monte_handicap = f'{monte_handicap} {handicap_probability:.1%}'
        monte_total = predicted_ou
        monte_half_full = _text(_value(
            settled, prediction, '半全场Top3', '半全场首选',
        ))
        monte_confidence = _text(_value(settled, prediction, '置信等级'))
        monte_risk = '历史文件未保存蒙特风险'
    if monte_half_full:
        monte_half_full = ' / '.join(
            part.strip() for part in monte_half_full.split('/')[:2] if part.strip()
        )
    monte_state = '原始蒙特记录' if has_raw_monte else '历史预测字段回填'

    def marked(value: str, hit: bool) -> str:
        if not value:
            return ''
        return f'{value}（{"命中" if hit else "未中"}）'

    actual_goals = home_goals + away_goals
    monte_score_hit = bool(monte_top3 and actual_score_text in monte_top3)
    monte_result_hit = bool(
        monte_result and monte_result.lstrip().startswith(actual_result),
    )
    monte_handicap_hit = bool(
        monte_handicap and actual_handicap
        and monte_handicap.lstrip().startswith(actual_handicap),
    )
    monte_total_hit = bool(
        monte_total and (
            actual_ou in monte_total
            or f'{actual_goals}球' in monte_total
            or (actual_goals >= 4 and '4球以上' in monte_total)
            or (actual_goals <= 1 and '1球以内' in monte_total)
        )
    )
    monte_half_full_hit = bool(
        monte_half_full and actual_half_full and actual_half_full in monte_half_full,
    )

    match_time = _text(_value(
        settled, prediction, 'match_time', '比赛时间', 'match_date',
    ))
    detail = {
        '赛事编号': _text(_value(settled, prediction, 'match_number', '赛事编号')),
        '比赛时间': match_time,
        '联赛': _text(_value(settled, prediction, 'league', '联赛')),
        '主队': _text(_value(settled, prediction, 'home', '主队')),
        '客队': _text(_value(settled, prediction, 'away', '客队')),
        '完场比分': actual_score_text,
        '命中项目': '、'.join(hit_items) if hit_items else '无',
        '胜负': result_display,
        '让球（首/次）': handicap_display,
        '大小球': ou_display,
        '半全场（首/次）': half_display,
        '比分（首/次1/次2/冷/进）': score_display,
        '胜负模型': _text(_value(
            settled, prediction, 'model_category', '胜负模型类别', '模型类别',
        )),
        '模拟Top3比分': marked(monte_top3, monte_score_hit),
        '模拟胜负': marked(monte_result, monte_result_hit),
        '模拟让球': marked(monte_handicap, monte_handicap_hit),
        '模拟总进球': marked(monte_total, monte_total_hit),
        '模拟半全场': marked(monte_half_full, monte_half_full_hit),
        '模拟可信度': monte_confidence,
        '蒙特风险': monte_risk,
        '模拟数据状态': monte_state,
        '模拟模型来源': _text(_value(
            settled, prediction, '模拟模型来源',
        )) or ('历史原模型字段回填' if not has_raw_monte else '旧版同分布模拟'),
    }
    internal = {
        '_result_pick': predicted_result,
        '_model': detail['胜负模型'],
        '_league': detail['联赛'],
        '_result_hit': int(result_hit),
        '_valid_total': sum((
            int(result_valid), int(handicap_valid), int(ou_valid),
            int(half_valid), int(score_valid),
        )),
        '_hit_total': sum((
            int(result_hit), int(handicap_first_hit), int(ou_hit),
            int(half_first_hit), int(score_hit),
        )),
        '_metrics': {
            'result': _metric(result_hit, result_valid),
            'handicap': _metric(handicap_first_hit, handicap_valid),
            'over_under': _metric(ou_hit, ou_valid),
            'half_full': _metric(half_first_hit, half_valid),
            'score': _metric(score_hit, score_valid),
        },
        '_score_hit_source': score_hit_source,
    }
    return detail, internal


def _metric_summary(rows: list[dict]) -> dict:
    summary = {}
    for key, label in METRIC_LABELS.items():
        hits = sum(row['_metrics'][key]['hit'] for row in rows)
        valid = sum(row['_metrics'][key]['valid'] for row in rows)
        summary[key] = {
            'label': label,
            'hits': hits,
            'valid': valid,
            'accuracy': hits / valid if valid else None,
        }
    return summary


def _format_metric(metric: dict) -> str:
    if not metric['valid']:
        return f'{metric["label"]} --'
    return (
        f'{metric["label"]} {metric["hits"]}/{metric["valid"]} '
        f'{metric["accuracy"]:.1%}'
    )


def _patterns(rows: list[dict], metrics: dict) -> str:
    if not rows:
        return ''
    eligible = [metric for metric in metrics.values() if metric['valid'] >= 2]
    parts = []
    if eligible:
        best = max(eligible, key=lambda item: (item['accuracy'], item['valid']))
        parts.append(
            f'昨日相对最稳：{best["label"]} '
            f'{best["hits"]}/{best["valid"]}（{best["accuracy"]:.1%}）'
        )

    direction_rows = [row for row in rows if row['_result_pick']]
    direction_stats = []
    for direction in ('胜', '平', '负'):
        group = [row for row in direction_rows if row['_result_pick'] == direction]
        if len(group) >= 2:
            hits = sum(row['_result_hit'] for row in group)
            direction_stats.append((hits / len(group), len(group), hits, direction))
    if direction_stats:
        accuracy, samples, hits, direction = max(direction_stats)
        parts.append(f'{direction}方向较好 {hits}/{samples}（{accuracy:.1%}）')

    score_sources = [row['_score_hit_source'] for row in rows if row['_score_hit_source']]
    if score_sources:
        counts = pd.Series(score_sources).value_counts()
        parts.append(f'比分命中主要来自{counts.index[0]}选（{int(counts.iloc[0])}场）')

    if not parts:
        parts.append('昨日有效样本较少，暂不提炼方向规律')
    parts.append('单日结果只用于复盘，模型启停仍以滚动实战为准')
    return '；'.join(parts) + '。'


def load_yesterday_hit_report(
        today: Optional[date] = None,
        settled_path: Optional[Path] = None,
        report_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """Return yesterday's settled details without performing any network request."""
    today = today or date.today()
    target_day = today - timedelta(days=1)
    settled_path = Path(settled_path or SETTLED_PATH)
    report_root = Path(report_root or REPORT_ROOT)
    settled = _safe_read_csv(settled_path)
    empty_summary = {
        'date': target_day.isoformat(),
        'requested_date': target_day.isoformat(),
        'is_fallback': False,
        'settled': 0,
        'matches_with_hits': 0,
        'metrics': _metric_summary([]),
        'headline': f'{target_day.isoformat()} 暂无已结算赛果，请先点“补赛果并复盘”。',
        'patterns': '',
    }
    if settled.empty or 'match_date' not in settled.columns:
        return pd.DataFrame(columns=DETAIL_COLUMNS), empty_summary

    # Parse row by row: pandas' strict vector parser can otherwise reject a
    # valid timestamp when the same column mixes YYYY-MM-DD and date-time text.
    match_dates = settled['match_date'].map(
        lambda value: pd.to_datetime(value, errors='coerce'),
    ).map(lambda value: value.date() if pd.notna(value) else None)
    display_day = target_day
    is_fallback = False
    selected = settled.loc[match_dates.eq(target_day)].copy()
    if selected.empty:
        available = sorted({
            value for value in match_dates
            if value is not None and value < today
        })
        if not available:
            return pd.DataFrame(columns=DETAIL_COLUMNS), empty_summary
        display_day = available[-1]
        is_fallback = True
        selected = settled.loc[match_dates.eq(display_day)].copy()
    settled = selected

    if 'match_id' not in settled.columns:
        settled['match_id'] = ''
    settled['_match_id'] = settled['match_id'].map(_normalize_match_id)
    sort_columns = [
        column for column in ('settled_at', 'prediction_date') if column in settled.columns
    ]
    if sort_columns:
        settled = settled.sort_values(sort_columns, kind='stable')
    nonempty_ids = settled['_match_id'].ne('')
    with_ids = settled.loc[nonempty_ids].drop_duplicates('_match_id', keep='last')
    settled = pd.concat([with_ids, settled.loc[~nonempty_ids]], ignore_index=True)

    predictions = _prediction_rows(settled, report_root)
    details: list[dict] = []
    internal_rows: list[dict] = []
    for _, row in settled.iterrows():
        prediction = predictions.get(row['_match_id'], {})
        try:
            detail, internal = _build_detail(row, prediction)
        except ValueError:
            continue
        details.append(detail)
        internal_rows.append(internal)

    if not details:
        return pd.DataFrame(columns=DETAIL_COLUMNS), empty_summary
    detail_frame = pd.DataFrame(details, columns=DETAIL_COLUMNS)
    kickoff = pd.to_datetime(detail_frame['比赛时间'], errors='coerce')
    detail_frame = detail_frame.assign(_kickoff=kickoff).sort_values(
        ['_kickoff', '赛事编号'], kind='stable', na_position='last',
    ).drop(columns='_kickoff').reset_index(drop=True)

    metrics = _metric_summary(internal_rows)
    fallback_prefix = (
        f'{target_day.isoformat()}赛果尚未补齐｜显示最近已结算日 '
        if is_fallback else ''
    )
    headline = (
        fallback_prefix
        + f'{display_day.isoformat()} 已结算 {len(detail_frame)} 场｜'
        + '｜'.join(_format_metric(metrics[key]) for key in METRIC_LABELS)
    )
    summary = {
        'date': display_day.isoformat(),
        'requested_date': target_day.isoformat(),
        'is_fallback': is_fallback,
        'settled': len(detail_frame),
        'matches_with_hits': sum(row['_hit_total'] > 0 for row in internal_rows),
        'metrics': metrics,
        'headline': headline,
        'patterns': _patterns(internal_rows, metrics),
    }
    return detail_frame, summary
