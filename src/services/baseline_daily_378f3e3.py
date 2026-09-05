"""Frozen daily selection rule from Git 378f3e3, shadow evaluation only.

Never used to populate live recommendations or place bets.
"""
from __future__ import annotations
import re
from datetime import datetime
import numpy as np
import pandas as pd
from src.services.baseline_value_378f3e3 import evaluate_value, historical_calibration
from src.gui.windows.sporttery import _upcoming_predictions, _sort_by_match_number, _ticket_card_date


def build_daily_recommendations(
        predictions: pd.DataFrame, future_only: bool = True,
) -> pd.DataFrame:
    """Select at most five tiered fixtures for each visible lottery card day.

    The formal model owns the pick.  Market movement and the independent
    Monte Carlo model are vetoes, never alternative sources of a pick.  Exact
    score and half/full remain visible in the main table but are deliberately
    excluded from the high-hit-rate daily list.
    """
    columns = [
        '相对安全等级', '行动结论',
        '比赛日期', '赛事编号', '联赛', '对阵', '推荐玩法', '重点选项',
        '最佳比分', '每日2串1', '2串1组合概率', '2串1组合SP',
        '推荐等级', '推荐性质', '正式主模型', '数据状态',
        '正式模型概率', '价值评估', '建议仓位',
        '盘口验证', '蒙特卡洛是否同向', '阵容验证',
        '比分参考', '半全场参考', '入选理由',
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    source = _upcoming_predictions(predictions) if future_only else predictions.copy()
    active = _sort_by_match_number(source).reset_index(drop=True)
    allow_observation_conflicts = len(active) >= 3

    def number(row: pd.Series, column: str) -> float:
        value = pd.to_numeric(row.get(column), errors='coerce')
        return float(value) if pd.notna(value) else float('nan')

    def first_simulation_pick(row: pd.Series, market: str) -> str:
        column = {
            '胜平负': '模拟胜负', '让球胜平负': '模拟让球',
        }[market]
        text = str(row.get(column) or '').strip()
        return re.split(r'[\s/｜]', text, maxsplit=1)[0]

    def implied(row: pd.Series, columns_: tuple[str, str, str]) -> np.ndarray | None:
        odds = np.array([number(row, column) for column in columns_], dtype=float)
        if not np.isfinite(odds).all() or np.any(odds <= 1.0):
            return None
        inverse = 1.0 / odds
        return inverse / inverse.sum()

    def formal_margin(row: pd.Series, columns_: tuple[str, str, str]) -> float:
        values = np.array([number(row, column) for column in columns_], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return 0.0
        values.sort()
        return float(values[-1] - values[-2])

    def probability_text(row: pd.Series, column: str) -> str:
        value = number(row, column)
        return f'{value:.1%}' if np.isfinite(value) else ''

    def score_reference(row: pd.Series) -> str:
        picks = []
        for pick_column, probability_column in (
                ('首选比分', '首选比分概率'),
                ('次选比分', '次选比分概率'),
                ('第三比分', '第三比分概率'),
        ):
            pick = str(row.get(pick_column) or '').strip()
            if not pick or pick.lower() == 'nan' or pick in picks:
                continue
            probability = probability_text(row, probability_column)
            picks.append(f'{pick}（{probability}）' if probability else pick)
        return ' / '.join(picks[:3]) or '暂无可靠比分'

    def best_score(row: pd.Series) -> str:
        pick = str(row.get('首选比分') or '').strip()
        if not pick or pick.lower() == 'nan':
            return '—'
        probability = number(row, '首选比分概率')
        return (
            f'◎ {pick}（{probability:.1%}）'
            if np.isfinite(probability) else f'◎ {pick}'
        )

    def high_odds_reference(row: pd.Series) -> str:
        """Return one auditable high-price candidate, never a core pick.

        A high SP alone is not enough. The formal probability must leave at
        least a small raw edge, while a conservative probability haircut may
        not make the candidate severely negative. This intentionally yields
        no candidate on many fixtures rather than manufacturing a long shot.
        """
        candidates = []
        fallback_candidates = []

        def consider(
                market: str, label: str, probability_column: str,
                odds_column: str, haircut: float,
        ) -> None:
            probability = number(row, probability_column)
            odds = number(row, odds_column)
            if (
                not np.isfinite(probability) or not np.isfinite(odds)
                or probability < 0.18 or odds < 2.80
            ):
                return
            raw_ev = probability * odds - 1.0
            conservative_ev = max(0.01, probability - haircut) * odds - 1.0
            fallback_candidates.append((raw_ev, probability, odds, market, label))
            if raw_ev < 0.05 or conservative_ev < -0.08:
                return
            candidates.append((
                conservative_ev, raw_ev, probability, odds, market, label,
            ))

        for label, probability_column, odds_column in zip(
                ('胜', '平', '负'),
                ('模型主胜概率', '模型平局概率', '模型客胜概率'),
                ('官方胜奖金', '官方平奖金', '官方负奖金')):
            consider('胜平负', label, probability_column, odds_column, 0.025)
        line = number(row, '官方让球数')
        if np.isfinite(line):
            for label, probability_column, odds_column in zip(
                    ('让胜', '让平', '让负'),
                    ('模型让胜概率', '模型让平概率', '模型让负概率'),
                    ('官方让胜奖金', '官方让平奖金', '官方让负奖金')):
                consider(
                    '让球', f'{line:+g}球 {label}',
                    probability_column, odds_column, 0.040,
                )
        if not candidates:
            if not fallback_candidates:
                return '—'
            raw_ev, probability, odds, market, label = max(
                fallback_candidates, key=lambda item: (item[0], item[1]),
            )
            return (
                f'◇ {market}·{label}（SP {odds:.2f}｜模型{probability:.1%}｜'
                f'理论EV {raw_ev:+.1%}｜高风险观察）'
            )
        _, raw_ev, probability, odds, market, label = max(candidates)
        return (
            f'◆ {market}·{label}（SP {odds:.2f}｜模型{probability:.1%}｜'
            f'优势{raw_ev:+.1%}｜高风险）'
        )

    def half_full_reference(row: pd.Series) -> str:
        picks = []
        for pick_column, probability_column in (
                ('半全场首选', '半全场首选概率'),
                ('半全场次选', '半全场次选概率'),
        ):
            pick = str(row.get(pick_column) or '').strip()
            if not pick or pick.lower() == 'nan' or pick in picks:
                continue
            probability = probability_text(row, probability_column)
            picks.append(f'{pick}（{probability}）' if probability else pick)
        return ' / '.join(picks) or '暂无可靠半全场'

    def market_support(row: pd.Series, market: str, formal_pick: str) -> tuple[bool, str]:
        gate = str(row.get('盘口门控') or '')
        # Respect the short-wave market state produced by market_flow_gate.
        # A reversal, oscillation, or unconfirmed rapid move is not a usable
        # confirmation signal for the daily card.
        if any(word in gate for word in (
                '冲突', '震荡', '不稳定', '反复', '过快', '等待确认', '暂不主推')):
            return False, gate or '盘口不稳定'
        codes = {'胜': 0, '平': 1, '负': 2}
        pick_index = codes.get(formal_pick)
        if pick_index is None:
            return False, '正式方向无法核验'
        if market == '胜平负':
            opening_columns = ('首次采集胜奖金', '首次采集平奖金', '首次采集负奖金')
            current_columns = ('官方胜奖金', '官方平奖金', '官方负奖金')
        else:
            opening_line = number(row, '首次采集让球数')
            current_line = number(row, '官方让球数')
            if not np.isfinite(opening_line) or not np.isfinite(current_line):
                return False, '缺少让球线快照'
            if not np.isclose(opening_line, current_line):
                return False, f'让球线变化 {opening_line:+g}→{current_line:+g}'
            opening_columns = ('首次采集让胜奖金', '首次采集让平奖金', '首次采集让负奖金')
            current_columns = ('官方让胜奖金', '官方让平奖金', '官方让负奖金')
        opening = implied(row, opening_columns)
        current = implied(row, current_columns)
        if opening is None or current is None:
            return False, '缺少初盘或当前盘口'
        if int(np.argmax(current)) != pick_index:
            return False, '当前盘口首选与正式模型不同向'
        movement = float(current[pick_index] - opening[pick_index])
        if movement < -0.015:
            return False, f'临场明显反向 {movement:+.1%}'
        if movement < -0.005:
            return False, f'正式方向走弱 {movement:+.1%}'
        if movement > 0.015:
            state = '持续走强'
        elif abs(movement) <= 0.003:
            state = '当前支持·去水概率基本不变'
        else:
            state = '当前支持·轻微走强'
        return True, f'{state} {movement:+.1%}'

    candidates_by_day: dict[str, list[dict]] = {}
    for _, row in active.iterrows():
        sale_state = str(row.get('官方销售状态') or '').strip()
        sync_window = str(row.get('同步时段') or '').strip()
        if sync_window == '停止推荐' or sale_state in {
                '可能已停售', '已退出当前在售列表'}:
            continue
        card_day = _ticket_card_date(row.get('比赛时间'), row.get('赛事编号'))
        day_text = card_day.isoformat() if card_day is not None else str(row.get('比赛时间') or '')[:10]
        market_candidates: list[tuple[str, str, str, float, float, float]] = []

        def add(
                market: str, choice: str, formal_pick: str, probability: float,
                probability_columns: tuple[str, str, str],
                odds_columns: tuple[str, str, str],
        ) -> None:
            if not choice or not np.isfinite(probability) or probability <= 0:
                return
            pick_index = {'胜': 0, '平': 1, '负': 2}.get(formal_pick)
            if pick_index is None:
                return
            odds = number(row, odds_columns[pick_index])
            if not np.isfinite(odds) or odds <= 1.0:
                return
            margin = formal_margin(row, probability_columns)
            market_candidates.append((
                market, choice, formal_pick, probability, margin, odds,
            ))

        regular_offered = all(np.isfinite(number(row, column)) for column in (
            '官方胜奖金', '官方平奖金', '官方负奖金',
        ))
        if regular_offered:
            add(
                '胜平负', str(row.get('胜平负首选') or ''),
                str(row.get('胜平负首选') or ''),
                number(row, '胜平负首选概率'),
                ('模型主胜概率', '模型平局概率', '模型客胜概率'),
                ('官方胜奖金', '官方平奖金', '官方负奖金'),
            )
        line = number(row, '官方让球数')
        handicap_offered = np.isfinite(line) and all(
            np.isfinite(number(row, column)) for column in (
                '官方让胜奖金', '官方让平奖金', '官方让负奖金',
            )
        )
        if handicap_offered:
            handicap_pick = str(row.get('让球首选') or '').strip()
            add(
                '让球胜平负', f'{line:+g}球 {handicap_pick}', handicap_pick,
                number(row, '让球首选概率'),
                ('模型让胜概率', '模型让平概率', '模型让负概率'),
                ('官方让胜奖金', '官方让平奖金', '官方让负奖金'),
            )
        for market, choice, formal_compare, probability, margin, official_odds in market_candidates:
            draw_protection = (
                str(row.get('平局双选保护') or '').strip()
                if market == '胜平负' else ''
            )
            if draw_protection not in ('胜平', '平负'):
                draw_protection = ''
            monte_pick = first_simulation_pick(row, market)
            monte_compare = monte_pick.removeprefix('让') if market == '让球胜平负' else monte_pick
            if not monte_pick:
                continue
            monte_aligned = (
                formal_compare == monte_compare
                or bool(draw_protection and monte_compare in draw_protection)
            )
            monte_conflict = not monte_aligned
            formal_model = str(row.get('胜负模型类别') or '').strip()
            # Legacy/imported reports without provenance keep their historical
            # behavior. Only rows explicitly labelled as market-derived lose
            # the independent-model validation claim.
            score_model_status = str(row.get('比分模型状态') or '').strip()
            market_derived = (
                bool(score_model_status)
                and '专用模型启用' not in score_model_status
                if market == '让球胜平负'
                else formal_model in {'市场基线', '通用/市场模型'}
            )
            if market_derived:
                supported, trend_text = market_support(row, market, formal_compare)
                support_text = f'同源趋势检查｜{trend_text}｜不计独立验证'
            else:
                supported, support_text = market_support(row, market, formal_compare)
            if not supported:
                continue
            lineup_status = str(row.get('首发状态') or '')
            lineup_conflict = bool(row.get('阵容方向冲突'))
            lineup_warning = str(row.get('阵容预警级别') or '无')
            if lineup_conflict or lineup_warning == '高':
                continue
            empirical_accuracy = number(
                row,
                '让球历史命中率' if market == '让球胜平负' else '同阈值历史命中率',
            )
            empirical_samples = number(
                row,
                '让球回测样本数' if market == '让球胜平负' else '筛选回测样本数',
            )
            if not np.isfinite(empirical_accuracy) or not np.isfinite(empirical_samples):
                learned_accuracy, learned_samples = historical_calibration(
                    market, probability,
                )
                if learned_accuracy is not None:
                    empirical_accuracy = learned_accuracy
                    empirical_samples = float(learned_samples)
            decision = evaluate_value(
                market, probability, official_odds,
                empirical_accuracy=(
                    empirical_accuracy if np.isfinite(empirical_accuracy) else None
                ),
                empirical_samples=(
                    int(empirical_samples) if np.isfinite(empirical_samples) else 0
                ),
            )
            handicap_observation = (
                market == '让球胜平负'
                and probability >= 0.55 and not decision.promoted
            )
            handicap_independent = (
                market == '让球胜平负'
                and str(row.get('让球建议状态') or '').strip()
                in {'观察', '高置信主推', '精选主推'}
            )
            # Keep the three-way cross-check intact, but do not let a strict
            # value gate reduce a usable daily card to one row. These rows
            # are explicitly observations (no stake) and are eligible only
            # when the formal direction, current market, and Monte Carlo are
            # already aligned. The value gate still controls all starred
            # picks and suggested stakes.
            fallback_observation = (
                not decision.promoted
                and not handicap_observation
                and probability >= 0.45
                and margin >= 0.08
            )
            # A Monte Carlo disagreement must never become a starred pick or
            # a suggested stake. When the daily card would otherwise be too
            # sparse, retain it as a clearly labelled observation so the user
            # can see the conflict and decide independently.
            monte_observation = (
                allow_observation_conflicts
                and monte_conflict
                and not decision.promoted
                and not handicap_observation
                and probability >= 0.45
                and margin >= 0.08
            )
            if not monte_aligned and not monte_observation and not handicap_independent:
                continue
            if (
                not decision.promoted and not draw_protection
                and not handicap_observation and not fallback_observation
                and not monte_observation and not handicap_independent
            ):
                continue
            lineup_text = (
                f'已确认·预警{lineup_warning}' if lineup_status == '已确认'
                else '未确认·不调整模型'
            )
            display_grade = (
                '平局双选保护' if draw_protection else
                '蒙特反向观察' if monte_observation else
                '市场同源观察' if market_derived else
                '核心重点' if (
                    market == '让球胜平负'
                    and str(row.get('让球建议状态') or '').strip()
                    in {'高置信主推', '精选主推'}
                ) else
                '盘口观察' if handicap_observation else
                decision.grade if decision.promoted else
                '综合观察'
            )
            safety_rank, safety_label, action = {
                '核心重点': (5, 'A｜正式重点', '达到正式门槛，可优先考虑'),
                '可买优选': (4, 'B｜正式优选', '达到正式门槛，谨慎考虑'),
                '平局双选保护': (3, 'C｜保护方案', '仅作双选保护'),
                '盘口观察': (2, 'D｜仅观察', '不建议投注'),
                '综合观察': (2, 'D｜仅观察', '不建议投注'),
                '蒙特反向观察': (0, 'E｜方向冲突', '不建议投注'),
                '市场同源观察': (1, 'E｜同源信号', '不建议投注'),
            }[display_grade]
            grade_rank = 2 if display_grade == '核心重点' else (
                1 if display_grade == '可买优选' else 0
            )
            quality = (
                safety_rank * 100.0 + grade_rank * 10.0
                + decision.conservative_ev * 5.0
                + probability + margin
            )
            candidates_by_day.setdefault(day_text, []).append({
                '_quality': quality,
                '_safety_rank': safety_rank,
                '相对安全等级': safety_label,
                '行动结论': action,
                '比赛日期': day_text,
                '赛事编号': row.get('赛事编号', ''),
                '联赛': row.get('联赛', ''),
                '对阵': f'{row.get("主队", "")} vs {row.get("客队", "")}',
                '推荐玩法': '胜平负双选' if draw_protection else market,
                '重点选项': (
                    f'· {draw_protection or choice}'
                    if draw_protection or handicap_observation or fallback_observation or monte_observation
                    else f'★ {choice}'
                ),
                '最佳比分': best_score(row),
                '每日2串1': '',
                '2串1组合概率': '',
                '2串1组合SP': '',
                '_pair_probability': probability,
                '_pair_odds': official_odds,
                '_pair_choice': choice,
                '推荐等级': display_grade,
                '推荐性质': (
                    '正式主推' if display_grade in ('核心重点', '可买优选')
                    else '平局保护双选' if draw_protection else '观察/不投注'
                ),
                '正式主模型': str(row.get('胜负模型类别') or '市场基线'),
                '数据状态': '｜'.join(filter(None, (
                    str(row.get('数据采集来源') or ''),
                    str(row.get('数据完整性') or ''),
                ))),
                '正式模型概率': f'{probability:.1%}',
                '价值评估': (
                    f'SP {official_odds:.2f}｜保守概率 {decision.conservative_probability:.1%}'
                    f'｜EV {decision.conservative_ev:+.1%}'
                ),
                '建议仓位': (
                    '仅作平局保护，不计单选仓位'
                    if draw_protection else f'≤{decision.stake_fraction:.1%}本金'
                    if decision.stake_fraction > 0 else '不投注'
                ),
                '盘口验证': support_text,
                '蒙特卡洛是否同向': (
                    f'同向（蒙特：{monte_pick}）'
                    if monte_aligned else f'反向观察（蒙特：{monte_pick}）'
                ),
                '阵容验证': lineup_text,
                '比分参考': score_reference(row),
                '半全场参考': half_full_reference(row),
                '入选理由': (
                    (
                        '正式概率来自市场基线且蒙特方向相反；仅保留风险观察'
                        if market_derived and monte_observation else
                        '正式概率来自市场基线，与盘口属于同一信号；仅保留观察'
                        if market_derived else
                        '正式模型与盘口同向，但蒙特反向，仅展示观察方向'
                        if monte_observation else
                        '三方同向但保守EV未达标，仅展示观察方向'
                        if handicap_observation or fallback_observation else
                        f'{display_grade}；正式模型定方向；领先第二方向{margin:.1%}；'
                        '保守EV达标；盘口与独立模拟双重同向'
                    )
                ),
            })
    # Match the macOS card view: rank each lottery-card date independently and
    # keep up to five fixtures for that date.  Dates never compete for one
    # global quota, so tomorrow cannot hide recommendations purchasable today.
    rows = []
    for day_text in sorted(candidates_by_day):
        ranked = sorted(
            candidates_by_day[day_text],
            key=lambda item: item['_quality'], reverse=True,
        )
        day_rows, used_matches = [], set()
        for item in ranked:
            if len(day_rows) >= 5:
                break
            number = str(item['赛事编号'])
            if number in used_matches:
                continue
            day_rows.append(item)
            used_matches.add(number)
        rows.extend(day_rows)

    # A 2-leg combination may only use selected fixtures from the same card
    # date. Build it after the per-date selection so its displayed partner can
    # never refer to a row discarded from that date's final five.
    selected_by_day: dict[str, list[dict]] = {}
    for item in rows:
        selected_by_day.setdefault(str(item['比赛日期']), []).append(item)
    for day_rows in selected_by_day.values():
        # Build one daily 2-leg high-SP combination from independent fixtures.
        # Both legs need a meaningful probability; the pair is selected by
        # positive theoretical EV, then by probability and price.
        pairs = []
        for left_index, left in enumerate(day_rows):
            for right in day_rows[left_index + 1:]:
                p1, p2 = left.get('_pair_probability'), right.get('_pair_probability')
                o1, o2 = left.get('_pair_odds'), right.get('_pair_odds')
                if not all(np.isfinite(float(v)) for v in (p1, p2, o1, o2)):
                    continue
                if min(float(p1), float(p2)) < 0.50 or min(float(o1), float(o2)) <= 1.0:
                    continue
                combined_probability = float(p1) * float(p2)
                combined_odds = float(o1) * float(o2)
                pairs.append((
                    combined_probability * combined_odds - 1.0,
                    combined_probability,
                    combined_odds,
                    left,
                    right,
                ))
        if pairs:
            _, pair_probability, pair_odds, left, right = max(
                pairs, key=lambda item: (item[0], item[1], item[2]),
            )
            pair_text = (
                f'{left["赛事编号"]} {left["_pair_choice"]}@{float(left["_pair_odds"]):.2f}'
                f' × {right["赛事编号"]} {right["_pair_choice"]}@{float(right["_pair_odds"]):.2f}'
            )
            for item in (left, right):
                item['每日2串1'] = pair_text
                item['2串1组合概率'] = f'{pair_probability:.1%}'
                item['2串1组合SP'] = f'{pair_odds:.2f}'
    for row in rows:
        row.pop('_quality', None)
        row.pop('_safety_rank', None)
        row.pop('_pair_probability', None)
        row.pop('_pair_odds', None)
        row.pop('_pair_choice', None)
    return pd.DataFrame(rows, columns=columns)

