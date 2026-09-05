"""Prospective paired rule comparison; never reconstruct yesterday's picks.

Both rules receive the same live inputs. First daily batches are immutable.
This tests selection rules, not historical model retraining or actual bets.
"""
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path('storage/jingcai/forward_comparison')


def _identity(value):
    text = str(value).strip()
    return text[:-2] if text.endswith('.0') else text


def freeze(predictions, *, root=None):
    from src.gui.windows import sporttery
    from src.services.baseline_daily_378f3e3 import build_daily_recommendations as old_rule
    root = Path(root) if root is not None else ROOT
    clock = pd.Timestamp.now(tz='Asia/Shanghai')
    # Explicit China time, including when the application runs on a Mac abroad.
    clock = clock.tz_localize('Asia/Shanghai') if clock.tzinfo is None else clock.tz_convert('Asia/Shanghai')
    frame = predictions.copy()
    if frame.empty or '比赛时间' not in frame:
        return
    kickoff = pd.to_datetime(frame['比赛时间'], errors='coerce')
    if kickoff.dt.tz is None:
        kickoff = kickoff.dt.tz_localize('Asia/Shanghai')
    frame = frame.loc[kickoff.gt(clock + pd.Timedelta(minutes=10))].copy()
    if frame.empty:
        return
    frame['_audit_day'] = [str(sporttery._ticket_card_date(r.get('比赛时间'), r.get('赛事编号')))
                           for _, r in frame.iterrows()]
    # Filter already frozen days before executing either rule.
    missing = [d for d in frame['_audit_day'].unique()
               if re.fullmatch(r'\d{4}-\d{2}-\d{2}', d) and not (root / f'{d}.json').exists()]
    if not missing:
        return
    # Both rules see all eligible dates, matching the live selector's context.
    shared = frame.drop(columns='_audit_day')
    old = old_rule(shared.copy(), future_only=False)
    new = sporttery.build_daily_recommendations(shared.copy(), future_only=False)
    input_text = shared.to_json(orient='records', force_ascii=False)
    hashes = {}
    for file in (Path(sporttery.__file__), Path(__file__),
                 Path(__file__).with_name('baseline_daily_378f3e3.py'),
                 Path(__file__).with_name('baseline_value_378f3e3.py'),
                 Path(__file__).with_name('market_evidence.py'),
                 Path(__file__).with_name('value_selection.py')):
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    for day in missing:
        items = []
        source = frame.loc[frame['_audit_day'].eq(day)]
        for version, selected in [('old', old), ('new', new)]:
            for _, pick in selected.loc[selected['比赛日期'].astype(str).eq(day)].iterrows():
                match = source.loc[source['赛事编号'].astype(str).eq(str(pick['赛事编号']))]
                if len(match) != 1:
                    continue
                match = match.iloc[0]
                market = pick['推荐玩法']
                direction = str(pick['重点选项']).strip()[-1:]
                columns = {'胜平负': {'胜': '官方胜奖金', '平': '官方平奖金', '负': '官方负奖金'},
                           '让球胜平负': {'胜': '官方让胜奖金', '平': '官方让平奖金', '负': '官方让负奖金'}}
                odd = pd.to_numeric(match.get(columns.get(market, {}).get(direction, '')), errors='coerce')
                eligible = market in columns and pd.notna(odd) and float(odd) > 1
                formal = eligible and pick.get('推荐性质') == '正式主推'
                items.append(dict(version=version, match_id=_identity(match.get('比赛ID')),
                                  number=str(pick['赛事编号']), day=day,
                                  kickoff=str(match['比赛时间']), market=market, direction=direction,
                                  line=float(match['官方让球数']) if market == '让球胜平负' else None,
                                  odds=float(odd) if eligible else None,
                                  formal=bool(formal), stake=1.0 if formal else 0.0,
                                  grade=str(pick.get('推荐等级', '')), eligible=bool(eligible),
                                  recommendation=json.loads(pd.DataFrame([pick]).to_json(orient='records', force_ascii=False))[0]))
        payload = dict(schema=1, captured_at=clock.isoformat(), day=day,
                       baseline='378f3e3', code_hashes=hashes,
                       scope='同一赛前输入的新旧筛选规则前瞻对照；等额1单位模拟；非历史重新训练',
                       input_sha256=hashlib.sha256(input_text.encode()).hexdigest(),
                       inputs=json.loads(input_text), picks=items)
        try:
            with (root / f'{day}.json').open('x', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
        except FileExistsError:
            pass


def report(*, root=None, settled_path=None):
    from src.services.yesterday_review import _ticket_card_date
    root = Path(root) if root is not None else ROOT
    settled_path = Path(settled_path) if settled_path is not None else Path('storage/jingcai/learning/settled_predictions.csv')
    outcomes = pd.read_csv(settled_path) if settled_path.exists() else pd.DataFrame()
    results = {}
    for _, result in outcomes.iterrows():
        day = str(_ticket_card_date(result.get('match_date'), result.get('match_number')))
        results[(day, str(result.get('match_number')))] = result
    def empty():
        return dict(冻结批次=0, 正式推荐=0, 已结算=0, 命中=0, 已结算投入=0.0, 净利润=0.0,
                    观察项=0, 观察已结算=0, 观察命中=0)
    stats = {}
    for path in sorted(root.glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        hashes = payload.get('code_hashes', {})
        cohort = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
        for version in ('old', 'new'):
            stats.setdefault((cohort, version), empty())['冻结批次'] += 1
        for pick in payload['picks']:
            stat = stats[(cohort, pick['version'])]
            formal = pick['formal']
            stat['正式推荐' if formal else '观察项'] += 1
            result = results.get((pick['day'], pick['number']))
            if result is None or not pick['eligible']:
                continue
            if pick['match_id'] not in ('', 'None', 'nan') and _identity(result.get('match_id')) != pick['match_id']:
                continue
            home = pd.to_numeric(result.get('home_goals'), errors='coerce')
            away = pd.to_numeric(result.get('away_goals'), errors='coerce')
            if pd.isna(home) or pd.isna(away):
                continue
            difference = home - away + (pick['line'] or 0)
            actual = '胜' if difference > 0 else '负' if difference < 0 else '平'
            hit = actual == pick['direction']
            if formal:
                stat['已结算'] += 1
                stat['命中'] += int(hit)
                stat['已结算投入'] += pick['stake']
                stat['净利润'] += pick['stake'] * (pick['odds'] - 1 if hit else -1)
            else:
                stat['观察已结算'] += 1
                stat['观察命中'] += int(hit)
    if not stats:
        stats = {('尚未冻结', v): empty() for v in ('old', 'new')}
    for (cohort, v), row in stats.items():
        row['版本组'] = cohort
        row['方案'] = '旧版378f3e3' if v == 'old' else '新版（冻结时版本）'
        row['ROI'] = row['净利润'] / row['已结算投入'] if row['已结算投入'] else None
        row['命中率'] = row['命中'] / row['已结算'] if row['已结算'] else None
        row['待结算'] = row['正式推荐'] - row['已结算']
    return pd.DataFrame(stats.values())
