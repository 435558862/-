"""Incremental market evidence. Ranking score is not a win probability."""
import math
import pandas as pd
from src.services.presale_validation import timestamp


def assess_market(rows, market, pick, deadline, *, as_of=None, line=None):
    def result(text, state=None, score=0.0):
        return dict(text=text, state=state, score=score)
    end = timestamp(deadline)
    now = timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz='Asia/Shanghai')
    if end is None or now is None:
        return result('证据不足：缺少参考截止时间')
    key = {'胜平负': 'had', '让球胜平负': 'hhad'}.get(market)
    code = {'胜': 'H', '平': 'D', '负': 'A'}.get(pick)
    if key is None or code is None:
        return result('证据不足：方向无法识别')
    limit = min(end, now)
    ticks = {}
    for row in rows:
        t = timestamp(row.get('captured_at'))
        if t is None or t > limit:
            continue
        odds = row.get(key) or {}
        try:
            values = [float(odds[c]) for c in ('H', 'D', 'A')]
            if not all(math.isfinite(v) and v > 1 for v in values):
                continue
            p = (1 / float(odds[code])) / sum(1 / v for v in values)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            continue
        ticks[t] = (t, p, odds.get('line'))
    ticks = sorted(ticks.values())
    anchors = []
    for minutes in (120, 60, 30):
        anchor = end - pd.Timedelta(minutes=minutes)
        choices = [x for x in ticks if anchor - pd.Timedelta(minutes=45) <= x[0] <= anchor]
        if anchor <= limit and choices:
            anchors.append(choices[-1])
    coverage = f'截止前档位{len(anchors)}/3'
    if not ticks:
        return result(f'证据不足：无可用快照｜{coverage}')
    age = (limit - ticks[-1][0]).total_seconds() / 60
    if age > 45:
        return result(f'证据过旧：最新快照距核验时点{age:.0f}分钟｜{coverage}')
    recent = [x for x in ticks if x[0] >= limit - pd.Timedelta(minutes=180)]
    points = sorted({x[0]: x for x in recent + anchors}.values())
    if len(points) < 2 or (points[-1][0] - points[0][0]).total_seconds() < 600:
        return result(f'证据不足：需两次快照且跨度10分钟｜{coverage}')
    if key == 'hhad':
        try:
            lines = [float(x[2]) for x in points]
            if not math.isfinite(float(line)) or not all(math.isfinite(v) for v in lines):
                return result('证据不足：缺少让球数')
            if any(abs(v - float(line)) > 1e-9 for v in lines):
                return result('反对：让球线变化，暂停直接比较', False)
        except (ValueError, TypeError):
            return result('证据不足：缺少让球数')
    probs = [x[1] for x in points]
    changes = [b - a for a, b in zip(probs, probs[1:])]
    movement = probs[-1] - probs[0]
    drop = probs[-1] - max(probs)
    detail = f'{len(points)}次快照｜最新{age:.0f}分钟前｜{coverage}'
    if movement <= -.015 or drop <= -.015:
        return result(f'反对：方向走弱{movement * 100:+.1f}pp，较高点{drop * 100:+.1f}pp｜{detail}', False)
    signs = [1 if d > 0 else -1 for d in changes if abs(d) >= .005]
    if sum(a != b for a, b in zip(signs, signs[1:])) >= 2:
        return result(f'反对：去水概率反复震荡｜{detail}', False)
    if movement >= .015 and all(d >= -.003 for d in changes):
        return result(f'支持：持续走强{movement * 100:+.1f}pp｜{detail}', True, .04)
    if movement < -.005:
        return result(f'谨慎：轻微走弱{movement * 100:+.1f}pp｜{detail}', True, -.03)
    return result(f'不反对：去水变化{movement * 100:+.1f}pp｜{detail}', True,
                  .015 if movement > .003 else 0.0)
