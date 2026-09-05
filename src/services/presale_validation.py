"""Observed pre-deadline odds; never substitute later ticks for missing ones."""
import math
import pandas as pd


def timestamp(value):
    try:
        t = pd.Timestamp(value)
        if pd.isna(t):
            return None
        return t.tz_localize('Asia/Shanghai') if t.tzinfo is None else t.tz_convert('Asia/Shanghai')
    except (ValueError, TypeError):
        return None


def presale_validation(rows, market, pick, deadline, *, as_of=None, line=None):
    """Validate one direction at 120/60/30 minutes before a reference cutoff.

    ``None`` means that the evidence is not available yet and must not veto a
    recommendation.  Only a confirmed reversal or handicap-line change returns
    ``False``.
    """
    end = timestamp(deadline)
    now = timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz='Asia/Shanghai')
    if end is None or now is None:
        return '证据不足：缺少参考截止时间', None
    key = 'had' if market == '胜平负' else 'hhad'
    code = {'胜': 'H', '平': 'D', '负': 'A'}.get(pick)
    if code is None:
        return '证据不足：方向无法识别', None
    ticks = []
    for row in rows:
        t = timestamp(row.get('captured_at'))
        if t is None or t > min(end, now):
            continue
        odds = row.get(key) or {}
        try:
            values = [float(odds[c]) for c in ('H', 'D', 'A')]
            if not all(math.isfinite(v) and v > 1 for v in values):
                continue
            p = (1 / float(odds[code])) / sum(1 / v for v in values)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            continue
        ticks.append((t, p, odds.get('line')))
    ticks.sort(key=lambda item: item[0])
    points = []
    for minutes in (120, 60, 30):
        anchor = end - pd.Timedelta(minutes=minutes)
        if anchor > now:
            wait = max(1, int((anchor - now).total_seconds() // 60))
            return f'等待参考截止窗口：约{wait}分钟后进入{minutes}分钟档', None
        # Never borrow a later tick. Sparse acquisition is accepted only when
        # the most recent earlier observation is reasonably fresh.
        eligible = [x for x in ticks if anchor - pd.Timedelta(minutes=45) <= x[0] <= anchor]
        if not eligible:
            return f'证据不足：缺少参考截止前{minutes}分钟快照', None
        points.append(eligible[-1])
    if key == 'hhad':
        try:
            if any(float(x[2]) != float(line) for x in points):
                return '反对：让球线变化，概率不可直接比较', False
        except (ValueError, TypeError):
            return '证据不足：缺少让球数', None
    a, b, c = [x[1] for x in points]
    if c - b <= -0.015 or c - a <= -0.015:
        return f'反对：停售前方向走弱 {(c-a)*100:+.1f}pp', False
    if b >= a and c >= b and c - a >= 0.015:
        return f'支持：持续走强 {(c-a)*100:+.1f}pp', True
    return f'不反对：去水变化 {(c-a)*100:+.1f}pp', True
