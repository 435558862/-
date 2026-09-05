import pandas as pd
from src.services.market_evidence import assess_market


def tick(t, p):
    return {'captured_at': t.isoformat(), 'had': {'H': 1/p, 'D': 2/(1-p), 'A': 2/(1-p)}}


def test_uses_available_trend_before_deadline_windows():
    now = pd.Timestamp('2026-09-05 12:00', tz='Asia/Shanghai')
    rows = [tick(now-pd.Timedelta(minutes=40), .5), tick(now-pd.Timedelta(minutes=5), .54)]
    result = assess_market(rows, '胜平负', '胜', now+pd.Timedelta(hours=6), as_of=now)
    assert result['state'] is True
    assert result['score'] > 0
    assert '档位0/3' in result['text']


def test_peak_reversal_vetoes_even_above_opening():
    now = pd.Timestamp('2026-09-05 12:00', tz='Asia/Shanghai')
    rows = [tick(now-pd.Timedelta(minutes=60), .5), tick(now-pd.Timedelta(minutes=30), .56), tick(now, .53)]
    result = assess_market(rows, '胜平负', '胜', now+pd.Timedelta(hours=1), as_of=now)
    assert result['state'] is False


def test_future_or_stale_ticks_never_create_support():
    now = pd.Timestamp('2026-09-05 12:00', tz='Asia/Shanghai')
    rows = [tick(now-pd.Timedelta(minutes=100), .5), tick(now+pd.Timedelta(minutes=1), .6)]
    result = assess_market(rows, '胜平负', '胜', now+pd.Timedelta(hours=1), as_of=now)
    assert result['state'] is None
    assert result['score'] == 0


def test_proportional_price_changes_do_not_create_strengthening():
    now = pd.Timestamp('2026-09-05 12:00', tz='Asia/Shanghai')
    first = tick(now-pd.Timedelta(minutes=40), .5)
    last = tick(now, .5)
    last['had'] = {k: v*.95 for k, v in last['had'].items()}
    result = assess_market([first, last], '胜平负', '胜', now+pd.Timedelta(hours=1), as_of=now)
    assert result['state'] is True
    assert result['score'] == 0
