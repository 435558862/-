import pandas as pd

from src.services.presale_validation import presale_validation


def _ticks(deadline, probabilities, *, handicap=False, line=-1):
    deadline = pd.Timestamp(deadline)
    deadline = (
        deadline.tz_localize('Asia/Shanghai')
        if deadline.tzinfo is None else deadline.tz_convert('Asia/Shanghai')
    )
    rows = []
    for minutes, probability in zip((120, 60, 30), probabilities):
        # Construct fair three-way odds whose selected-home devig probability
        # is exactly the requested value.
        other = (1.0 - probability) / 2.0
        market = {'H': 1 / probability, 'D': 1 / other, 'A': 1 / other}
        if handicap:
            market['line'] = line
        rows.append({
            'captured_at': (deadline - pd.Timedelta(minutes=minutes + 5)).isoformat(),
            'hhad' if handicap else 'had': market,
        })
    return rows


def test_future_window_is_waiting_not_rejection():
    deadline = pd.Timestamp('2026-09-05 20:00', tz='Asia/Shanghai')
    text, state = presale_validation(
        [], '胜平负', '胜', deadline,
        as_of=deadline - pd.Timedelta(hours=4),
    )
    assert state is None
    assert '等待' in text


def test_three_predeadline_points_confirm_strengthening():
    deadline = pd.Timestamp('2026-09-05 20:00', tz='Asia/Shanghai')
    text, state = presale_validation(
        _ticks(deadline, (0.50, 0.51, 0.53)), '胜平负', '胜', deadline,
        as_of=deadline,
    )
    assert state is True
    assert '持续走强' in text


def test_confirmed_late_reversal_vetoes():
    deadline = pd.Timestamp('2026-09-05 20:00', tz='Asia/Shanghai')
    text, state = presale_validation(
        _ticks(deadline, (0.54, 0.54, 0.51)), '胜平负', '胜', deadline,
        as_of=deadline,
    )
    assert state is False
    assert '走弱' in text


def test_handicap_line_change_vetoes_comparison():
    deadline = pd.Timestamp('2026-09-05 20:00', tz='Asia/Shanghai')
    rows = _ticks(deadline, (0.58, 0.59, 0.60), handicap=True, line=-1)
    rows[-1]['hhad']['line'] = 0
    text, state = presale_validation(
        rows, '让球胜平负', '胜', deadline, as_of=deadline, line=-1,
    )
    assert state is False
    assert '让球线变化' in text
