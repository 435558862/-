"""Conservative, market-specific value gates for daily recommendations.

The formal model owns the probability and the official fixed award owns the
price.  This module never chooses a direction.  It only answers whether the
already chosen direction is worth promoting after an uncertainty haircut.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import pandas as pd


SETTLED_PATH = Path('storage/jingcai/learning/settled_predictions.csv')


@dataclass(frozen=True)
class ValueRule:
    uncertainty_haircut: float
    candidate_probability: float
    core_probability: float
    candidate_ev: float
    core_ev: float
    candidate_stake_cap: float
    core_stake_cap: float


@dataclass(frozen=True)
class ValueDecision:
    market: str
    raw_probability: float
    calibrated_probability: float
    conservative_probability: float
    official_odds: float
    conservative_ev: float
    grade: str
    stake_fraction: float
    calibration_samples: int

    @property
    def promoted(self) -> bool:
        return self.grade in {'核心重点', '可买优选'}


RULES = {
    '胜平负': ValueRule(
        uncertainty_haircut=0.025,
        candidate_probability=0.55,
        core_probability=0.60,
        candidate_ev=0.02,
        core_ev=0.05,
        candidate_stake_cap=0.005,
        core_stake_cap=0.010,
    ),
    '让球胜平负': ValueRule(
        uncertainty_haircut=0.040,
        candidate_probability=0.57,
        core_probability=0.62,
        candidate_ev=0.03,
        core_ev=0.07,
        candidate_stake_cap=0.005,
        core_stake_cap=0.010,
    ),
}


@lru_cache(maxsize=4)
def _settled_calibration_rows(path_text: str, modified_ns: int) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except (OSError, UnicodeError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def historical_calibration(
        market: str, probability: float,
        *, path: Path | None = None, selection: str | None = None,
        handicap_line: float | None = None, as_of=None,
) -> tuple[float | None, int]:
    """Return a direction-specific, chronological settled-bin hit rate."""
    target = path or SETTLED_PATH
    try:
        modified_ns = target.stat().st_mtime_ns
    except OSError:
        return None, 0
    frame = _settled_calibration_rows(str(target), modified_ns)
    if frame.empty:
        return None, 0
    if as_of is not None and 'settled_at' in frame.columns:
        cutoff = pd.to_datetime(as_of, errors='coerce')
        settled = pd.to_datetime(frame['settled_at'], errors='coerce')
        if pd.notna(cutoff):
            if getattr(cutoff, 'tzinfo', None) is not None:
                cutoff = cutoff.tz_localize(None)
            try:
                settled = settled.dt.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            frame = frame.loc[settled.notna() & settled.le(cutoff)].copy()
            if frame.empty:
                return None, 0
    if market == '胜平负':
        required = {'predicted_result', 'result_hit', 'model_p_home', 'model_p_draw', 'model_p_away'}
        if not required.issubset(frame.columns):
            return None, 0
        labels = {'胜': 'model_p_home', '平': 'model_p_draw', '负': 'model_p_away'}
        probabilities = pd.Series(float('nan'), index=frame.index, dtype=float)
        for label, column in labels.items():
            mask = frame['predicted_result'].astype(str).eq(label)
            probabilities.loc[mask] = pd.to_numeric(frame.loc[mask, column], errors='coerce')
        hits = pd.to_numeric(frame['result_hit'], errors='coerce')
        direction_valid = (
            frame['predicted_result'].astype(str).eq(str(selection))
            if selection in labels else pd.Series(True, index=frame.index)
        )
    elif market == '让球胜平负':
        required = {'handicap_probability', 'handicap_hit'}
        if not required.issubset(frame.columns):
            return None, 0
        probabilities = pd.to_numeric(frame['handicap_probability'], errors='coerce')
        hits = pd.to_numeric(frame['handicap_hit'], errors='coerce')
        direction_valid = pd.Series(True, index=frame.index)
        if selection in {'胜', '平', '负'} and 'predicted_handicap' in frame.columns:
            normalized = frame['predicted_handicap'].astype(str).str.replace('让', '', regex=False)
            direction_valid &= normalized.eq(str(selection))
        if handicap_line is not None and 'handicap_line' in frame.columns:
            lines = pd.to_numeric(frame['handicap_line'], errors='coerce')
            direction_valid &= lines.sub(float(handicap_line)).abs().le(1e-9)
    elif market == '半场胜平负':
        required = {'half_p_home', 'half_p_draw', 'half_p_away',
                    'actual_half_full', 'half_model_source', 'settled_at'}
        if not required.issubset(frame.columns):
            return None, 0
        columns = ['half_p_home', 'half_p_draw', 'half_p_away']
        distribution = frame[columns].apply(pd.to_numeric, errors='coerce')
        good = distribution.notna().all(axis=1) & distribution.ge(0).all(axis=1)
        good &= distribution.sum(axis=1).between(0.99, 1.01)
        predicted = distribution.fillna(-1).idxmax(axis=1).map(dict(zip(columns, ('胜', '平', '负'))))
        probabilities = distribution.max(axis=1)
        actual = frame['actual_half_full'].astype(str).str[0]
        hits = predicted.eq(actual).astype(float)
        direction_valid = good & actual.isin(['胜', '平', '负'])
        direction_valid &= frame['half_model_source'].astype(str).str.endswith('专用半场胜平负模型（已验证）')
        if selection in {'胜', '平', '负'}:
            direction_valid &= predicted.eq(selection)
    else:
        return None, 0
    valid = direction_valid & probabilities.notna() & hits.notna() & probabilities.between(
        max(0.0, float(probability) - 0.05), min(1.0, float(probability) + 0.05),
    )
    sample = pd.DataFrame({'probability': probabilities[valid], 'hit': hits[valid]})
    sample = sample.tail(500)
    if sample.empty:
        return None, 0
    return float(sample['hit'].mean()), len(sample)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _conservative_calibration(
        probability: float,
        empirical_accuracy: float | None,
        empirical_samples: int,
) -> float:
    """Shrink over-confidence but never boost a live probability.

    Historical hit rates are used only after a meaningful settled sample is
    available.  They can lower a live estimate, never raise one, which keeps
    sparse or lucky history from manufacturing value.
    """
    if (
        empirical_samples < 80
        or not _finite(empirical_accuracy)
        or not 0.0 <= float(empirical_accuracy) <= 1.0
    ):
        return probability
    weight = min(0.50, empirical_samples / 400.0)
    blended = (1.0 - weight) * probability + weight * float(empirical_accuracy)
    return min(probability, blended)


def evaluate_value(
        market: str,
        probability: float,
        official_odds: float,
        *,
        empirical_accuracy: float | None = None,
        empirical_samples: int = 0,
) -> ValueDecision:
    """Return a risk-adjusted value decision for one fixed-award selection."""
    rule = RULES.get(market)
    if rule is None:
        raise ValueError(f'unsupported value market: {market}')
    if not _finite(probability) or not 0.0 < float(probability) < 1.0:
        raise ValueError('probability must be finite and between zero and one')
    if not _finite(official_odds) or float(official_odds) <= 1.0:
        raise ValueError('official odds must be finite and greater than one')

    raw = float(probability)
    odds = float(official_odds)
    calibrated = _conservative_calibration(
        raw, empirical_accuracy, max(0, int(empirical_samples or 0)),
    )
    conservative = max(0.01, calibrated - rule.uncertainty_haircut)
    ev = conservative * odds - 1.0
    if raw >= rule.core_probability and ev >= rule.core_ev:
        grade, cap = '核心重点', rule.core_stake_cap
    elif raw >= rule.candidate_probability and ev >= rule.candidate_ev:
        grade, cap = '可买优选', rule.candidate_stake_cap
    else:
        grade, cap = '观察', 0.0

    full_kelly = max(0.0, (conservative * odds - 1.0) / (odds - 1.0))
    stake = min(cap, 0.25 * full_kelly) if cap else 0.0
    return ValueDecision(
        market=market,
        raw_probability=raw,
        calibrated_probability=calibrated,
        conservative_probability=conservative,
        official_odds=odds,
        conservative_ev=ev,
        grade=grade,
        stake_fraction=stake,
        calibration_samples=max(0, int(empirical_samples or 0)),
    )
