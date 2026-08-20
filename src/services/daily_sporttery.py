"""Daily Sporttery prediction pipeline for supported Big-Five fixtures."""

import json
import logging
import re
import unicodedata
from functools import lru_cache
from math import exp
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd

from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.services.daily_learning import (
    load_selection_profile,
    model_result_is_allowed,
    predict_generic_probabilities,
    review_and_learn,
)
from src.services.odds_tracking import record_odds_snapshots
from src.network.fixtures.sporttery import (
    SportteryMobileClient,
    SportteryScraper,
    latest_had_odds,
    latest_hhad_odds,
)
from src.preprocessing.utils.inputs import construct_inputs_by_fixture
from src.preprocessing.utils.target import TargetType, class_to_score


LEAGUE_ALIASES = {
    '英超': ('英超', '英格兰超级联赛', '英格兰足球超级联赛'),
    '西甲': ('西甲', '西班牙甲级联赛', '西班牙足球甲级联赛'),
    '德甲': ('德甲', '德国甲级联赛', '德国足球甲级联赛'),
    '意甲': ('意甲', '意大利甲级联赛', '意大利足球甲级联赛'),
    '法甲': ('法甲', '法国甲级联赛', '法国足球甲级联赛'),
    '瑞超': ('瑞超', '瑞典超级联赛', '瑞典足球超级联赛'),
    '葡超': ('葡超', '葡萄牙超级联赛', '葡萄牙足球超级联赛'),
    '日职': ('日职', '日本职业联赛', '日本职业足球联赛', 'J联赛'),
    '韩职': ('韩职', '韩国职业联赛', '韩国职业足球联赛', 'K联赛'),
}
OUTCOME_LABELS = np.array(['胜', '平', '负'])
HALF_FULL_LABELS = np.array([
    '胜胜', '胜平', '胜负', '平胜', '平平', '平负', '负胜', '负平', '负负',
])
# Chronological audit: 4,816 official Sporttery matches through 2026-08-14.
# Recent 962-match confirmation was 76.1% at 0.70, 71.1% at 0.625 and 65.1%
# at 0.55.  Only the first two tiers are recommendations; lower-confidence
# fixtures remain visible for analysis but are no longer labelled as a main pick.
MARKET_SELECTION_PROFILE = (
    (0.70, '精选主推', 0.7653, 0.0982),
    (0.625, '高置信主推', 0.7257, 0.2157),
    (0.55, '观察', 0.6527, 0.3922),
    (0.00, '跳过', 0.5214, 1.0),
)
CUP_MARKET_MODEL_PATH = Path(
    'storage/models/market/champions_league_1x2.joblib',
)
CLUBELO_CACHE_ROOT = Path('storage/network/clubelo')
EUROPEAN_CUP_MARKERS = (
    '欧冠', '欧洲冠军联赛', '冠军联赛', '欧联', '欧罗巴',
    'champions league', 'europa league',
)

# Official Sporttery names to ClubElo names. Unknown teams are deliberately
# left missing; a wrong fuzzy match is more damaging than a missing rating.
CLUBELO_TEAM_ALIASES = {
    '阿拉木图凯拉特': 'Kairat',
    '索菲亚列夫斯基': 'Levski',
    '萨格勒布迪纳摩': 'Dinamo Zagreb',
    '维京': 'Viking',
    '凯尔特人': 'Celtic',
    '博德闪耀': 'Bodoe Glimt',
    '圣吉尔联合': 'St Gillis',
    '萨巴赫': 'Sabah',
    '奥胡斯': 'Aarhus',
    '奈梅亨': 'Nijmegen',
    '奥林匹亚科斯': 'Olympiakos',
    '采列': 'Celje',
    '亚拉腊': 'Ararat',
    '布拉迪斯拉发': 'Slovan Bratislava',
    '米亚尔比': 'Mjaellby',
    '格拉茨风暴': 'Sturm Graz',
    '费内巴切': 'Fenerbahce',
    '里昂': 'Lyon',
    '布拉格斯巴达': 'Sparta Praha',
    '巴黎圣日尔曼': 'Paris SG',
    '巴黎圣日耳曼': 'Paris SG',
    '阿斯顿维拉': 'Aston Villa',
    '皇家马德里': 'Real Madrid',
    '巴塞罗那': 'Barcelona',
    '拜仁慕尼黑': 'Bayern Munich',
    '曼城': 'Man City',
    '曼联': 'Man United',
    '利物浦': 'Liverpool',
    '阿森纳': 'Arsenal',
    '切尔西': 'Chelsea',
    '国际米兰': 'Inter',
    '尤文图斯': 'Juventus',
    '马德里竞技': 'Ath Madrid',
    '本菲卡': 'Benfica',
    '波尔图': 'Porto',
    '葡萄牙体育': 'Sp Lisbon',
    '多特蒙德': 'Dortmund',
    '勒沃库森': 'Leverkusen',
    'AC米兰': 'Milan',
    '罗马': 'Roma',
    '那不勒斯': 'Napoli',
}


def _sort_by_match_number(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort tickets by weekday label first, then by the daily sequence."""
    if frame.empty or '赛事编号' not in frame.columns:
        return frame.reset_index(drop=True)

    weekday_order = {
        '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '日': 7, '天': 7,
    }

    def ticket_key(value: object) -> tuple[int, int]:
        ticket = re.sub(r'\s+', '', str(value))
        weekday_match = re.search(r'(?:星期|周)([一二三四五六日天])', ticket)
        number_match = re.search(r'(\d+)$', ticket)
        weekday = weekday_order.get(weekday_match.group(1), 99) if weekday_match else 99
        number = int(number_match.group(1)) if number_match else 999999
        return weekday, number

    keys = frame['赛事编号'].map(ticket_key)
    return frame.assign(
        _星期排序=keys.map(lambda key: key[0]),
        _赛事编号排序=keys.map(lambda key: key[1]),
    ).sort_values(
        ['_星期排序', '_赛事编号排序'], kind='stable',
    ).drop(columns=[
        '_星期排序', '_赛事编号排序',
    ]).reset_index(drop=True)


def _field(row: dict, *names, default=''):
    for name in names:
        value = row.get(name)
        if value not in (None, ''):
            return value
    return default


def identify_league(name: str) -> Optional[str]:
    normalized = str(name).replace(' ', '')
    for league, aliases in LEAGUE_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return league
    return None


def _display_fields(raw: dict) -> Tuple[str, str, str, str]:
    """Parse the official select fallback: 编号 联赛 主队 VS 客队."""
    display = str(raw.get('displayText') or '').strip()
    if not display:
        return '', '', '', ''
    match_num, _, remainder = display.partition(' ')
    league = identify_league(remainder)
    if league is None or ' VS ' not in remainder:
        return match_num, remainder, '', ''
    league_alias = next(
        (alias for alias in LEAGUE_ALIASES[league] if alias in remainder), league,
    )
    teams = remainder.split(league_alias, 1)[-1].strip()
    home, away = [value.strip() for value in teams.split(' VS ', 1)]
    return match_num, league_alias, home, away


def _devig(odds: Dict[str, float]) -> np.ndarray:
    inverse = 1.0 / np.array([odds['H'], odds['D'], odds['A']], dtype=np.float64)
    return inverse / inverse.sum()


def _normalize_club_name(value: str) -> str:
    value = unicodedata.normalize('NFKD', str(value))
    value = ''.join(char for char in value if not unicodedata.combining(char))
    value = value.casefold().replace('&', ' and ')
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    return ' '.join(value.split())


@lru_cache(maxsize=1)
def _load_cup_market_artifact():
    """Load only a model that passed the sealed chronological audit."""
    if not CUP_MARKET_MODEL_PATH.exists():
        return None
    try:
        artifact = joblib.load(CUP_MARKET_MODEL_PATH)
    except Exception:
        logging.exception('欧战市场校准模型加载失败，回退官方赔率。')
        return None
    if not artifact.get('deployable'):
        logging.warning('欧战市场校准模型未通过部署门槛，回退官方赔率。')
        return None
    return artifact


@lru_cache(maxsize=8)
def _clubelo_ratings(as_of: str) -> dict[str, float]:
    """Return a dated ClubElo snapshot, with a durable offline cache."""
    CLUBELO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = CLUBELO_CACHE_ROOT / f'{as_of}.csv'
    if not cache_path.exists():
        payload = None
        for url in (
            f'https://api.clubelo.com/{as_of}',
            f'http://api.clubelo.com/{as_of}',
        ):
            try:
                with urlopen(url, timeout=30) as response:
                    payload = response.read()
                break
            except Exception:
                continue
        if payload is not None:
            temporary = cache_path.with_suffix('.csv.tmp')
            temporary.write_bytes(payload)
            temporary.replace(cache_path)
        else:
            logging.warning('ClubElo 获取失败：%s，已自动使用本地缓存。', as_of)
            cached = sorted(CLUBELO_CACHE_ROOT.glob('*.csv'), reverse=True)
            if not cached:
                return {}
            cache_path = cached[0]
    try:
        data = pd.read_csv(cache_path, usecols=['Club', 'Elo']).dropna()
    except Exception:
        logging.exception('ClubElo 缓存读取失败：%s', cache_path)
        return {}
    return {
        _normalize_club_name(club): float(rating)
        for club, rating in zip(data['Club'], data['Elo'])
    }


def _match_date(raw: dict) -> date:
    value = _field(
        raw, 'matchDate', 'businessDate', 'matchTime',
        'matchDateTime', 'startTime', default=date.today().isoformat(),
    )
    parsed = pd.to_datetime(value, errors='coerce')
    return parsed.date() if not pd.isna(parsed) else date.today()


def _cup_stage(league_name: str, match_day: date) -> str:
    normalized = str(league_name).casefold()
    if not any(marker.casefold() in normalized for marker in EUROPEAN_CUP_MARKERS):
        return 'unknown'
    if match_day.month in (6, 7, 8):
        return 'qualification'
    if match_day.month in (9, 10, 11, 12, 1):
        return 'league phase'
    return 'play offs'


def _cup_market_features(
        odds: Dict[str, float],
        match_day: date,
        stage: str,
        home_elo: float,
        away_elo: float,
        elo_missing: bool,
) -> np.ndarray:
    odds_array = np.array([odds['H'], odds['D'], odds['A']], dtype=np.float64)
    inverse = 1.0 / odds_array
    probability = inverse / inverse.sum()
    ordered = np.sort(probability)
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)))
    month_angle = 2.0 * np.pi * match_day.month / 12.0
    stage = stage.casefold()
    both_ratings = not elo_missing
    elo_difference = home_elo - away_elo if both_ratings else 0.0
    return np.hstack([
        probability,
        np.log(odds_array),
        [inverse.sum()],
        [entropy],
        [ordered[-1] - ordered[-2]],
        probability ** 2,
        [probability[0] - probability[2]],
        [np.sin(month_angle), np.cos(month_angle)],
        [
            float('qualif' in stage),
            float('group' in stage),
            float('play off' in stage),
            float('league phase' in stage),
        ],
        [
            (home_elo - 1500.0) / 400.0,
            (away_elo - 1500.0) / 400.0,
            elo_difference / 400.0,
            float(elo_missing),
        ],
    ]).reshape(1, -1)


def _calibrated_cup_probabilities(
        raw: dict,
        league_name: str,
        home: str,
        away: str,
        odds: Dict[str, float],
) -> tuple[Optional[np.ndarray], bool]:
    """Apply the small audited cup correction; always fail safe to market."""
    normalized_league = str(league_name).casefold()
    if not any(
        marker.casefold() in normalized_league for marker in EUROPEAN_CUP_MARKERS
    ):
        return None, False
    artifact = _load_cup_market_artifact()
    if artifact is None:
        return None, False
    match_day = _match_date(raw)
    # The training audit used a two-day lag to avoid same-day result leakage.
    ratings_day = (match_day - timedelta(days=2)).isoformat()
    ratings = _clubelo_ratings(ratings_day)
    home_name = CLUBELO_TEAM_ALIASES.get(home, home)
    away_name = CLUBELO_TEAM_ALIASES.get(away, away)
    home_rating = ratings.get(_normalize_club_name(home_name))
    away_rating = ratings.get(_normalize_club_name(away_name))
    elo_complete = home_rating is not None and away_rating is not None
    features = _cup_market_features(
        odds, match_day, _cup_stage(league_name, match_day),
        float(home_rating) if home_rating is not None else 1500.0,
        float(away_rating) if away_rating is not None else 1500.0,
        not elo_complete,
    )
    try:
        learned = artifact['model'].predict_proba(features)[0]
        weight = float(artifact['model_weight'])
        result = weight * learned + (1.0 - weight) * _devig(odds)
        return result / result.sum(), elo_complete
    except Exception:
        logging.exception('欧战市场校准失败，回退官方赔率。')
        return None, False


def _market_selection(max_probability: float) -> dict:
    """Return an honest selective-prediction grade from held-out history."""
    learned = load_selection_profile()
    if learned is not None:
        rows = sorted(
            learned['rows'], key=lambda row: float(row['threshold']), reverse=True,
        )
        for row in rows:
            if max_probability >= float(row['threshold']):
                return {
                    'grade': str(row['grade']),
                    'threshold': float(row['threshold']),
                    'accuracy': float(row['accuracy']),
                    'coverage': float(row['coverage']),
                    'samples': int(row['samples']),
                    'period': str(learned.get('period') or ''),
                }
    for threshold, grade, accuracy, coverage in MARKET_SELECTION_PROFILE:
        if max_probability >= threshold:
            return {
                'grade': grade,
                'threshold': threshold,
                'accuracy': accuracy,
                'coverage': coverage,
                'samples': 4816,
                'period': '2025-08-11至2026-08-14',
            }
    raise AssertionError('Market selection profile must include a zero threshold.')


def _poisson_pmf(expected_goals: float, max_goals: int) -> np.ndarray:
    """Return Poisson probabilities from zero through ``max_goals``."""
    probabilities = np.zeros(max_goals + 1, dtype=np.float64)
    probabilities[0] = exp(-expected_goals)
    for goals in range(1, max_goals + 1):
        probabilities[goals] = (
            probabilities[goals - 1] * expected_goals / goals
        )
    return probabilities


def _score_grid(home_goals: float, away_goals: float, max_goals: int) -> np.ndarray:
    grid = np.outer(
        _poisson_pmf(home_goals, max_goals),
        _poisson_pmf(away_goals, max_goals),
    )
    return grid / grid.sum()


def _outcome_probabilities(grid: np.ndarray) -> np.ndarray:
    probabilities = np.array([
        np.tril(grid, k=-1).sum(),
        np.trace(grid),
        np.triu(grid, k=1).sum(),
    ], dtype=np.float64)
    return probabilities / probabilities.sum()


def _fit_market_goals(market_probabilities: np.ndarray) -> Tuple[float, float]:
    """Fit neutral Poisson scoring rates to official de-vigged 1X2 odds.

    This is deliberately a transparent fallback, not a trained-team model. A
    coarse-to-fine deterministic search is fast for the small daily fixture
    list and avoids introducing another runtime dependency.
    """
    best_home, best_away, best_loss = 1.35, 1.10, float('inf')

    def search(home_values, away_values):
        nonlocal best_home, best_away, best_loss
        for home_goals in home_values:
            for away_goals in away_values:
                estimated = _outcome_probabilities(
                    _score_grid(float(home_goals), float(away_goals), 11),
                )
                loss = float(np.square(estimated - market_probabilities).sum())
                if loss < best_loss:
                    best_home, best_away, best_loss = (
                        float(home_goals), float(away_goals), loss,
                    )

    search(np.arange(0.20, 4.21, 0.15), np.arange(0.20, 4.21, 0.15))
    search(
        np.arange(max(0.10, best_home - 0.20), best_home + 0.201, 0.025),
        np.arange(max(0.10, best_away - 0.20), best_away + 0.201, 0.025),
    )
    return best_home, best_away


def _half_full_market_probabilities(
        home_goals: float,
        away_goals: float,
) -> np.ndarray:
    """Approximate nine half/full outcomes from fitted scoring rates."""
    first_half = _score_grid(home_goals * 0.45, away_goals * 0.45, 7)
    second_half = _score_grid(home_goals * 0.55, away_goals * 0.55, 7)
    probabilities = np.zeros(9, dtype=np.float64)
    for first_home in range(first_half.shape[0]):
        for first_away in range(first_half.shape[1]):
            half_result = (
                0 if first_home > first_away else 1
                if first_home == first_away else 2
            )
            for second_home in range(second_half.shape[0]):
                for second_away in range(second_half.shape[1]):
                    full_home = first_home + second_home
                    full_away = first_away + second_away
                    full_result = (
                        0 if full_home > full_away else 1
                        if full_home == full_away else 2
                    )
                    probabilities[half_result * 3 + full_result] += (
                        first_half[first_home, first_away]
                        * second_half[second_home, second_away]
                    )
    return probabilities / probabilities.sum()


def _official_ttg_over_under(ttg: Optional[dict]) -> Optional[np.ndarray]:
    """Derive O/U 2.5 from the official total-goals (ttg) market, when offered."""
    if not ttg:
        return None
    try:
        values = [float(ttg[f's{i}']) for i in range(8)]
        if any(value <= 1.0 for value in values):
            return None
        probabilities = 1.0 / np.asarray(values, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    probabilities /= probabilities.sum()
    return np.array(
        [float(probabilities[:3].sum()), float(probabilities[3:].sum())],
        dtype=np.float64,
    )


def _official_crs_score(crs: Optional[dict]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build a 49-class exact-score distribution from the official crs market."""
    if not crs:
        return None
    grid = np.zeros((7, 7), dtype=np.float64)
    found = False
    for key, value in crs.items():
        match = re.fullmatch(r's(\d{2})s(\d{2})', key)
        if match is None:
            continue
        home, away = int(match.group(1)), int(match.group(2))
        if home > 5 or away > 5:
            continue
        try:
            odd = float(value)
        except (TypeError, ValueError):
            continue
        if odd <= 1.0:
            continue
        grid[home, away] = 1.0 / odd
        found = True
    if not found:
        return None
    # "Other" buckets cover every score outside the 6x6 grid (e.g. 6-0, 0-6).
    try:
        other_home = float(crs['s1sh'])
        other_draw = float(crs['s1sd'])
        other_away = float(crs['s1sa'])
    except (KeyError, TypeError, ValueError):
        other_home = other_draw = other_away = 0.0
    if other_home > 1.0:
        grid[6, 0] += 1.0 / other_home
    if other_draw > 1.0:
        grid[6, 6] += 1.0 / other_draw
    if other_away > 1.0:
        grid[0, 6] += 1.0 / other_away
    score = grid.reshape(-1)
    total = float(score.sum())
    if total <= 0:
        return None
    return score / total, np.arange(49, dtype=np.int32)


def _official_hafu_half_full(hafu: Optional[dict]) -> Optional[np.ndarray]:
    """Build the 9-outcome half/full distribution from the official hafu market."""
    if not hafu:
        return None
    keys = ('hh', 'hd', 'ha', 'dh', 'dd', 'da', 'ah', 'ad', 'aa')
    values = []
    for key in keys:
        try:
            value = float(hafu[key])
        except (KeyError, TypeError, ValueError):
            return None
        if value <= 1.0:
            return None
        values.append(1.0 / value)
    probabilities = np.asarray(values, dtype=np.float64)
    probabilities /= probabilities.sum()
    return probabilities


def _market_baseline_probabilities(
        odds: Dict[str, float],
        ttg: Optional[dict] = None,
        crs: Optional[dict] = None,
        hafu: Optional[dict] = None,
) -> dict:
    """Build auditable fallback predictions using official market odds.

    Over/under, exact score and half/full use the official ttg/crs/hafu
    markets when the calculator feed offers them, falling back to the
    transparent Poisson fit when they are unavailable.
    """
    result = _devig(odds)
    home_goals, away_goals = _fit_market_goals(result)
    full_grid = _score_grid(home_goals, away_goals, 11)
    model_grid = _score_grid(home_goals, away_goals, 6)

    over_under = _official_ttg_over_under(ttg)
    if over_under is None:
        total_goals = np.add.outer(np.arange(12), np.arange(12))
        under = float(full_grid[total_goals <= 2].sum())
        over = float(full_grid[total_goals > 2].sum())
        over_under = np.array([under, over], dtype=np.float64)

    official_score = _official_crs_score(crs)
    if official_score is not None:
        score, score_classes = official_score
    else:
        score = model_grid.reshape(-1)
        score = score / score.sum()
        score_classes = np.arange(49, dtype=np.int32)

    half_full = _official_hafu_half_full(hafu)
    if half_full is None:
        half_full = _half_full_market_probabilities(home_goals, away_goals)

    return {
        'result': result,
        'over_under': over_under,
        'score': score,
        'score_classes': score_classes,
        'half_full': half_full,
        'home_goals': home_goals,
        'away_goals': away_goals,
    }


def _model_probabilities(model_db: ModelDatabase, model_id: str, fixture: pd.DataFrame):
    model, config = model_db.load_model(model_id)
    if model is None:
        raise RuntimeError(f'找不到模型：{model_id}')
    return model.predict_proba(fixture)[0], config


def _result_model_is_reliable(config: Optional[dict]) -> bool:
    """Reject a saved 1X2 model only when its own holdout audit is clearly weak."""
    if not config:
        return False
    tuning = config.get('train', {}).get('tuning', {})
    accuracy = tuning.get('test_accuracy')
    if accuracy is None:
        # Older models have no comparable holdout metadata. Preserve their
        # behavior until they are retrained rather than inventing a score.
        return True
    sample_count = tuning.get('test_samples', tuning.get('test_sample_count'))
    if sample_count is not None and int(sample_count) < 100:
        return False
    return float(accuracy) >= 0.50


def _handicap_probabilities(
        score_probabilities: np.ndarray,
        score_classes: np.ndarray,
        handicap: float,
) -> np.ndarray:
    """Aggregate exact-score probabilities into handicap H/D/A outcomes."""
    result = np.zeros(3, dtype=np.float64)
    for probability, target_class in zip(score_probabilities, score_classes):
        home, away = divmod(int(target_class), 7)
        difference = home + handicap - away
        result[0 if difference > 0 else 1 if difference == 0 else 2] += probability
    return result / result.sum()


def _upset_score(
        score_probabilities: np.ndarray,
        score_classes: np.ndarray,
        market_probabilities: np.ndarray,
        excluded: frozenset = frozenset(),
) -> Tuple[str, float]:
    """Pick the strongest exact score in the market's least-likely 1X2 outcome.

    Scores already shown as the main 首选/次选/第三 picks are skipped so the
    upset row never repeats the hot pick.
    """
    upset_outcome = int(np.argmin(market_probabilities))
    candidates = []
    for index, target_class in enumerate(score_classes):
        home, away = divmod(int(target_class), 7)
        outcome = 0 if home > away else 1 if home == away else 2
        if (
            outcome == upset_outcome
            and class_to_score(score_classes[index]) not in excluded
        ):
            candidates.append(index)
    if not candidates:
        return '', float('nan')
    best = max(candidates, key=lambda index: score_probabilities[index])
    return class_to_score(score_classes[best]), float(score_probabilities[best])


def _score_ranking_consistent_with_total(
        score_probabilities: np.ndarray,
        score_classes: np.ndarray,
        prefer_over: bool,
        limit: int = 3,
) -> np.ndarray:
    """Rank exact scores inside the selected O/U side.

    The recommended score must always agree with the over/under pick: an over
    2.5 call is paired with an over score and an under call with an under
    score. The raw modal score is still exposed separately.
    """
    totals = np.array([
        sum(divmod(int(target_class), 7)) for target_class in score_classes
    ])
    eligible = np.flatnonzero(totals > 2 if prefer_over else totals <= 2)
    ranked = eligible[np.argsort(score_probabilities[eligible])[::-1]]
    if len(ranked) >= limit:
        return ranked[:limit]
    remaining = np.array([
        index for index in np.argsort(score_probabilities)[::-1]
        if index not in set(ranked)
    ], dtype=np.int32)
    return np.concatenate([ranked, remaining])[:limit]


def _predict_supported_match(
        raw: dict,
        league: Optional[str],
        home_cn: str,
        away_cn: str,
        home: str,
        away: str,
        odds: Dict[str, float],
        handicap_odds: Optional[Dict[str, float]],
        display_league: Optional[str] = None,
        fallback_reason: str = '',
) -> dict:
    market_prob = _devig(odds)
    market_selection = _market_selection(float(market_prob.max()))
    model_db = None
    fixture = None
    score_classes = None
    prediction_basis = '历史数据训练模型'
    confidence = '正常'
    estimated_teams = []
    trained_result_active = False
    dedicated_model_league = ''
    result_model_category = '市场基线'
    if league is None:
        baseline = _market_baseline_probabilities(
            odds, raw.get('ttg'), raw.get('crs'), raw.get('hafu'),
        )
        result_prob = baseline['result']
        ou_prob = baseline['over_under']
        score_prob = baseline['score']
        score_classes = baseline['score_classes']
        half_full_prob = baseline['half_full']
        prediction_basis = '官方赔率市场基线（未训练联赛）'
        if fallback_reason:
            prediction_basis = f'官方赔率市场基线（{fallback_reason}）'
        confidence = '低'
        calibrated, elo_complete = _calibrated_cup_probabilities(
            raw, display_league or '', home_cn, away_cn, odds,
        )
        if calibrated is not None:
            result_prob = calibrated
            result_model_category = '欧战校准模型'
            prediction_basis = (
                '官方赔率80% + 欧战历史/Elo校准20%'
                if elo_complete else '官方赔率80% + 欧战历史校准20%（Elo缺失）'
            )
            confidence = '中' if elo_complete else '较低'
        else:
            generic = predict_generic_probabilities(
                display_league or '', odds,
                home=home_cn,
                away=away_cn,
                match_date=_match_date(raw).isoformat(),
            )
            if generic is not None:
                result_prob = generic
                result_model_category = '通用模型'
                prediction_basis = '每日官方赛果复盘通用模型 + 市场基线'
                confidence = '中'
    else:
        raw_history = LeagueDatabase().load_league(league)
        # HTR is a half-full target, not a required pre-match feature. Keeping
        # fixtures without HTR gives all prediction models the freshest history.
        required_history = raw_history.drop(columns=['HTR'], errors='ignore').columns
        history = raw_history.dropna(subset=required_history).reset_index(drop=True)
        known_teams = set(history['Home']) | set(history['Away'])
        estimated_teams = [team for team in (home, away) if team not in known_teams]
        if estimated_teams:
            # Median-filled team statistics can create confident but fictional
            # outputs for promoted/new clubs. Use the official market until the
            # club has real league history.
            baseline = _market_baseline_probabilities(
                odds, raw.get('ttg'), raw.get('crs'), raw.get('hafu'),
            )
            result_prob = baseline['result']
            ou_prob = baseline['over_under']
            score_prob = baseline['score']
            score_classes = baseline['score_classes']
            half_full_prob = baseline['half_full']
            prediction_basis = '官方赔率市场基线（球队历史不足）'
            confidence = '低'
        else:
            fixture = construct_inputs_by_fixture(
                history,
                pd.DataFrame([{
                    'Home': home, 'Away': away,
                    '1': odds['H'], 'X': odds['D'], '2': odds['A'],
                }]),
            )
            model_db = ModelDatabase(league)
            result_prob, result_config = _model_probabilities(
                model_db, f'{league}胜平负模型', fixture,
            )
            ou_prob, _ = _model_probabilities(model_db, f'{league}大小球模型', fixture)
            score_prob, _ = _model_probabilities(model_db, f'{league}比分模型', fixture)
            half_full_prob, _ = _model_probabilities(
                model_db, f'{league}半全场模型', fixture,
            )
            # Only mark a row as dedicated after every required league model
            # loaded successfully.  Recognized-but-unmapped/new-team rows use
            # the generic/market view and must not leak into a league picker.
            dedicated_model_league = league
            result_model_category = f'{league}专用模型'
            if not _result_model_is_reliable(result_config):
                result_prob = market_prob
                result_model_category = '市场基线'
                prediction_basis = '历史模型独立测试未达50%，自动回退官方赔率'
                confidence = '较低'
            elif not model_result_is_allowed(f'{league}专用模型'):
                result_prob = market_prob
                result_model_category = '市场基线'
                prediction_basis = '专用模型近期实战低于市场，自动回退官方赔率'
                confidence = '较低'
            else:
                trained_result_active = True

    edge = result_prob - market_prob
    best = int(edge.argmax())
    if score_classes is None:
        score_classes = np.asarray(
            model_db.load_model(f'{league}比分模型')[0].classifier.classes_,
            dtype=np.int32,
        )
    raw_top_score_column = int(np.argmax(score_prob))
    prefer_over = bool(ou_prob[1] >= ou_prob[0])
    top_score_columns = _score_ranking_consistent_with_total(
        score_prob, score_classes, prefer_over,
    )
    top_scores = [
        f'{class_to_score(score_classes[i])} {score_prob[i]:.1%}'
        for i in top_score_columns
    ]
    ranked_scores = [class_to_score(score_classes[i]) for i in top_score_columns]
    top_half_full = np.argsort(half_full_prob)[-3:][::-1]
    ranked_half_full = [HALF_FULL_LABELS[i] for i in top_half_full]
    result_pick = OUTCOME_LABELS[int(np.argmax(result_prob))]
    ou_pick = '大于2.5球' if prefer_over else '小于2.5球'
    upset_score, upset_score_probability = _upset_score(
        score_prob, score_classes, market_prob,
        excluded=frozenset(ranked_scores[:3]),
    )
    handicap_probability = np.full(3, np.nan)
    handicap_market = np.full(3, np.nan)
    handicap_edge = np.full(3, np.nan)
    handicap_best = ''
    handicap_pick = ''
    handicap_second_pick = ''
    if handicap_odds is not None:
        handicap_model_id = f'{league}让球胜负模型' if league else ''
        handicap_model = (
            model_db.load_model(handicap_model_id)[0]
            if model_db is not None and model_db.model_exists(handicap_model_id)
            else None
        )
        if handicap_model is not None:
            handicap_score_prob = handicap_model.predict_proba(fixture)[0]
            handicap_score_classes = np.asarray(
                handicap_model.classifier.classes_, dtype=np.int32,
            )
        else:
            # Backward-compatible fallback while a league has no dedicated model.
            handicap_score_prob = score_prob
            handicap_score_classes = score_classes
        handicap_probability = _handicap_probabilities(
            handicap_score_prob, handicap_score_classes, handicap_odds['line'],
        )
        handicap_market = _devig(handicap_odds)
        handicap_edge = handicap_probability - handicap_market
        handicap_best = OUTCOME_LABELS[int(np.nanargmax(handicap_edge))]
        handicap_ranking = np.argsort(handicap_probability)[::-1]
        handicap_pick = OUTCOME_LABELS[int(handicap_ranking[0])]
        handicap_second_pick = OUTCOME_LABELS[int(handicap_ranking[1])]
    return {
        '赛事编号': _field(raw, 'matchNumStr', 'matchNum'),
        '比赛ID': _field(raw, 'matchId'),
        '比赛时间': _field(raw, 'matchDate', 'matchTime', 'matchDateTime', 'startTime'),
        '联赛': display_league or league or str(_field(
            raw, 'leagueAllName', 'leagueName', 'leagueAbbName', default='未识别联赛',
        )),
        '主队': home_cn,
        '客队': away_cn,
        '主队模型名': home,
        '客队模型名': away,
        '官方胜奖金': odds['H'],
        '官方平奖金': odds['D'],
        '官方负奖金': odds['A'],
        '模型主胜概率': result_prob[0],
        '模型平局概率': result_prob[1],
        '模型客胜概率': result_prob[2],
        '胜平负首选': result_pick,
        '胜平负首选概率': float(result_prob[int(np.argmax(result_prob))]),
        '市场去水主胜概率': market_prob[0],
        '市场去水平局概率': market_prob[1],
        '市场去水客胜概率': market_prob[2],
        '最大价值方向': OUTCOME_LABELS[best] if trained_result_active else '',
        '最大概率优势': edge[best] if trained_result_active else np.nan,
        '建议状态': (
            '观察' if trained_result_active and edge[best] >= 0.03
            else '跳过' if trained_result_active else market_selection['grade']
        ),
        '预测依据': prediction_basis,
        '专用模型联赛': dedicated_model_league,
        '模型类别': (
            f'{dedicated_model_league}专用模型'
            if dedicated_model_league else '通用/市场模型'
        ),
        '胜负模型类别': result_model_category,
        '置信等级': confidence,
        '估算球队': '、'.join(estimated_teams),
        '市场最高概率': float(market_prob.max()),
        '市场筛选阈值': market_selection['threshold'],
        '同阈值历史命中率': market_selection['accuracy'],
        '同阈值历史覆盖率': market_selection['coverage'],
        '筛选回测样本数': market_selection['samples'],
        '筛选回测期间': market_selection['period'],
        '官方让球数': handicap_odds['line'] if handicap_odds else np.nan,
        '官方让胜奖金': handicap_odds['H'] if handicap_odds else np.nan,
        '官方让平奖金': handicap_odds['D'] if handicap_odds else np.nan,
        '官方让负奖金': handicap_odds['A'] if handicap_odds else np.nan,
        '模型让胜概率': handicap_probability[0],
        '模型让平概率': handicap_probability[1],
        '模型让负概率': handicap_probability[2],
        '让球首选': handicap_pick,
        '让球次选': handicap_second_pick,
        '让球首选概率': (
            float(handicap_probability[handicap_ranking[0]]) if handicap_odds else np.nan
        ),
        '让球次选概率': (
            float(handicap_probability[handicap_ranking[1]]) if handicap_odds else np.nan
        ),
        '让球最大价值方向': handicap_best,
        '让球最大概率优势': (
            float(np.nanmax(handicap_edge)) if handicap_odds else np.nan
        ),
        '小于2.5球概率': ou_prob[0],
        '大于2.5球概率': ou_prob[1],
        '大小球首选': ou_pick,
        '大小球首选概率': float(np.max(ou_prob)),
        '半全场首选': ranked_half_full[0],
        '半全场次选': ranked_half_full[1],
        '半全场首选概率': float(half_full_prob[top_half_full[0]]),
        '半全场次选概率': float(half_full_prob[top_half_full[1]]),
        '半全场第三选择': ranked_half_full[2],
        '半全场Top3': ' / '.join(
            f'{HALF_FULL_LABELS[i]} {half_full_prob[i]:.1%}' for i in top_half_full
        ),
        '首选比分': ranked_scores[0],
        '首选比分概率': float(score_prob[top_score_columns[0]]),
        '次选比分': ranked_scores[1],
        '次选比分概率': float(score_prob[top_score_columns[1]]),
        '第三比分': ranked_scores[2],
        '最可能比分Top3': ' / '.join(top_scores),
        '原始最高概率比分': class_to_score(score_classes[raw_top_score_column]),
        '原始最高概率比分概率': float(score_prob[raw_top_score_column]),
        '爆冷比分': upset_score,
        '比分爆冷': upset_score,
        '爆冷比分概率': upset_score_probability,
        '数据来源': 'https://m.sporttery.cn/mjc/jsq/zqspf/',
    }


def run_daily_sporttery(
        output_root: Path = Path('storage/jingcai'),
        headless: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    today = date.today().isoformat()
    try:
        review_and_learn()
    except Exception:
        # Daily review must never prevent today's predictions. The learning
        # state remains auditable and the next manual sync retries it.
        logging.exception('每日自动复盘失败，本次继续生成预测。')
    raw_path = output_root / 'raw' / f'{today}.json'
    try:
        matches = SportteryMobileClient().snapshot(raw_path)
    except RuntimeError:
        logging.exception('官方移动端接口失败，切换浏览器备用方案。')
        with SportteryScraper(headless=headless) as scraper:
            matches = scraper.snapshot(
                raw_path,
                include_bonus=lambda row: identify_league(
                    str(_field(row, 'leagueAllName', 'leagueName', 'leagueAbbName'))
                    or _display_fields(row)[1]
                ) is not None,
            )

    # Append odds snapshots so 盘口分析 can track open-to-kickoff drift.
    record_odds_snapshots(matches)

    aliases = json.loads(
        Path('storage/network/sporttery_team_aliases.json').read_text(encoding='utf-8'),
    )
    predictions, skipped = [], []
    for raw in matches:
        display_num, display_league, display_home, display_away = _display_fields(raw)
        league_name = str(
            _field(raw, 'leagueAllName', 'leagueName', 'leagueAbbName', default=display_league)
        )
        league = identify_league(league_name)
        home_cn = str(_field(
            raw, 'homeTeamAllName', 'homeTeamName', 'homeTeamAbbName', default=display_home,
        ))
        away_cn = str(_field(
            raw, 'awayTeamAllName', 'awayTeamName', 'awayTeamAbbName', default=display_away,
        ))
        if display_num and not raw.get('matchNumStr'):
            raw['matchNumStr'] = display_num
        reason = ''
        mapped = (
            league is not None
            and home_cn in aliases.get(league, {})
            and away_cn in aliases.get(league, {})
        )
        odds = latest_had_odds(raw if raw.get('had') else (raw.get('fixedBonus') or {}))
        handicap_odds = latest_hhad_odds(raw)
        if odds is None:
            reason = '未提供胜平负固定奖金'

        if reason:
            skipped.append({
                '赛事编号': _field(raw, 'matchNumStr', 'matchNum'),
                '比赛ID': _field(raw, 'matchId'),
                '官方联赛': league_name,
                '主队': home_cn,
                '客队': away_cn,
                '跳过原因': reason,
            })
            continue
        try:
            predictions.append(_predict_supported_match(
                raw, league if mapped else None, home_cn, away_cn,
                aliases[league][home_cn] if mapped else home_cn,
                aliases[league][away_cn] if mapped else away_cn,
                odds, handicap_odds,
                display_league=league or league_name,
                fallback_reason=(
                    '球队尚未映射' if league is not None and not mapped
                    else '未训练联赛'
                ),
            ))
        except Exception as error:
            logging.exception('竞彩场次预测失败：%s', raw.get('matchId'))
            skipped.append({
                '赛事编号': _field(raw, 'matchNumStr', 'matchNum'),
                '比赛ID': _field(raw, 'matchId'),
                '官方联赛': league_name,
                '主队': home_cn,
                '客队': away_cn,
                '跳过原因': f'预测失败：{error}',
            })

    prediction_df = _sort_by_match_number(pd.DataFrame(predictions))
    skipped_df = _sort_by_match_number(pd.DataFrame(skipped))
    report_dir = output_root / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(report_dir / f'{today}-竞彩预测.csv', index=False)
    skipped_df.to_csv(report_dir / f'{today}-未覆盖场次.csv', index=False)
    prediction_df.to_csv(report_dir / '最新竞彩预测.csv', index=False)
    skipped_df.to_csv(report_dir / '最新未覆盖场次.csv', index=False)
    return prediction_df, skipped_df
