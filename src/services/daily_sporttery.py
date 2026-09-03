"""Daily Sporttery prediction pipeline for supported Big-Five fixtures."""

import json
import logging
import re
import unicodedata
import zlib
from functools import lru_cache
from difflib import SequenceMatcher
from math import exp
from datetime import date, datetime, timedelta
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
    model_result_blend_weight,
    model_result_is_allowed,
    predict_generic_probabilities,
    review_and_learn,
)
from src.services.odds_tracking import (
    market_quality_metrics,
    market_flow_gate,
    read_odds_series,
    record_odds_snapshots,
)
from src.services.lineups import fetch_lineup_analysis
from src.services.draw_calibration import (
    calibrate_draw, draw_gate_applies, draw_protection_pick, select_result_index,
)
from src.services.team_names import resolve_model_team
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
    '英冠': ('英冠', '英格兰冠军联赛', '英格兰足球冠军联赛'),
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
HAFU_ODD_KEYS = (
    ('胜胜', 'hh'), ('胜平', 'hd'), ('胜负', 'ha'),
    ('平胜', 'dh'), ('平平', 'dd'), ('平负', 'da'),
    ('负胜', 'ah'), ('负平', 'ad'), ('负负', 'aa'),
)
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
ESPN_HISTORY_ROOT = Path('storage/network/espn_history')
ESPN_LEAGUE_SLUGS = {
    '英超': 'eng.1', '英冠': 'eng.2', '西甲': 'esp.1', '德甲': 'ger.1',
    '德乙': 'ger.2', '意甲': 'ita.1', '法甲': 'fra.1', '法乙': 'fra.2',
    '葡超': 'por.1', '荷甲': 'ned.1', '荷乙': 'ned.2', '瑞超': 'swe.1',
    '日职': 'jpn.1', '韩职': 'kor.1', '沙职': 'ksa.1', '巴西杯': 'bra.copa_do_brazil',
    '欧罗巴': 'uefa.europa', '欧联': 'uefa.europa', '欧冠': 'uefa.champions',
}
ESPN_TEAM_ALIASES = {
    'athbilbao': ('Athletic Club', 'ATH'),
    'parissg': ('Paris Saint-Germain', 'PSG'),
    'mancity': ('Manchester City', 'MNC'),
    'manunited': ('Manchester United', 'MAN'),
    'inter': ('Internazionale', 'Inter Milan'),
}

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
    """Sort by ticket date, then the sequence printed on that ticket.

    Chinese lottery tickets are grouped by their sale-day label.  A ``周五``
    fixture kicking off at 00:30 on Saturday still belongs before every
    ``周六`` fixture.  The kickoff date alone therefore is not a valid card
    sorting key.
    """
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
    if '比赛时间' in frame.columns:
        match_dates = pd.to_datetime(
            frame['比赛时间'].astype(str).str.slice(0, 10), errors='coerce',
        )
    else:
        match_dates = pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns]')
    def ticket_date(match_date, ticket_weekday):
        if pd.isna(match_date) or ticket_weekday == 99:
            return pd.NaT
        # pandas Monday=0; the ticket parser uses Monday=1. Walking backwards
        # resolves the common after-midnight case and also remains correct at
        # Sunday/Monday and year boundaries.
        kickoff_weekday = match_date.weekday() + 1
        days_after_ticket = (kickoff_weekday - ticket_weekday) % 7
        return match_date - pd.Timedelta(days=days_after_ticket)

    ticket_dates = pd.Series(
        [ticket_date(match_date, key[0])
         for match_date, key in zip(match_dates, keys)],
        index=frame.index,
        dtype='datetime64[ns]',
    )
    # Rows with a usable date are authoritative across week/year boundaries.
    # Legacy rows without dates retain weekday order and follow dated rows.
    missing_date = match_dates.isna()
    return frame.assign(
        _缺少日期排序=missing_date.astype(int),
        _票面日期排序=ticket_dates.fillna(match_dates),
        _星期排序=[key[0] if missing else 0 for key, missing in zip(keys, missing_date)],
        _赛事编号排序=keys.map(lambda key: key[1]),
    ).sort_values(
        ['_缺少日期排序', '_票面日期排序', '_星期排序', '_赛事编号排序'],
        kind='stable', na_position='last',
    ).drop(columns=[
        '_缺少日期排序', '_票面日期排序', '_星期排序', '_赛事编号排序',
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
        # Short aliases such as "西甲" must never match inside another
        # official name such as "巴西甲级联赛". Official fields are exact;
        # displayText fallbacks place the league at the beginning.
        if any(normalized == alias or normalized.startswith(alias) for alias in aliases):
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


def _implied_had_from_handicap_market(
        handicap_odds: Dict[str, float],
        ttg: Optional[dict] = None,
) -> Dict[str, float]:
    """Infer a hidden 1X2 feature input when only official HHAD is sold."""
    handicap_target = _devig(handicap_odds)
    total_target = _official_ttg_over_under(ttg)
    best_loss = float('inf')
    best_grid = None
    for home_goals in np.arange(0.20, 4.51, 0.10):
        for away_goals in np.arange(0.20, 4.51, 0.10):
            grid = _score_grid(float(home_goals), float(away_goals), 6)
            handicap = _handicap_probabilities(
                grid.reshape(-1), np.arange(49, dtype=np.int32),
                handicap_odds['line'],
            )
            loss = float(np.square(handicap - handicap_target).sum())
            if total_target is not None:
                totals = np.add.outer(np.arange(7), np.arange(7))
                under = float(grid[totals <= 2].sum())
                loss += 0.45 * float(np.square(
                    np.array([under, 1.0 - under]) - total_target,
                ).sum())
            else:
                loss += 0.002 * float((home_goals + away_goals - 2.7) ** 2)
            if loss < best_loss:
                best_loss, best_grid = loss, grid
    result = np.clip(_outcome_probabilities(best_grid), 1e-6, 1.0)
    return {'H': 1.0 / result[0], 'D': 1.0 / result[1], 'A': 1.0 / result[2]}


def _implied_had_without_result_market(raw: dict) -> Dict[str, float]:
    """Build a hidden feature input so every official fixture stays visible."""
    official_score = _official_crs_score(raw.get('crs'))
    if official_score is not None:
        score, classes = official_score
        result = np.zeros(3, dtype=np.float64)
        for probability, target_class in zip(score, classes):
            home, away = divmod(int(target_class), 7)
            result[0 if home > away else 1 if home == away else 2] += probability
    else:
        official_half_full = _official_hafu_half_full(raw.get('hafu'))
        if official_half_full is not None:
            result = official_half_full.reshape(3, 3).sum(axis=0)
        else:
            result = np.array([0.385, 0.230, 0.385], dtype=np.float64)
    result = np.clip(result / result.sum(), 1e-6, 1.0)
    return {'H': 1.0 / result[0], 'D': 1.0 / result[1], 'A': 1.0 / result[2]}


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
    # Elo moves gradually. Reusing a snapshot up to seven days old makes the
    # UI instant when ClubElo is unavailable, without materially changing the
    # strength feature used by the market model.
    if not cache_path.exists():
        cached = sorted(CLUBELO_CACHE_ROOT.glob('*.csv'), reverse=True)
        if cached:
            try:
                age = date.fromisoformat(as_of) - date.fromisoformat(cached[0].stem)
                if timedelta(0) <= age <= timedelta(days=7):
                    cache_path = cached[0]
            except ValueError:
                pass
    if not cache_path.exists():
        payload = None
        for url in (
            f'https://api.clubelo.com/{as_of}',
            f'http://api.clubelo.com/{as_of}',
        ):
            try:
                with urlopen(url, timeout=8) as response:
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


def _market_selection(max_probability: float, league_name: str = '') -> dict:
    """Return an honest selective-prediction grade from held-out history."""
    learned = load_selection_profile()
    league_profile = _league_selection_profile(league_name)
    if league_profile:
        learned = league_profile
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


@lru_cache(maxsize=64)
def _league_selection_profile(league_name: str) -> Optional[dict]:
    """Learn conservative per-league thresholds from official settled odds."""
    if not league_name:
        return None
    path = Path('storage/jingcai/learning/official_market_history.csv')
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path)
        names = data.get('league', pd.Series('', index=data.index)).fillna('').astype(str)
        aliases = LEAGUE_ALIASES.get(league_name, (league_name,))
        mask = pd.Series(False, index=data.index)
        for alias in aliases:
            mask |= names.str.contains(str(alias), regex=False)
        frame = data.loc[mask].tail(1000).copy()
        if len(frame) < 250:
            return None
        frame = frame.sort_values('match_date', kind='stable').reset_index(drop=True)
        audit_rows = max(50, int(len(frame) * 0.20))
        calibration = frame.iloc[:-audit_rows].copy()
        audit = frame.iloc[-audit_rows:].copy()
        odds = calibration[['odds_home', 'odds_draw', 'odds_away']].to_numpy(float)
        inverse = 1.0 / odds
        probabilities = inverse / inverse.sum(axis=1, keepdims=True)
        confidence = probabilities.max(axis=1)
        predicted = probabilities.argmax(axis=1)
        actual = pd.to_numeric(calibration['actual_result'], errors='coerce').to_numpy()
        audit_odds = audit[['odds_home', 'odds_draw', 'odds_away']].to_numpy(float)
        audit_inverse = 1.0 / audit_odds
        audit_probability = audit_inverse / audit_inverse.sum(axis=1, keepdims=True)
        audit_actual = pd.to_numeric(audit['actual_result'], errors='coerce').to_numpy()

        def choose(minimum, target, minimum_samples):
            for threshold in np.arange(minimum, 0.801, 0.005):
                selected = confidence >= threshold
                if selected.sum() >= minimum_samples:
                    accuracy = float(np.mean(predicted[selected] == actual[selected]))
                    if accuracy >= target:
                        audit_selected = audit_probability.max(axis=1) >= threshold
                        audit_accuracy = (
                            float(np.mean(
                                audit_probability[audit_selected].argmax(axis=1)
                                == audit_actual[audit_selected]
                            )) if audit_selected.any() else 0.0
                        )
                        return (
                            round(float(threshold), 3), min(accuracy, audit_accuracy),
                            int(audit_selected.sum()),
                        )
            return 0.80, 0.0, 0

        high, high_acc, high_n = choose(0.625, 0.70, 40)
        selected, selected_acc, selected_n = choose(max(0.675, high + 0.02), 0.74, 25)
        observe, observe_acc, observe_n = choose(0.55, 0.62, 60)
        rows = [
            {'threshold': selected, 'grade': '精选主推', 'accuracy': selected_acc,
             'coverage': selected_n / len(audit), 'samples': selected_n},
            {'threshold': high, 'grade': '高置信主推', 'accuracy': high_acc,
             'coverage': high_n / len(audit), 'samples': high_n},
            {'threshold': observe, 'grade': '观察', 'accuracy': observe_acc,
             'coverage': observe_n / len(audit), 'samples': observe_n},
            {'threshold': 0.0, 'grade': '跳过',
             'accuracy': float(np.mean(
                 audit_probability.argmax(axis=1) == audit_actual
             )), 'coverage': 1.0, 'samples': len(audit)},
        ]
        return {
            'rows': rows,
            'period': f'{league_name}时间滚动校准{len(calibration)}场/审计{len(audit)}场',
        }
    except Exception:
        logging.exception('联赛独立阈值计算失败：%s', league_name)
        return None


def _calibrate_draw_probability(probabilities: np.ndarray,
                                market: np.ndarray,
                                history: Optional[pd.DataFrame],
                                league: str = '', home: str = '', away: str = '',
                                draw_flow: float = 0.0,
                                hhad_line_change: float = 0.0,
                                ttg_expected_change: float = 0.0) -> np.ndarray:
    """Calibrate draws using league prior, market draw price and strength gap."""
    result = np.asarray(probabilities, dtype=np.float64).copy()
    learned = calibrate_draw(
        result, league, home, away, market,
        draw_flow, hhad_line_change, ttg_expected_change,
    )
    if learned is not None:
        return learned
    if history is None or len(history) < 100 or 'Result' not in history:
        return result / result.sum()
    labels = history['Result'].astype(str).str.upper()
    draw_prior = float(labels.isin(('D', 'X', '平')).tail(1000).mean())
    if not 0.15 <= draw_prior <= 0.40:
        return result / result.sum()
    strength_gap = abs(float(market[0]) - float(market[2]))
    target = 0.55 * float(market[1]) + 0.45 * draw_prior
    if strength_gap <= 0.10:
        target *= 1.06
    elif strength_gap >= 0.30:
        target *= 0.94
    calibrated_draw = float(np.clip(0.75 * result[1] + 0.25 * target, 0.12, 0.42))
    other = result[[0, 2]]
    other_sum = float(other.sum())
    if other_sum <= 0:
        return result / result.sum()
    result[1] = calibrated_draw
    result[[0, 2]] = other / other_sum * (1.0 - calibrated_draw)
    return result


def _sale_context(raw: dict) -> dict:
    """Expose official sale state and a conservative pre-kickoff deadline."""
    match_date = str(raw.get('matchDate') or '')[:10]
    match_time = str(raw.get('matchTime') or '')[:8]
    status = str(raw.get('sellStatus') or '')
    match_status = str(raw.get('matchStatus') or '')
    selling = status == '2' or match_status.casefold() == 'selling'
    result = {
        '官方销售状态': '销售中' if selling else '可能已停售',
        '参考投注截止': '',
        '距参考截止分钟': np.nan,
        '建议临场同步时段': '',
        '投注时间提示': '请以购彩平台实际停售倒计时为准',
        '同步时段': '常规同步',
    }
    try:
        kickoff = pd.Timestamp(f'{match_date} {match_time}', tz='Asia/Shanghai')
        safe_deadline = kickoff - pd.Timedelta(minutes=10)
        now = pd.Timestamp.now(tz='Asia/Shanghai')
        minutes = (safe_deadline - now).total_seconds() / 60
        kickoff_minutes = (kickoff - now).total_seconds() / 60
        window_start = kickoff - pd.Timedelta(minutes=60)
        window_end = kickoff - pd.Timedelta(minutes=30)
        result['参考投注截止'] = safe_deadline.strftime('%m-%d %H:%M')
        result['距参考截止分钟'] = round(minutes, 1)
        result['建议临场同步时段'] = (
            f'{window_start.strftime("%m-%d %H:%M")}～'
            f'{window_end.strftime("%H:%M")}'
        )
        if 30 <= kickoff_minutes <= 60 and selling:
            result['同步时段'] = '临场增强窗口'
            result['投注时间提示'] = f'当前为最后赔率窗口，距开赛约{kickoff_minutes:.0f}分钟'
        elif kickoff_minutes > 60 and selling:
            result['投注时间提示'] = '尚早，请在建议临场时段重新同步后确认'
        elif minutes < 0 or not selling:
            result['同步时段'] = '停止推荐'
            result['投注时间提示'] = '已过参考线或官方已停售，请勿按本场下单'
        else:
            result['投注时间提示'] = f'临近参考截止，剩余约{max(0, minutes):.0f}分钟'
    except (TypeError, ValueError):
        pass
    return result


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
    model, config = _cached_league_model(model_db.league_id, model_id)
    if model is None:
        raise RuntimeError(f'找不到模型：{model_id}')
    return model.predict_proba(fixture)[0], config


@lru_cache(maxsize=16)
def _cached_model_database(league: str) -> ModelDatabase:
    """Read a league index once during one daily prediction batch."""
    return ModelDatabase(league)


@lru_cache(maxsize=96)
def _cached_league_model(league: str, model_id: str):
    """Load each trained classifier once instead of once per fixture."""
    return _cached_model_database(league).load_model(model_id)


def _clear_prediction_model_cache():
    # Training may replace checkpoints while the GUI stays open. A new sync
    # always starts with a fresh cache, then safely reuses models in that batch.
    _cached_league_model.cache_clear()
    _cached_model_database.cache_clear()


def _holdout_audit(config: Optional[dict]) -> dict:
    """Normalize legacy tuning metadata and newer train/test reports."""
    if not config:
        return {}
    train = config.get('train', {}) or {}
    tuning = train.get('tuning', {}) or {}
    test = train.get('test', {}) or {}
    return {
        'accuracy': tuning.get('test_accuracy', test.get('accuracy')),
        'baseline': tuning.get(
            'majority_baseline', test.get('majority_baseline'),
        ),
        'samples': tuning.get(
            'test_samples', tuning.get('test_sample_count', test.get('samples')),
        ),
        'p_value': tuning.get('mcnemar_p_value_vs_baseline'),
    }


def _sealed_audit_is_reliable(
        config: Optional[dict], *, minimum_accuracy: float | None = None,
        minimum_edge: float = 0.005,
) -> bool:
    """Apply one chronological holdout standard to every model family."""
    audit = _holdout_audit(config)
    accuracy, baseline = audit.get('accuracy'), audit.get('baseline')
    if accuracy is None or baseline is None:
        return False
    if minimum_accuracy is not None and float(accuracy) < minimum_accuracy:
        return False
    if float(accuracy) < float(baseline) + minimum_edge:
        return False
    samples = audit.get('samples')
    if samples is not None and int(samples) < 100:
        return False
    p_value = audit.get('p_value')
    if p_value is not None and float(p_value) > 0.10:
        return False
    return True


def _result_model_is_reliable(config: Optional[dict]) -> bool:
    """Reject a saved 1X2 model only when its own holdout audit is clearly weak."""
    return _sealed_audit_is_reliable(
        config, minimum_accuracy=0.50,
    )


def _over_under_model_is_reliable(config: Optional[dict]) -> bool:
    """Use a trained O/U model only when its sealed test beats its baseline."""
    return _sealed_audit_is_reliable(config)


def _score_model_is_reliable(config: Optional[dict]) -> bool:
    """Exact-score models must beat their sealed modal-score baseline."""
    return _sealed_audit_is_reliable(config)


def _half_full_model_is_reliable(config: Optional[dict]) -> bool:
    """Admit a half/full model to value screening only after a sealed audit.

    Half-time dutching is a new, high-variance strategy, so legacy checkpoints
    without comparable holdout metadata remain visible as references but may
    not be labelled as independently verified combination signals.
    """
    return _sealed_audit_is_reliable(config)


def _half_result_model_is_reliable(config: Optional[dict]) -> bool:
    """Enable a direct half-time model only after chronological validation."""
    if not _sealed_audit_is_reliable(config):
        return False
    samples = _holdout_audit(config).get('samples')
    if samples is None:
        return False
    tuning = (config or {}).get('train', {}).get('tuning', {})
    return bool(
        tuning.get('selective_validated', False)
        and int(tuning.get('selective_samples') or 0) >= 30
        and float(tuning.get('selective_accuracy') or 0.0) >= 0.55
    )


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


def _directional_score(score_probabilities, score_classes, prefer_over, excluded):
    """Choose a distinct assertive score consistent with the O/U direction."""
    candidates = []
    for index, target_class in enumerate(score_classes):
        score = class_to_score(target_class)
        if score in excluded:
            continue
        home, away = divmod(int(target_class), 7)
        total = home + away
        if (prefer_over and total >= 4) or (not prefer_over and total <= 1):
            candidates.append((float(score_probabilities[index]), score))
    return max(candidates, default=(float('nan'), ''), key=lambda item: item[0])


def _aggressive_upset_score(
        score_probabilities: np.ndarray,
        score_classes: np.ndarray,
        market_probabilities: np.ndarray,
        excluded: frozenset,
) -> Tuple[float, str]:
    """Select an evidence-backed high-scoring result in the coldest 1X2 side."""
    upset_outcome = int(np.argmin(market_probabilities))
    candidates = []
    for index, target_class in enumerate(score_classes):
        score = class_to_score(target_class)
        if score in excluded:
            continue
        home, away = divmod(int(target_class), 7)
        outcome = 0 if home > away else 1 if home == away else 2
        probability = float(score_probabilities[index])
        if home + away >= 4 and outcome == upset_outcome and probability >= 0.005:
            candidates.append((probability, score))
    return max(candidates, default=(float('nan'), ''), key=lambda item: item[0])


def _team_key(value: object) -> str:
    text = unicodedata.normalize('NFKD', str(value or ''))
    return re.sub(r'[^a-z0-9]', '', text.encode('ascii', 'ignore').decode().lower())


def _espn_team_matches(team: dict, candidates) -> bool:
    identifiers = [team.get('displayName'), team.get('shortDisplayName'),
                   team.get('name'), team.get('abbreviation')]
    left = [_team_key(value) for value in identifiers if _team_key(value)]
    expanded = list(candidates)
    for candidate in candidates:
        expanded.extend(ESPN_TEAM_ALIASES.get(_team_key(candidate), ()))
    right = [_team_key(value) for value in expanded if _team_key(value)]
    for source in left:
        for target in right:
            if source == target or (min(len(source), len(target)) >= 5 and (
                source in target or target in source
            )):
                return True
            if min(len(source), len(target)) >= 5 and SequenceMatcher(
                None, source, target,
            ).ratio() >= 0.86:
                return True
    return False


@lru_cache(maxsize=64)
def _espn_season(slug: str, year: int) -> dict:
    ESPN_HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    path = ESPN_HISTORY_ROOT / f'{slug}-{year}.json'
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    url = (
        f'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard'
        f'?limit=1000&dates={year}'
    )
    try:
        with urlopen(url, timeout=12) as response:
            payload = json.loads(response.read())
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        return payload
    except Exception as error:
        logging.warning('ESPN历史比分获取失败：%s %s，%s', slug, year, error)
        return {}


def _online_goal_history(
        display_league: str, home_candidates, away_candidates,
        match_date: Optional[date], home: str, away: str,
) -> Optional[pd.DataFrame]:
    slug = ESPN_LEAGUE_SLUGS.get(str(display_league))
    if not slug or match_date is None:
        return None
    rows = []
    for year in (match_date.year, match_date.year - 1):
        for event in _espn_season(slug, year).get('events', []):
            competition = (event.get('competitions') or [{}])[0]
            status = ((competition.get('status') or {}).get('type') or {})
            if not status.get('completed'):
                continue
            sides = {item.get('homeAway'): item for item in competition.get('competitors', [])}
            if 'home' not in sides or 'away' not in sides:
                continue
            try:
                event_date = pd.to_datetime(event.get('date'), utc=True).date()
                hg, ag = int(sides['home']['score']), int(sides['away']['score'])
            except (TypeError, ValueError, KeyError):
                continue
            if event_date >= match_date:
                continue
            home_team, away_team = sides['home'].get('team', {}), sides['away'].get('team', {})
            home_name = home if _espn_team_matches(home_team, home_candidates) else (
                away if _espn_team_matches(home_team, away_candidates) else home_team.get('displayName', '')
            )
            away_name = home if _espn_team_matches(away_team, home_candidates) else (
                away if _espn_team_matches(away_team, away_candidates) else away_team.get('displayName', '')
            )
            rows.append({'Date': event_date.isoformat(), 'Home': home_name,
                         'Away': away_name, 'HG': hg, 'AG': ag})
    frame = pd.DataFrame(rows).drop_duplicates(['Date', 'Home', 'Away']) if rows else None
    return frame if frame is not None and len(frame) >= 30 else None


def _historical_league_prior(
        history: Optional[pd.DataFrame], match_date: Optional[date],
) -> Optional[Tuple[float, float]]:
    if history is None or not {'Date', 'HG', 'AG'}.issubset(history.columns):
        return None
    frame = history[['Date', 'HG', 'AG']].copy()
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    frame['HG'] = pd.to_numeric(frame['HG'], errors='coerce')
    frame['AG'] = pd.to_numeric(frame['AG'], errors='coerce')
    frame = frame.dropna()
    if match_date is not None:
        frame = frame.loc[frame['Date'].dt.date < match_date]
    frame = frame.tail(1200)
    if len(frame) < 30:
        return None
    rates = float(frame['HG'].mean()), float(frame['AG'].mean())
    return rates if all(np.isfinite(value) and value > 0 for value in rates) else None


@lru_cache(maxsize=16)
def _portable_score_prior(match_date_text: str) -> Tuple[float, float]:
    """Cross-league real-score prior used only when a league cannot be resolved."""
    match_date = date.fromisoformat(match_date_text)
    rates = []
    for league_name in LEAGUE_ALIASES:
        try:
            rate = _historical_league_prior(
                LeagueDatabase().load_league(league_name), match_date,
            )
        except (OSError, KeyError, ValueError):
            rate = None
        if rate is not None:
            rates.append(rate)
    if rates:
        return tuple(np.mean(np.asarray(rates), axis=0).tolist())
    # Portable datasets normally make this unreachable. These constants are
    # the bundled datasets' long-run home/away goal means, not market inputs.
    return 1.45, 1.15


def _historical_goal_strengths(
        history: Optional[pd.DataFrame], home: str, away: str,
        match_date: Optional[date] = None,
) -> Optional[Tuple[float, float, int, int]]:
    """Estimate independent home/away goal rates from pre-match history only."""
    if history is None or history.empty or not home or not away:
        return None
    required = {'Date', 'Home', 'Away', 'HG', 'AG'}
    if not required.issubset(history.columns):
        return None
    frame = history[list(required)].copy()
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    frame['HG'] = pd.to_numeric(frame['HG'], errors='coerce')
    frame['AG'] = pd.to_numeric(frame['AG'], errors='coerce')
    frame = frame.dropna().sort_values('Date')
    if match_date is not None:
        frame = frame.loc[frame['Date'].dt.date < match_date]
    if frame.empty:
        return None
    # Recent league environment prevents old high/low-scoring eras from
    # dominating while retaining enough matches for a stable prior.
    league_recent = frame.tail(1200)
    league_home = float(league_recent['HG'].mean())
    league_away = float(league_recent['AG'].mean())
    if not np.isfinite(league_home + league_away) or min(league_home, league_away) <= 0:
        return None
    home_rows = frame.loc[frame['Home'].astype(str).eq(str(home))].tail(18)
    away_rows = frame.loc[frame['Away'].astype(str).eq(str(away))].tail(18)
    if len(home_rows) < 3 or len(away_rows) < 3:
        return None

    def shrink(values: pd.Series, prior: float, strength: float = 7.0) -> float:
        sample = pd.to_numeric(values, errors='coerce').dropna()
        # Recent matches carry more information, but the league prior prevents
        # a short hot/cold streak from producing implausible scoring rates.
        weights = np.power(0.5, np.arange(len(sample) - 1, -1, -1) / 6.0)
        weighted_sum = float(np.dot(sample.to_numpy(dtype=float), weights))
        return float((weighted_sum + strength * prior) / (weights.sum() + strength))

    home_scored = shrink(home_rows['HG'], league_home)
    home_conceded = shrink(home_rows['AG'], league_away)
    away_scored = shrink(away_rows['AG'], league_away)
    away_conceded = shrink(away_rows['HG'], league_home)
    home_lambda = home_scored * away_conceded / league_home
    away_lambda = away_scored * home_conceded / league_away
    return (
        float(np.clip(home_lambda, 0.20, 4.20)),
        float(np.clip(away_lambda, 0.20, 4.20)),
        len(home_rows), len(away_rows),
    )


def _league_dixon_coles_rho(
    history: Optional[pd.DataFrame],
    match_date: Optional[date],
) -> float:
    """Estimate a conservative league-level low-score dependency.

    It is intentionally fitted on historical matches only, never to the
    current fixture or its odds.  Shrinkage towards zero avoids unstable rho
    values in small datasets.
    """
    if history is None or not {'Date', 'HG', 'AG'}.issubset(history.columns):
        return 0.0
    frame = history[['Date', 'HG', 'AG']].copy()
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
    frame['HG'] = pd.to_numeric(frame['HG'], errors='coerce')
    frame['AG'] = pd.to_numeric(frame['AG'], errors='coerce')
    frame = frame.dropna().sort_values('Date')
    if match_date is not None:
        frame = frame.loc[frame['Date'].dt.date < match_date]
    frame = frame.tail(1200)
    if len(frame) < 80:
        return 0.0
    home_rate = float(frame['HG'].mean())
    away_rate = float(frame['AG'].mean())
    low = frame.loc[frame['HG'].le(1) & frame['AG'].le(1)]
    counts = low.groupby(['HG', 'AG']).size().to_dict()
    best_rho, best_score = 0.0, -float('inf')
    for rho in np.linspace(-0.15, 0.15, 121):
        tau = {
            (0.0, 0.0): 1.0 - home_rate * away_rate * rho,
            (0.0, 1.0): 1.0 + home_rate * rho,
            (1.0, 0.0): 1.0 + away_rate * rho,
            (1.0, 1.0): 1.0 - rho,
        }
        if min(tau.values()) <= 0:
            continue
        score = sum(counts.get(cell, 0) * np.log(value) for cell, value in tau.items())
        # Equivalent to about 160 neutral observations: enough to stop a
        # noisy league window from pinning rho at the search boundary.
        score -= 80.0 * rho * rho
        if score > best_score:
            best_rho, best_score = float(rho), float(score)
    return best_rho


def _dixon_coles_sample_weights(
    homes: np.ndarray,
    aways: np.ndarray,
    home_rates: np.ndarray,
    away_rates: np.ndarray,
    rho: float,
) -> np.ndarray:
    weights = np.ones(len(homes), dtype=np.float64)
    if abs(rho) < 1e-12:
        return weights
    masks_and_values = (
        ((homes == 0) & (aways == 0), 1.0 - home_rates * away_rates * rho),
        ((homes == 0) & (aways == 1), 1.0 + home_rates * rho),
        ((homes == 1) & (aways == 0), 1.0 + away_rates * rho),
        ((homes == 1) & (aways == 1), np.full(len(homes), 1.0 - rho)),
    )
    for mask, values in masks_and_values:
        weights[mask] = np.asarray(values)[mask]
    return np.clip(weights, 0.05, 3.0)


def _monte_carlo_summary(
        history: Optional[pd.DataFrame],
        home: str,
        away: str,
        match_date: Optional[date],
        lineup_shift: float,
        lineup_confirmed: bool,
        handicap_line: Optional[float],
    seed_value: object,
    simulations: int = 10_000,
    fallback_goal_rates: Optional[Tuple[float, float]] = None,
    historical_prior_rates: Optional[Tuple[float, float]] = None,
    historical_prior_source: str = '',
) -> dict:
    """Run a strictly independent simulation from pre-match team history only.

    The legacy arguments for lineup and fallback rates remain in the signature
    for compatibility, but deliberately do not affect the simulation. Market
    odds, trained-model outputs and lineup judgements must never be fed back
    into columns labelled as independent Monte Carlo data.
    """
    strengths = _historical_goal_strengths(history, home, away, match_date)
    prior_used = False
    if strengths is None:
        rates = historical_prior_rates or ()
        if len(rates) != 2 or not all(np.isfinite(value) and value > 0 for value in rates):
            return {
                '模拟次数': 0, '模拟Top3比分': '', '模拟胜负': '', '模拟让球': '',
                '模拟总进球': '', '模拟竞彩总进球': '',
                '模拟竞彩总进球概率': float('nan'),
                '模拟半全场': '', '模拟可信度': '',
                '模拟半场胜概率': float('nan'),
                '模拟半场平概率': float('nan'),
                '模拟半场负概率': float('nan'),
                '模拟最高赛果概率': float('nan'),
                '模拟模型来源': '历史攻防样本不足（未使用赔率/模型兜底）',
            }
        base_home, base_away = map(float, rates)
        home_samples = away_samples = 0
        prior_used = True
    else:
        base_home, base_away, home_samples, away_samples = strengths

    seed = zlib.crc32(str(seed_value).encode('utf-8'))
    rng = np.random.default_rng(seed)
    # Shared tempo produces correlated high/low-scoring scenarios; separate
    # team shocks represent day-of-match finishing and defensive variation.
    tempo = rng.lognormal(mean=-0.5 * 0.18 ** 2, sigma=0.18, size=simulations)
    home_shock = rng.lognormal(mean=-0.5 * 0.12 ** 2, sigma=0.12, size=simulations)
    away_shock = rng.lognormal(mean=-0.5 * 0.12 ** 2, sigma=0.12, size=simulations)
    home_lambda = np.clip(base_home * tempo * home_shock, 0.03, 5.5)
    away_lambda = np.clip(base_away * tempo * away_shock, 0.03, 5.5)
    half_homes = rng.poisson(home_lambda * 0.45)
    half_aways = rng.poisson(away_lambda * 0.45)
    homes = half_homes + rng.poisson(home_lambda * 0.55)
    aways = half_aways + rng.poisson(away_lambda * 0.55)

    rho = _league_dixon_coles_rho(history, match_date)
    sample_weights = _dixon_coles_sample_weights(
        homes, aways, home_lambda, away_lambda, rho,
    )
    weight_total = float(sample_weights.sum())

    encoded_scores = np.minimum(homes, 6) * 7 + np.minimum(aways, 6)
    counts = np.bincount(encoded_scores, weights=sample_weights, minlength=49)
    simulated_score_probability = counts / weight_total
    top_columns = np.argsort(simulated_score_probability)[::-1][:3]

    result_index = np.where(homes > aways, 0, np.where(homes == aways, 1, 2))
    result_probability = (
        np.bincount(result_index, weights=sample_weights, minlength=3)
        / weight_total
    )
    result_pick = int(np.argmax(result_probability))
    totals = homes + aways
    total_bands = np.array([
        sample_weights[totals <= 1].sum() / weight_total,
        sample_weights[(totals >= 2) & (totals <= 3)].sum() / weight_total,
        sample_weights[totals >= 4].sum() / weight_total,
    ])
    total_labels = ('0-1球', '2-3球', '4球以上')
    lottery_total_index = np.minimum(totals, 7).astype(int)
    lottery_total_probability = (
        np.bincount(
            lottery_total_index, weights=sample_weights, minlength=8,
        ) / weight_total
    )
    lottery_total_pick = int(np.argmax(lottery_total_probability))
    lottery_total_labels = ('0球', '1球', '2球', '3球', '4球', '5球', '6球', '7+球')

    half_result = np.where(
        half_homes > half_aways, 0,
        np.where(half_homes == half_aways, 1, 2),
    )
    half_full_index = half_result * 3 + result_index
    half_probability = np.bincount(
        half_full_index, weights=sample_weights, minlength=len(HALF_FULL_LABELS),
    ) / weight_total
    half_result_probability = np.bincount(
        half_result, weights=sample_weights, minlength=3,
    ) / weight_total
    half_top = np.argsort(half_probability)[::-1][:2]

    handicap_text = ''
    if handicap_line is not None and np.isfinite(handicap_line):
        difference = homes + float(handicap_line) - aways
        handicap_index = np.where(
            difference > 0, 0, np.where(difference == 0, 1, 2),
        )
        handicap_probability = (
            np.bincount(handicap_index, weights=sample_weights, minlength=3)
            / weight_total
        )
        handicap_pick = int(np.argmax(handicap_probability))
        handicap_text = (
            f'让{OUTCOME_LABELS[handicap_pick]} '
            f'{handicap_probability[handicap_pick]:.1%}'
        )
    maximum_result = float(result_probability.max())
    confidence_score = (
        5 if maximum_result >= 0.70 else 4 if maximum_result >= 0.60
        else 3 if maximum_result >= 0.52 else 2 if maximum_result >= 0.45 else 1
    )
    if prior_used:
        confidence_score = min(confidence_score, 2)
    elif min(home_samples, away_samples) < 8:
        confidence_score = min(confidence_score, 3)
    return {
        '模拟次数': simulations,
        '模拟Top3比分': ' / '.join(
            f'{class_to_score(index)} '
            f'{simulated_score_probability[index]:.1%}'
            for index in top_columns
        ),
        '模拟胜负': f'{OUTCOME_LABELS[result_pick]} {result_probability[result_pick]:.1%}',
        '模拟让球': handicap_text,
        '模拟总进球': (
            f'{total_labels[int(np.argmax(total_bands))]} '
            f'{float(total_bands.max()):.1%}'
        ),
        '模拟竞彩总进球': lottery_total_labels[lottery_total_pick],
        '模拟竞彩总进球概率': float(lottery_total_probability[lottery_total_pick]),
        '模拟半全场': ' / '.join(
            f'{HALF_FULL_LABELS[index]} {half_probability[index]:.1%}'
            for index in half_top
        ),
        '模拟半场胜概率': float(half_result_probability[0]),
        '模拟半场平概率': float(half_result_probability[1]),
        '模拟半场负概率': float(half_result_probability[2]),
        '模拟可信度': '★' * confidence_score + '☆' * (5 - confidence_score),
        '模拟最高赛果概率': maximum_result,
        '模拟模型来源': (
            f'{historical_prior_source or "真实比分历史先验"}（球队样本不足，低置信；'
            f'未使用赔率/正式模型/首发校正）'
            if prior_used else (
                f'历史攻防双泊松蒙特卡洛（近期加权；主场{home_samples}场/'
                f'客场{away_samples}场；联赛DCρ={rho:+.3f}；'
                f'未使用赔率/正式模型/首发校正）'
            )
        ),
    }


def backfill_missing_simulations(predictions: pd.DataFrame) -> pd.DataFrame:
    """Populate legacy report rows whose independent simulation was left blank."""
    if predictions.empty:
        return predictions.copy()
    result = predictions.copy()
    simulation_columns = (
        '模拟次数', '模拟Top3比分', '模拟胜负', '模拟让球', '模拟总进球',
        '模拟竞彩总进球', '模拟竞彩总进球概率',
        '模拟半全场', '模拟半场胜概率', '模拟半场平概率',
        '模拟半场负概率', '模拟可信度', '模拟最高赛果概率', '模拟模型来源',
    )
    for column in simulation_columns:
        if column not in result.columns:
            result[column] = ''
    missing = (
        result['模拟胜负'].fillna('').astype(str).str.strip().eq('')
        | result['模拟竞彩总进球'].fillna('').astype(str).str.strip().eq('')
    )
    for index, row in result.loc[missing].iterrows():
        raw_date = str(row.get('比赛时间') or '')[:10]
        try:
            match_day = date.fromisoformat(raw_date)
        except ValueError:
            match_day = date.today()
        prior = _portable_score_prior(match_day.isoformat())
        handicap = pd.to_numeric(row.get('官方让球数'), errors='coerce')
        summary = _monte_carlo_summary(
            None,
            str(row.get('主队模型名') or row.get('主队') or ''),
            str(row.get('客队模型名') or row.get('客队') or ''),
            match_day, 0.0, False,
            float(handicap) if pd.notna(handicap) else None,
            row.get('比赛ID') or row.get('赛事编号') or index,
            historical_prior_rates=prior,
            historical_prior_source='本地跨联赛真实比分先验',
        )
        for column in simulation_columns:
            result.at[index, column] = summary[column]
    return result


def _diverse_score_ranking(
        score_probabilities: np.ndarray,
        score_classes: np.ndarray,
) -> np.ndarray:
    """Keep the modal score, then cover a stable and a different-result script."""
    ranked = list(np.argsort(score_probabilities)[::-1])
    if len(ranked) <= 2:
        return np.asarray(ranked, dtype=np.int32)
    first = ranked[0]
    first_home, first_away = divmod(int(score_classes[first]), 7)
    first_outcome = 0 if first_home > first_away else 1 if first_home == first_away else 2
    second = ranked[1]
    third = next((
        index for index in ranked[1:]
        if index != second and (
            0 if divmod(int(score_classes[index]), 7)[0]
            > divmod(int(score_classes[index]), 7)[1]
            else 1 if divmod(int(score_classes[index]), 7)[0]
            == divmod(int(score_classes[index]), 7)[1] else 2
        ) != first_outcome
    ), ranked[2])
    return np.asarray([first, second, third], dtype=np.int32)


def _calibrate_score_probabilities(
        score_probabilities: np.ndarray,
        score_classes: np.ndarray,
        result_probabilities: np.ndarray,
        over_under_probabilities: np.ndarray,
) -> np.ndarray:
    """Align exact scores with the independently calibrated 1X2 and O/U views.

    Iterative proportional fitting preserves the model's within-group score
    ordering while preventing a generic modal score from ignoring the match's
    result and goal-total evidence.
    """
    values = np.asarray(score_probabilities, dtype=np.float64).copy()
    classes = np.asarray(score_classes, dtype=np.int32)
    result_target = np.asarray(result_probabilities, dtype=np.float64)
    total_target = np.asarray(over_under_probabilities, dtype=np.float64)
    if (
        values.ndim != 1 or values.size != classes.size or values.size == 0
        or result_target.shape != (3,) or total_target.shape != (2,)
        or not np.isfinite(values).all() or values.sum() <= 0
    ):
        return values
    values = np.maximum(values, 1e-12)
    values /= values.sum()
    homes, aways = classes // 7, classes % 7
    outcomes = np.where(homes > aways, 0, np.where(homes == aways, 1, 2))
    totals = (homes + aways > 2).astype(int)
    for _ in range(20):
        for group, targets in ((outcomes, result_target), (totals, total_target)):
            for index, target in enumerate(targets):
                mask = group == index
                current = float(values[mask].sum())
                if current > 0 and np.isfinite(target) and target >= 0:
                    values[mask] *= float(target) / current
        values /= values.sum()
    return values


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
        odds_series: Optional[dict] = None,
        lineup_analysis: Optional[dict] = None,
        regular_market_offered: bool = True,
        data_feed_source: str = '官方实时接口',
) -> dict:
    market_prob = _devig(odds)
    preliminary_flow = market_flow_gate(
        _field(raw, 'matchId'), '', series=odds_series,
    )
    market_quality = market_quality_metrics(
        _field(raw, 'matchId'), series=odds_series,
    )
    match_series = (odds_series or {}).get(str(_field(raw, 'matchId')), [])
    opening_snapshot = match_series[0] if match_series else {}
    selection_league = display_league or league or ''
    market_selection = _market_selection(float(market_prob.max()), selection_league)
    model_db = None
    fixture = None
    score_classes = None
    prediction_basis = '历史数据训练模型'
    confidence = '正常'
    estimated_teams = []
    simulation_history = None
    trained_result_active = False
    dedicated_model_league = ''
    result_model_category = '市场基线'
    result_model_status = '市场基线'
    over_under_model_status = '市场基线'
    score_model_status = '市场基线'
    half_full_model_status = '市场基线'
    half_result_model_status = '未启用，使用半全场聚合概率'
    half_result_source = '半全场概率聚合（非独立半场模型）'
    direct_half_result_probability = None
    half_result_threshold = float('nan')
    half_full_source = (
        '官方半全场市场基线'
        if _official_hafu_half_full(raw.get('hafu')) is not None
        else '泊松市场拟合'
    )
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
            result_model_status = '欧战校准模型启用'
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
                result_model_status = '通用模型启用'
    else:
        raw_history = LeagueDatabase().load_league(league)
        simulation_history = raw_history
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
            baseline = _market_baseline_probabilities(
                odds, raw.get('ttg'), raw.get('crs'), raw.get('hafu'),
            )
            fixture = construct_inputs_by_fixture(
                history,
                pd.DataFrame([{
                    'Home': home, 'Away': away,
                    '1': odds['H'], 'X': odds['D'], '2': odds['A'],
                }]),
            )
            model_db = _cached_model_database(league)
            result_prob, result_config = _model_probabilities(
                model_db, f'{league}胜平负模型', fixture,
            )
            ou_prob, ou_config = _model_probabilities(
                model_db, f'{league}大小球模型', fixture,
            )
            if not _over_under_model_is_reliable(ou_config):
                ou_prob = baseline['over_under']
                over_under_model_status = '弱模型已禁用，回退市场基线'
            else:
                over_under_model_status = f'{league}专用模型启用'
            score_prob, score_config = _model_probabilities(
                model_db, f'{league}比分模型', fixture,
            )
            if not _score_model_is_reliable(score_config):
                score_prob = baseline['score']
                score_classes = baseline['score_classes']
                score_model_status = '弱模型已禁用，回退市场基线'
            else:
                score_model_status = f'{league}专用模型启用'
            half_full_prob, half_full_config = _model_probabilities(
                model_db, f'{league}半全场模型', fixture,
            )
            if _half_full_model_is_reliable(half_full_config):
                half_full_source = f'{league}专用半全场模型（已验证）'
                half_full_model_status = f'{league}专用模型启用'
            else:
                half_full_prob = baseline['half_full']
                half_full_source = '市场半全场基线（弱模型已禁用）'
                half_full_model_status = '弱模型已禁用，回退市场基线'
            try:
                candidate_half_result, half_result_config = _model_probabilities(
                    model_db, f'{league}半场胜平负模型', fixture,
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                half_result_config = None
            if _half_result_model_is_reliable(half_result_config):
                direct_half_result_probability = np.asarray(
                    candidate_half_result, dtype=np.float64,
                )
                direct_half_result_probability /= direct_half_result_probability.sum()
                half_result_source = f'{league}专用半场胜平负模型（已验证）'
                half_result_model_status = f'{league}专用半场模型启用'
                half_result_threshold = float(
                    half_result_config['train']['tuning']['selective_threshold']
                )
            elif half_result_config:
                half_result_model_status = '半场模型未通过独立测试，使用半全场聚合概率'
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
                result_model_status = '弱模型/审计缺失，回退市场基线'
            elif not model_result_is_allowed(f'{league}专用模型'):
                result_prob = market_prob
                result_model_category = '市场基线'
                prediction_basis = '专用模型近期实战低于市场，自动回退官方赔率'
                confidence = '较低'
                result_model_status = '近期实战治理禁用，回退市场基线'
            else:
                category = f'{league}专用模型'
                blend_weight = model_result_blend_weight(category)
                if blend_weight <= 0.0:
                    result_prob = market_prob
                    result_model_category = '市场基线'
                    prediction_basis = '专用模型尚未证明高于市场，使用官方赔率市场基线'
                    confidence = '正常'
                    result_model_status = '尚未证明优势，回退市场基线'
                else:
                    result_prob = (
                        blend_weight * np.asarray(result_prob, dtype=np.float64)
                        + (1.0 - blend_weight) * market_prob
                    )
                    result_prob = _calibrate_draw_probability(
                        result_prob, market_prob, history,
                        league=selection_league, home=home_cn, away=away_cn,
                        draw_flow=preliminary_flow.get('draw_change') or 0.0,
                        hhad_line_change=market_quality.get('hhad_line_change') or 0.0,
                        ttg_expected_change=market_quality.get('ttg_expected_change') or 0.0,
                    )
                    prediction_basis = (
                        f'{category}{blend_weight:.0%} + 官方赔率{1.0-blend_weight:.0%}'
                        '，含联赛平局校准'
                    )
                    trained_result_active = True
                    result_model_status = f'{league}专用模型启用'

    lineup_analysis = lineup_analysis or {}
    lineup_shift = float(lineup_analysis.get('probability_shift') or 0.0)
    result_prob_before_lineup = np.asarray(result_prob, dtype=float).copy()
    if lineup_analysis.get('status') == '已确认' and lineup_shift:
        # Shift only between home and away, retain the calibrated draw mass and
        # cap the adjustment in the lineup service at four percentage points.
        result_prob = np.asarray(result_prob, dtype=np.float64).copy()
        result_prob[0] = max(0.03, result_prob[0] + lineup_shift)
        result_prob[2] = max(0.03, result_prob[2] - lineup_shift)
        result_prob /= result_prob.sum()
        prediction_basis += f'；确认首发校正{lineup_shift:+.1%}'

    edge = result_prob - market_prob
    best = int(edge.argmax())
    if score_classes is None:
        score_classes = np.asarray(
            _cached_league_model(league, f'{league}比分模型')[0].classifier.classes_,
            dtype=np.int32,
        )
    score_prob = _calibrate_score_probabilities(
        score_prob, score_classes, result_prob, ou_prob,
    )
    raw_top_score_column = int(np.argmax(score_prob))
    score_recommendation_active = bool(
        float(score_prob[raw_top_score_column]) >= 0.12
    )
    prefer_over = bool(ou_prob[1] >= ou_prob[0])
    # Exact-score ranking must follow the score distribution itself. The old
    # rule forced every displayed score onto the O/U side; live review showed
    # that reduced first-score accuracy from 10.53% to 8.77%. O/U still has a
    # separate directional score below and must not overwrite the main pick.
    top_score_columns = _diverse_score_ranking(score_prob, score_classes)
    top_scores = [
        f'{class_to_score(score_classes[i])} {score_prob[i]:.1%}'
        for i in top_score_columns
    ]
    ranked_scores = [class_to_score(score_classes[i]) for i in top_score_columns]
    lottery_total_probability = np.zeros(8, dtype=np.float64)
    for score_class, probability in zip(score_classes, score_prob):
        score_text = class_to_score(score_class)
        score_match = re.fullmatch(r'(\d+)-(\d+)', score_text)
        if score_match is None:
            continue
        home_goals, away_goals = map(int, score_match.groups())
        lottery_total_probability[min(home_goals + away_goals, 7)] += float(probability)
    lottery_total_pick = int(np.argmax(lottery_total_probability))
    lottery_total_labels = ('0球', '1球', '2球', '3球', '4球', '5球', '6球', '7+球')
    top_half_full = np.argsort(half_full_prob)[-3:][::-1]
    ranked_half_full = [HALF_FULL_LABELS[i] for i in top_half_full]
    aggregated_half_result_probability = np.asarray(
        half_full_prob, dtype=np.float64,
    ).reshape(3, 3).sum(axis=1)
    half_result_probability = (
        direct_half_result_probability
        if direct_half_result_probability is not None
        else aggregated_half_result_probability
    )
    result_index = select_result_index(result_prob, market_prob, ou_prob[0])
    result_pick = OUTCOME_LABELS[result_index]
    draw_gate_pick = result_pick == '平' and draw_gate_applies(
        result_prob, market_prob, ou_prob[0],
    )
    draw_protection = draw_protection_pick(result_prob, market_prob, ou_prob[0])
    final_selection = _market_selection(
        float(np.max(result_prob)), selection_league,
    )
    flow_gate = market_flow_gate(
        _field(raw, 'matchId'), result_pick, series=odds_series,
    )
    advice = final_selection['grade']
    # Strict recommendation gate: only two audited tiers can be called a pick.
    if advice not in ('精选主推', '高置信主推'):
        advice = '观察' if advice == '观察' else '跳过'
    if flow_gate['state'] in ('conflict', 'unstable'):
        advice = '跳过'
    # The draw gate already passed a sealed chronological test. Do not require
    # a second positive flow signal: stable or sparse snapshots are neutral,
    # while genuine conflict/instability is still rejected above. Because the
    # sealed draw precision is lower than the main-pick tiers, neutral-flow
    # draws remain observations rather than being mislabeled as strong picks.
    if (
            draw_gate_pick
            and advice == '跳过'
            and flow_gate['state'] in ('agree', 'stable', 'insufficient')
    ):
        advice = '观察'
    lineup_conflict = bool(
        abs(lineup_shift) >= 0.016
        and (
            (result_pick == '胜' and lineup_shift < 0)
            or (result_pick == '负' and lineup_shift > 0)
            or result_pick == '平'
        )
    )
    lineup_high_warning = lineup_analysis.get('warning_level') == '高'
    if lineup_conflict or lineup_high_warning:
        advice = '跳过'
    sale = _sale_context(raw)
    # Accuracy-first mode: an early-board prediction remains visible for
    # analysis, but only a sync inside the verified late window may be a pick.
    if sale['同步时段'] != '临场增强窗口':
        advice = '跳过'
    conclusion_parts = [
        advice,
        f'{result_pick} {float(result_prob[result_index]):.1%}',
    ]
    if lineup_analysis.get('status') == '已确认':
        if lineup_shift:
            direction = '主队利好' if lineup_shift > 0 else '客队利好'
            conclusion_parts.append(f'首发{direction}，已修正{abs(lineup_shift):.1%}')
        else:
            conclusion_parts.append('首发已核验，暂无可靠修正')
    elif lineup_analysis.get('status') == '待公布':
        conclusion_parts.append('首发待公布')
    if lineup_conflict:
        conclusion_parts.append('阵容与原方向冲突')
    elif lineup_high_warning:
        conclusion_parts.append('阵容高风险预警，已撤下重点')
    if flow_gate['state'] == 'conflict':
        conclusion_parts.append('盘口反向')
    elif flow_gate['state'] == 'unstable':
        conclusion_parts.append('盘口不稳')
    ou_pick = '大于2.5球' if prefer_over else '小于2.5球'
    upset_score, upset_score_probability = _upset_score(
        score_prob, score_classes, market_prob,
        excluded=frozenset(ranked_scores[:3]),
    )
    directional_probability, directional_score = _aggressive_upset_score(
        score_prob, score_classes, market_prob,
        frozenset([*ranked_scores[:3], upset_score]),
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
            _cached_league_model(league, handicap_model_id)[0]
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
    # The comparison simulation is intentionally isolated from the official
    # odds/model pipeline. Real-score league/cross-league priors keep sparse
    # fixtures populated without feeding market or trained-model probabilities.
    simulation_source = '本地历史'
    if _historical_goal_strengths(
        simulation_history, home, away, _match_date(raw),
    ) is None:
        online_history = _online_goal_history(
            selection_league,
            (home, home_cn, _field(raw, 'homeTeamAllName', 'homeTeamAbbName'),
             _field(raw, 'homeTeamAbbEnName', 'homeTeamCode')),
            (away, away_cn, _field(raw, 'awayTeamAllName', 'awayTeamAbbName'),
             _field(raw, 'awayTeamAbbEnName', 'awayTeamCode')),
            _match_date(raw), home, away,
        )
        if online_history is not None:
            simulation_history = online_history
            simulation_source = 'ESPN联网真实赛果'
    historical_prior = None
    historical_prior_source = ''
    if _historical_goal_strengths(
        simulation_history, home, away, _match_date(raw),
    ) is None:
        historical_prior = _historical_league_prior(
            simulation_history, _match_date(raw),
        )
        if historical_prior is not None:
            historical_prior_source = f'{simulation_source}联赛真实比分先验'
        else:
            historical_prior = _portable_score_prior(_match_date(raw).isoformat())
            historical_prior_source = '本地跨联赛真实比分先验'
    monte_carlo = _monte_carlo_summary(
        simulation_history, home, away, _match_date(raw), 0.0, False,
        handicap_odds['line'] if handicap_odds else None,
        _field(raw, 'matchId', default=f'{home_cn}-{away_cn}'),
        historical_prior_rates=historical_prior,
        historical_prior_source=historical_prior_source,
    )
    if (
        monte_carlo.get('模拟次数') and simulation_source != '本地历史'
        and historical_prior is None
    ):
        monte_carlo['模拟模型来源'] = (
            f'{simulation_source}｜{monte_carlo["模拟模型来源"]}'
        )
    official_hafu_odds = {}
    raw_hafu = raw.get('hafu') or {}
    for label, key in HAFU_ODD_KEYS:
        try:
            odd = float(raw_hafu[key])
        except (KeyError, TypeError, ValueError):
            odd = float('nan')
        official_hafu_odds[f'官方半全场{label}奖金'] = (
            odd if np.isfinite(odd) and odd > 1.0 else float('nan')
        )
    monte_risks = []
    if monte_carlo['模拟最高赛果概率'] < 0.50:
        monte_risks.append('胜负分散')
    if float(score_prob[raw_top_score_column]) < 0.12:
        monte_risks.append('比分离散')
    if flow_gate['state'] in ('conflict', 'unstable'):
        monte_risks.append('盘口冲突/震荡')
    if lineup_analysis.get('status') != '已确认':
        monte_risks.append('首发未确认')
    return {
        '赛事编号': _field(raw, 'matchNumStr', 'matchNum'),
        '比赛ID': _field(raw, 'matchId'),
        '比赛时间': (
            f'{str(raw.get("matchDate") or "")[:10]} '
            f'{str(raw.get("matchTime") or "")[:5]}'
        ).strip() or _field(raw, 'matchDateTime', 'startTime'),
        '联赛': display_league or league or str(_field(
            raw, 'leagueAllName', 'leagueName', 'leagueAbbName', default='未识别联赛',
        )),
        '主队': home_cn,
        '客队': away_cn,
        '主队模型名': home,
        '客队模型名': away,
        '官方胜奖金': odds['H'] if regular_market_offered else np.nan,
        '官方平奖金': odds['D'] if regular_market_offered else np.nan,
        '官方负奖金': odds['A'] if regular_market_offered else np.nan,
        # The official selling feed exposes current values only.  These are
        # explicitly named "首次采集" rather than being presented as a
        # guaranteed bookmaker opening line.
        '首次采集胜奖金': (opening_snapshot.get('had') or {}).get('H', np.nan),
        '首次采集平奖金': (opening_snapshot.get('had') or {}).get('D', np.nan),
        '首次采集负奖金': (opening_snapshot.get('had') or {}).get('A', np.nan),
        '模型主胜概率': result_prob[0],
        '模型平局概率': result_prob[1],
        '模型客胜概率': result_prob[2],
        '胜平负首选': result_pick,
        '胜平负首选概率': float(result_prob[result_index]),
        '平局双选保护': draw_protection,
        '平局保护触发': bool(draw_protection),
        '最终结论': '｜'.join(conclusion_parts),
        '市场去水主胜概率': market_prob[0],
        '市场去水平局概率': market_prob[1],
        '市场去水客胜概率': market_prob[2],
        '最大价值方向': (
            OUTCOME_LABELS[best]
            if trained_result_active and regular_market_offered else ''
        ),
        '最大概率优势': (
            edge[best] if trained_result_active and regular_market_offered else np.nan
        ),
        '建议状态': advice,
        '盘口门控': flow_gate['label'],
        '首发状态': lineup_analysis.get('status', '未获取'),
        '阵容分析': lineup_analysis.get('summary', '未到首发公布时间'),
        '阵容预警级别': lineup_analysis.get('warning_level', '无'),
        '阵容预警': '；'.join(lineup_analysis.get('warnings') or []),
        '阵容调整前胜概率': result_prob_before_lineup[0],
        '阵容调整前平概率': result_prob_before_lineup[1],
        '阵容调整前负概率': result_prob_before_lineup[2],
        '阵容调整后胜概率': result_prob[0],
        '阵容调整后平概率': result_prob[1],
        '阵容调整后负概率': result_prob[2],
        '主队阵型': lineup_analysis.get('home_formation', ''),
        '客队阵型': lineup_analysis.get('away_formation', ''),
        '主队首发': lineup_analysis.get('home_starting', ''),
        '客队首发': lineup_analysis.get('away_starting', ''),
        '主队轮换数': lineup_analysis.get('home_rotation'),
        '客队轮换数': lineup_analysis.get('away_rotation'),
        '主队核心缺阵数': lineup_analysis.get('home_missing_core', 0),
        '客队核心缺阵数': lineup_analysis.get('away_missing_core', 0),
        '主队门将变化': lineup_analysis.get('home_goalkeeper_changed', False),
        '客队门将变化': lineup_analysis.get('away_goalkeeper_changed', False),
        '阵容方向冲突': lineup_conflict,
        '阵容概率校正': lineup_shift,
        '盘口变化速度/小时': flow_gate.get('speed_per_hour'),
        '平局概率变化': flow_gate.get('draw_change'),
        '赔率快照数': flow_gate.get('observations', 0),
        '官方赔率返还率': market_quality.get('return_rate'),
        '让球线变化': market_quality.get('hhad_line_change'),
        '总进球预期变化': market_quality.get('ttg_expected_change'),
        '赔率来源数': market_quality.get('source_count', 0),
        '多公司数据可用': market_quality.get('multi_company_available', False),
        **sale,
        '预测依据': prediction_basis,
        '专用模型联赛': dedicated_model_league,
        '模型类别': (
            f'{dedicated_model_league}专用模型'
            if dedicated_model_league else '通用/市场模型'
        ),
        '胜负模型类别': result_model_category,
        '胜负模型状态': result_model_status,
        '大小球模型状态': over_under_model_status,
        '比分模型状态': score_model_status,
        '半全场模型状态': half_full_model_status,
        '半场模型状态': half_result_model_status,
        '模型治理状态': '；'.join(dict.fromkeys((
            result_model_status, over_under_model_status,
            score_model_status, half_full_model_status, half_result_model_status,
        ))),
        '置信等级': confidence,
        '估算球队': '、'.join(estimated_teams),
        '市场最高概率': float(market_prob.max()),
        '市场筛选阈值': final_selection['threshold'],
        '同阈值历史命中率': final_selection['accuracy'],
        '同阈值历史覆盖率': final_selection['coverage'],
        '筛选回测样本数': final_selection['samples'],
        '筛选回测期间': final_selection['period'],
        '官方让球数': handicap_odds['line'] if handicap_odds else np.nan,
        '官方让胜奖金': handicap_odds['H'] if handicap_odds else np.nan,
        '官方让平奖金': handicap_odds['D'] if handicap_odds else np.nan,
        '官方让负奖金': handicap_odds['A'] if handicap_odds else np.nan,
        '首次采集让球数': (opening_snapshot.get('hhad') or {}).get('line', np.nan),
        '首次采集让胜奖金': (opening_snapshot.get('hhad') or {}).get('H', np.nan),
        '首次采集让平奖金': (opening_snapshot.get('hhad') or {}).get('D', np.nan),
        '首次采集让负奖金': (opening_snapshot.get('hhad') or {}).get('A', np.nan),
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
        '竞彩总进球首选': lottery_total_labels[lottery_total_pick],
        '竞彩总进球首选概率': float(lottery_total_probability[lottery_total_pick]),
        '半全场首选': ranked_half_full[0],
        '半全场次选': ranked_half_full[1],
        '半全场首选概率': float(half_full_prob[top_half_full[0]]),
        '半全场次选概率': float(half_full_prob[top_half_full[1]]),
        '半全场第三选择': ranked_half_full[2],
        '半全场Top3': ' / '.join(
            f'{HALF_FULL_LABELS[i]} {half_full_prob[i]:.1%}' for i in top_half_full
        ),
        '半全场模型来源': half_full_source,
        '半场模型来源': half_result_source,
        '半场模型高置信门槛': half_result_threshold,
        '半场模型当前置信度': float(np.max(half_result_probability)),
        '正式半场胜概率': float(half_result_probability[0]),
        '正式半场平概率': float(half_result_probability[1]),
        '正式半场负概率': float(half_result_probability[2]),
        **official_hafu_odds,
        '首选比分': ranked_scores[0],
        '比分推荐状态': '推荐' if score_recommendation_active else '可信度不足',
        '比分推荐阈值': 0.12,
        '首选比分概率': float(score_prob[top_score_columns[0]]),
        '次选比分': ranked_scores[1],
        '次选比分概率': float(score_prob[top_score_columns[1]]),
        '第三比分': ranked_scores[2],
        '大小球进取比分': directional_score,
        '大小球进取比分概率': directional_probability,
        '进取比分依据': (
            '市场最低概率方向＋4球以上＋模型概率不低于0.5%'
            if directional_score else ''
        ),
        **monte_carlo,
        '蒙特风险': '；'.join(monte_risks) or '暂未发现显著冲突',
        '最可能比分Top3': ' / '.join(top_scores),
        '原始最高概率比分': class_to_score(score_classes[raw_top_score_column]),
        '原始最高概率比分概率': float(score_prob[raw_top_score_column]),
        '爆冷比分': upset_score,
        '比分爆冷': upset_score,
        '爆冷比分概率': upset_score_probability,
        '数据采集来源': data_feed_source,
        '数据完整性': '完整' if regular_market_offered else '胜平负赔率缺失/推导',
        '数据来源': 'https://m.sporttery.cn/mjc/jsq/zqspf/',
    }


def run_daily_sporttery(
        output_root: Path = Path('storage/jingcai'),
        headless: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _clear_prediction_model_cache()
    today = date.today().isoformat()
    # Historical review is handled by the five-hour background task and by the
    # dedicated manual-review button.  Running it here made every foreground
    # sync wait through several unavailable network requests before today's
    # fixtures were even loaded.
    raw_path = output_root / 'raw' / f'{today}.json'
    data_feed_source = '官方实时接口'
    try:
        matches = SportteryMobileClient().snapshot(raw_path)
    except RuntimeError as api_error:
        logging.exception('官方移动端接口失败，切换浏览器备用方案。')
        try:
            with SportteryScraper(headless=headless, timeout=12.0) as scraper:
                data_feed_source = '浏览器实时备用源'
                matches = scraper.snapshot(
                    raw_path,
                    include_bonus=lambda row: identify_league(
                        str(_field(row, 'leagueAllName', 'leagueName', 'leagueAbbName'))
                        or _display_fields(row)[1]
                    ) is not None,
                )
        except Exception as scraper_error:
            raise RuntimeError(
                '官方实时接口和浏览器实时抓取均不可用；为避免使用旧盘口，'
                '本次同步已停止，请稍后重试。'
            ) from scraper_error

    # Append odds snapshots so 盘口分析 can track open-to-kickoff drift.
    record_odds_snapshots(matches)
    odds_series = read_odds_series()
    lineup_analysis = fetch_lineup_analysis(matches)

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
        home_model = resolve_model_team(league, [
            raw.get('homeTeamAllName'), raw.get('homeTeamName'),
            raw.get('homeTeamAbbName'), display_home, home_cn,
        ]) if league else None
        away_model = resolve_model_team(league, [
            raw.get('awayTeamAllName'), raw.get('awayTeamName'),
            raw.get('awayTeamAbbName'), display_away, away_cn,
        ]) if league else None
        if display_num and not raw.get('matchNumStr'):
            raw['matchNumStr'] = display_num
        reason = ''
        mapped = (
            league is not None
            and home_model is not None
            and away_model is not None
        )
        odds = latest_had_odds(raw if raw.get('had') else (raw.get('fixedBonus') or {}))
        handicap_odds = latest_hhad_odds(raw)
        regular_market_offered = odds is not None
        if odds is None and handicap_odds is not None:
            odds = _implied_had_from_handicap_market(handicap_odds, raw.get('ttg'))
        elif odds is None:
            odds = _implied_had_without_result_market(raw)

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
                home_model if mapped else home_cn,
                away_model if mapped else away_cn,
                odds, handicap_odds,
                display_league=league or league_name,
                fallback_reason=(
                    '球队尚未映射' if league is not None and not mapped
                    else '未训练联赛'
                ),
                odds_series=odds_series,
                lineup_analysis=lineup_analysis.get(str(raw.get('matchId') or ''), {}),
                regular_market_offered=regular_market_offered,
                data_feed_source=data_feed_source,
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

    prediction_df = pd.DataFrame(predictions)
    skipped_df = _sort_by_match_number(pd.DataFrame(skipped))
    report_dir = output_root / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    # The official selling feed removes a fixture after its sales cutoff.  A
    # refresh must update the current rows without erasing earlier matches from
    # the same daily card (including confirmed lineups needed for review).
    latest_path = report_dir / '最新竞彩预测.csv'
    if latest_path.exists() and latest_path.stat().st_size > 0:
        try:
            previous = pd.read_csv(latest_path)
        except pd.errors.EmptyDataError:
            previous = pd.DataFrame()
        if not previous.empty and '比赛时间' in previous.columns:
            previous_dates = previous['比赛时间'].fillna('').astype(str).str[:10]
            previous = previous.loc[previous_dates.ge(today)].copy()
        if not previous.empty:
            identity = '比赛ID' if '比赛ID' in previous.columns else '赛事编号'
            current_ids = (
                set(prediction_df[identity].fillna('').astype(str))
                if identity in prediction_df.columns else set()
            )
            retained_only = ~previous[identity].fillna('').astype(str).isin(current_ids)
            previous.loc[retained_only, '官方销售状态'] = '已退出当前在售列表'
            previous.loc[retained_only, '同步时段'] = '停止推荐'
            previous.loc[retained_only, '投注时间提示'] = (
                '本场仅保留供复盘；已不在当前官方在售列表，请勿下单'
            )
            previous.loc[retained_only, '数据采集来源'] = '本日早前快照（已退出在售）'
        if not previous.empty:
            prediction_df = pd.concat([previous, prediction_df], ignore_index=True)
            identity = '比赛ID' if '比赛ID' in prediction_df.columns else '赛事编号'
            prediction_df = prediction_df.drop_duplicates(identity, keep='last')
    # Same-day rows retained after the selling cutoff may predate governance
    # columns. Label them honestly instead of leaving blank cells that look
    # like a broken audit trail.
    if not prediction_df.empty:
        if '数据采集来源' not in prediction_df:
            prediction_df['数据采集来源'] = ''
        prediction_df['数据采集来源'] = prediction_df['数据采集来源'].fillna('').replace(
            '', '本日早前快照（来源字段未记录）',
        )
        complete_odds = prediction_df.reindex(columns=(
            '官方胜奖金', '官方平奖金', '官方负奖金',
        )).notna().all(axis=1)
        if '数据完整性' not in prediction_df:
            prediction_df['数据完整性'] = ''
        missing_integrity = prediction_df['数据完整性'].fillna('').eq('')
        prediction_df.loc[missing_integrity, '数据完整性'] = np.where(
            complete_odds[missing_integrity], '完整', '胜平负赔率缺失/推导',
        )
        for column in (
            '胜负模型状态', '大小球模型状态', '比分模型状态', '半全场模型状态',
        ):
            if column not in prediction_df:
                prediction_df[column] = ''
            prediction_df[column] = prediction_df[column].fillna('').replace(
                '', '历史快照未记录治理状态',
            )
        if '模型治理状态' not in prediction_df:
            prediction_df['模型治理状态'] = ''
        missing_governance = prediction_df['模型治理状态'].fillna('').eq('')
        prediction_df.loc[missing_governance, '模型治理状态'] = prediction_df.loc[
            missing_governance,
            ['胜负模型状态', '大小球模型状态', '比分模型状态', '半全场模型状态'],
        ].apply(lambda row: '；'.join(dict.fromkeys(row.astype(str))), axis=1)
    prediction_df = _sort_by_match_number(prediction_df.reset_index(drop=True))
    prediction_df.to_csv(report_dir / f'{today}-竞彩预测.csv', index=False)
    prediction_df.to_csv(latest_path, index=False)
    return prediction_df, skipped_df
