"""Chinese display names for internally English-normalized football teams."""

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd


ALIASES_PATH = Path('storage/network/sporttery_team_aliases.json')

# Historical/relegated clubs do not normally appear in today's Sporttery alias
# file, but still need readable names in the league history table.
DISPLAY_OVERRIDES = {
    '葡超': {
        'Academica': '科英布拉大学', 'Aves': '阿维什', 'Beira Mar': '贝拉马尔',
        'Belenenses': '贝伦人', 'Boavista': '博阿维斯塔', 'Chaves': '沙维什',
        'Est Amadora': '阿马多拉之星', 'Farense': '法鲁人', 'Feirense': '费伦斯',
        'Feirense ': '费伦斯', 'Leiria': '莱里亚', 'Leixoes': '雷克索斯',
        'Maritimo': '马里迪莫', 'Naval': '纳瓦尔', 'Olhanense': '欧汉尼斯',
        'Pacos Ferreira': '费雷拉', 'Penafiel': '佩纳菲尔',
        'Portimonense': '波尔蒂芒人', 'Setubal': '塞图巴尔',
        'Trofense': '泰罗芬斯', 'Uniao Madeira': '马德拉联', 'Vizela': '维泽拉',
        'AVS': '阿维什镇', 'Academico Viseu': '维塞乌',
    },
    '西甲': {'Dep. A Coruna': '拉科鲁尼亚'},
    '瑞超': {
        'AFC Eskilstuna': '艾斯基斯杜拿', 'Atvidabergs': '阿特维达堡',
        'Brage': '布莱格', 'Dalkurd': '达尔库德', 'Falkenbergs': '法尔肯堡',
        'Gefle': '耶夫勒', 'Helsingborg': '赫尔辛堡', 'Jonkopings': '延雪平南区',
        'Landskrona': '兰斯科罗纳', 'Ljungskile': '永斯基尔',
        'Norrkoping': '北雪平', 'Orebro': '厄勒布鲁', 'Oster': '厄斯特',
        'Osters': '厄斯特', 'Ostersunds': '厄斯特松德', 'Sundsvall': '松兹瓦尔',
        'Syrianska': '西里安斯卡', 'Trelleborgs': '特雷勒堡',
        'Varberg': '瓦尔贝里', 'Varnamo': '瓦纳默',
    },
    '日职': {
        'Hokkaido Consadole Sapporo': '札幌冈萨多', 'Iwata': '磐田喜悦',
        'Kofu': '甲府风林', 'Kumamoto': '熊本深红', 'Montedio Yamagata': '山形山神',
        'Oita Trinita': '大分三神', 'Omiya Ardija': '大宫松鼠',
        'Sagan Tosu': '鸟栖砂岩', 'Tokushima': '德岛漩涡',
        'V-Varen Nagasaki': '长崎航海', 'Vegalta Sendai': '仙台七夕',
        'Yamaga': '松本山雅', 'Chiba': '千叶市原', 'Mito': '水户蜀葵',
    },
    '韩职': {
        'Daegu': '大邱FC', 'Suwon City': '水原FC',
        'Asan': '忠南牙山', 'Busan': '釜山IPark', 'Gyeongnam': '庆南FC',
        'Jeonnam': '全南天龙', 'Seongnam': '城南FC',
        'Seoul E.': '首尔衣恋', 'Suwon Bluewings': '水原三星蓝翼',
    },
}


@lru_cache(maxsize=1)
def _english_to_chinese() -> dict[str, dict[str, str]]:
    aliases = json.loads(ALIASES_PATH.read_text(encoding='utf-8'))
    result: dict[str, dict[str, str]] = {}
    for league, mapping in aliases.items():
        inverse: dict[str, str] = {}
        # JSON insertion order makes the first spelling the preferred display
        # name while still accepting all alternate Chinese names on input.
        for chinese, english in mapping.items():
            inverse.setdefault(english, chinese)
        result[league] = inverse
    for league, mapping in DISPLAY_OVERRIDES.items():
        result.setdefault(league, {}).update(mapping)
    return result


def chinese_team_name(league_id: str, team: str) -> str:
    """Return a Chinese display name, falling back safely for unknown clubs."""
    return _english_to_chinese().get(league_id, {}).get(str(team), str(team))


def chinese_team_name_any(team: str) -> str:
    """Translate when a dialog does not carry an explicit league id."""
    value = str(team)
    for mapping in _english_to_chinese().values():
        if value in mapping:
            return mapping[value]
    return value


def translate_fixture_columns(df: pd.DataFrame, league_id: str) -> pd.DataFrame:
    """Translate a copy for UI display without changing model input data."""
    shown = df.copy()
    for column in ('Home', 'Away'):
        if column in shown.columns:
            shown[column] = shown[column].map(
                lambda team: chinese_team_name(league_id, team),
            )
    return shown
