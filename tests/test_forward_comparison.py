import json
import pandas as pd

from src.services import forward_comparison as audit
from src.gui.windows import sporttery


def test_first_batch_is_immutable_and_past_games_are_excluded(tmp_path, monkeypatch):
    from src.services import baseline_daily_378f3e3 as baseline
    seen = []
    def select(frame, **kwargs):
        seen.append(set(frame['赛事编号']))
        return pd.DataFrame([{'比赛日期': '2099-09-05', '赛事编号': '周六001',
                              '推荐玩法': '胜平负', '重点选项': '★ 胜', '推荐性质': '正式主推'}])
    monkeypatch.setattr(baseline, 'build_daily_recommendations', select)
    monkeypatch.setattr(sporttery, 'build_daily_recommendations', select)
    monkeypatch.setattr(sporttery, '_ticket_card_date', lambda *args: '2099-09-05')
    inputs = pd.DataFrame([
        {'比赛时间': '2099-09-05 20:00', '赛事编号': '周六001', '比赛ID': 1, '官方胜奖金': 2.0},
        {'比赛时间': '2000-01-01 20:00', '赛事编号': '周六002', '比赛ID': 2, '官方胜奖金': 2.0},
    ])
    audit.freeze(inputs, root=tmp_path)
    path = tmp_path / '2099-09-05.json'
    first = path.read_bytes()
    inputs.loc[0, '官方胜奖金'] = 10.0
    audit.freeze(inputs, root=tmp_path)
    assert path.read_bytes() == first
    assert seen == [{'周六001'}, {'周六001'}]


def test_settlement_uses_frozen_price_and_separates_observations(tmp_path):
    pick = dict(version='old', match_id='1', number='周六001', day='2099-09-05',
                market='胜平负', direction='胜', line=None, odds=2.0,
                formal=True, stake=1.0, eligible=True)
    payload = dict(picks=[pick, dict(pick, version='new', formal=False, stake=0.0),
                          dict(pick, match_id='2', number='周六002')])
    (tmp_path / '2099-09-05.json').write_text(json.dumps(payload), encoding='utf-8')
    settled = tmp_path / 'settled.csv'
    pd.DataFrame([{'match_id': 1, 'match_number': '周六001', 'match_date': '2099-09-05 20:00',
                   'home_goals': 2, 'away_goals': 0}]).to_csv(settled, index=False)
    result = audit.report(root=tmp_path, settled_path=settled)
    assert result.iloc[0]['已结算投入'] == 1
    assert result.iloc[0]['ROI'] == 1
    assert result.iloc[0]['待结算'] == 1
    assert result.iloc[1]['观察命中'] == 1
    assert pd.isna(result.iloc[1]['ROI'])


def test_half_ledger_roi_uses_each_settled_stake(tmp_path, monkeypatch):
    monkeypatch.setattr(sporttery, 'HALF_TIME_COMBINATION_ROOT', tmp_path)
    rows = pd.DataFrame([
        {'比赛日期': '2099-09-05', '比赛ID': 1, '赛事编号': '周六001', '目标半场': '胜', '组合赔率': 2.0, '示例本金': 200},
        {'比赛日期': '2099-09-05', '比赛ID': 2, '赛事编号': '周六002', '目标半场': '胜', '组合赔率': 2.0, '示例本金': 100},
        {'比赛日期': '2099-09-05', '比赛ID': 3, '赛事编号': '周六003', '目标半场': '胜', '组合赔率': 2.0, '示例本金': 900},
    ])
    rows.to_csv(tmp_path / '2099-09-05.csv', index=False)
    monkeypatch.setattr(sporttery, '_settled_half_time_indexes', lambda: (
        {'1': {'actual_half_full': '胜胜'}, '2': {'actual_half_full': '平负'}}, {}))
    _, result = sporttery.build_half_time_combination_ledger()
    assert result['stake'] == 1200
    assert result['settled_stake'] == 300
    assert result['profit'] == 100
    assert result['roi'] == 1 / 3
    assert result['pending'] == 1
