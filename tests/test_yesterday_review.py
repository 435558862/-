from datetime import date

import pandas as pd

from src.services.yesterday_review import _metric_summary, _patterns, load_yesterday_hit_report


def test_pattern_names_result_direction_unambiguously():
    metric = lambda hit=0, valid=0: {'hit': hit, 'valid': valid}
    rows = [{
        '_result_pick': '胜', '_result_hit': hit,
        '_score_hit_source': '',
        '_metrics': {
            'result': metric(hit, 1), 'handicap': metric(),
            'over_under': metric(), 'half_full': metric(), 'score': metric(),
        },
    } for hit in (1, 1, 0)]
    text = _patterns(rows, _metric_summary(rows))
    assert '胜平负预测为主胜：2/3（66.7%）' in text
    assert '胜方向较好' not in text


def test_yesterday_details_backfill_old_settlement_from_source_report(tmp_path):
    reports = tmp_path / 'reports'
    reports.mkdir()
    settled_path = tmp_path / 'settled.csv'
    pd.DataFrame([
        {
            'prediction_date': '2026-08-23', 'match_id': 101,
            'match_date': '2026-08-24', 'match_number': '周日001',
            'league': '测试联赛', 'home': '主队A', 'away': '客队A',
            'home_goals': 2, 'away_goals': 1,
            'actual_result_label': '胜', 'official_half_score': '1:0',
            'predicted_result': '胜', 'predicted_score': '1-1',
            'predicted_over_under': '大于2.5球',
            'source_report': '2026-08-23-竞彩预测.csv',
            'model_category': '测试专用模型',
        },
        {
            'prediction_date': '2026-08-23', 'match_id': 102,
            'match_date': '2026-08-24 20:00', 'match_number': '周日002',
            'league': '测试联赛', 'home': '主队B', 'away': '客队B',
            'home_goals': 0, 'away_goals': 1,
            'actual_result_label': '负', 'official_half_score': '0:0',
            'predicted_result': '负', 'predicted_score': '0-0',
            'predicted_over_under': '小于2.5球',
            'source_report': '2026-08-23-竞彩预测.csv',
            'model_category': '市场基线',
        },
        {
            'prediction_date': '2026-08-22', 'match_id': 99,
            'match_date': '2026-08-23', 'home_goals': 1, 'away_goals': 1,
        },
    ]).to_csv(settled_path, index=False)
    pd.DataFrame([
        {
            '比赛ID': 101, '比赛时间': '2026-08-24 18:00',
            '官方让球数': -1, '让球首选': '负', '让球次选': '平',
            '半全场首选': '平胜', '半全场次选': '胜胜',
            '首选比分': '1-1', '次选比分': '2-1', '第三比分': '2-0',
            '比分爆冷': '0-1', '大小球进取比分': '3-1',
        },
        {
            '比赛ID': 102, '比赛时间': '2026-08-24 20:00',
            '官方让球数': 1, '让球首选': '平', '让球次选': '胜',
            '半全场首选': '平负', '半全场次选': '负负',
            '首选比分': '0-0', '次选比分': '1-0', '第三比分': '1-1',
            '比分爆冷': '0-1', '大小球进取比分': '1-3',
        },
    ]).to_csv(reports / '2026-08-23-竞彩预测.csv', index=False)

    details, summary = load_yesterday_hit_report(
        today=date(2026, 8, 25),
        settled_path=settled_path,
        report_root=reports,
    )

    assert len(details) == 2
    first = details.loc[details['赛事编号'].eq('周日001')].iloc[0]
    assert first['完场比分'] == '2-1'
    assert '次中' in first['让球（首/次）']
    assert '次中' in first['半全场（首/次）']
    assert first['比分（首/次1/次2/冷/进）'].endswith('2-1（次1中）')
    assert summary['metrics']['result']['hits'] == 2
    assert summary['metrics']['result']['valid'] == 2
    assert summary['metrics']['handicap']['hits'] == 1
    assert summary['metrics']['handicap']['valid'] == 2
    assert summary['metrics']['over_under']['hits'] == 2
    assert summary['metrics']['half_full']['hits'] == 1
    assert summary['metrics']['score']['hits'] == 2
    assert '胜负 2/2 100.0%' in summary['headline']
    assert '其他首选：让球首选 1/2 50.0%；半全场首选 1/2 50.0%' in summary['patterns']
    assert '胜方向较好' not in summary['patterns']


def test_yesterday_uses_actual_match_date_and_handles_year_boundary(tmp_path):
    settled_path = tmp_path / 'settled.csv'
    pd.DataFrame([
        {
            'match_id': 1, 'match_date': '2025-12-31 23:30',
            'match_number': '周三001', 'home_goals': 1, 'away_goals': 0,
            'predicted_result': '胜', 'predicted_score': '1-0',
        },
        {
            'match_id': 2, 'match_date': '2026-01-01 00:30',
            'match_number': '周四001', 'home_goals': 0, 'away_goals': 0,
            'predicted_result': '平', 'predicted_score': '0-0',
        },
    ]).to_csv(settled_path, index=False)

    details, summary = load_yesterday_hit_report(
        today=date(2026, 1, 1), settled_path=settled_path,
        report_root=tmp_path / 'reports',
    )

    assert details['赛事编号'].tolist() == ['周三001']
    assert summary['date'] == '2025-12-31'


def test_yesterday_includes_after_midnight_fixtures_from_daily_card(tmp_path):
    reports = tmp_path / 'reports'
    reports.mkdir()
    settled_path = tmp_path / 'settled.csv'
    pd.DataFrame([{
        'match_id': 101, 'match_date': '2026-08-28 01:00',
        'match_number': '周四001', 'home_goals': 2, 'away_goals': 0,
        'predicted_result': '胜', 'predicted_score': '2-0',
    }]).to_csv(settled_path, index=False)
    pd.DataFrame([{
        '比赛ID': 101, '比赛时间': '2026-08-28 01:00',
        '赛事编号': '周四001',
    }]).to_csv(reports / '2026-08-27-竞彩预测.csv', index=False)

    details, summary = load_yesterday_hit_report(
        today=date(2026, 8, 28), settled_path=settled_path,
        report_root=reports,
    )

    assert details['赛事编号'].tolist() == ['周四001']
    assert summary['date'] == '2026-08-27'
    assert summary['is_fallback'] is False


def test_missing_markets_do_not_count_as_losses(tmp_path):
    settled_path = tmp_path / 'settled.csv'
    pd.DataFrame([{
        'match_id': 1, 'match_date': '2026-08-24',
        'match_number': '周日001', 'home_goals': 1, 'away_goals': 1,
        'actual_result_label': '平', 'predicted_result': '平',
    }]).to_csv(settled_path, index=False)

    details, summary = load_yesterday_hit_report(
        today=date(2026, 8, 25), settled_path=settled_path,
        report_root=tmp_path / 'reports',
    )

    assert len(details) == 1
    assert summary['metrics']['result']['accuracy'] == 1.0
    for key in ('handicap', 'over_under', 'half_full', 'score'):
        assert summary['metrics'][key]['valid'] == 0
        assert summary['metrics'][key]['accuracy'] is None


def test_empty_yesterday_is_explicit_not_blank(tmp_path):
    settled_path = tmp_path / 'settled.csv'
    pd.DataFrame([{
        'match_id': 1, 'match_date': '2026-08-23',
        'home_goals': 1, 'away_goals': 0,
    }]).to_csv(settled_path, index=False)

    details, summary = load_yesterday_hit_report(
        today=date(2026, 8, 25), settled_path=settled_path,
        report_root=tmp_path / 'reports',
    )

    assert len(details) == 1
    assert details.columns.tolist()
    assert summary['date'] == '2026-08-23'
    assert summary['requested_date'] == '2026-08-24'
    assert summary['is_fallback'] is True
    assert '赛果尚未补齐' in summary['headline']
    assert '显示最近已结算日' in summary['headline']


def test_missing_monte_carlo_uses_historical_prior_not_professional_pick(tmp_path):
    settled_path = tmp_path / 'settled.csv'
    pd.DataFrame([{
        'match_id': 1, 'match_date': '2026-08-24',
        'home_goals': 2, 'away_goals': 1, 'official_half_score': '0:0',
        'predicted_half_full': '平胜',
        'predicted_half_full_second': '胜胜',
    }]).to_csv(settled_path, index=False)

    details, _ = load_yesterday_hit_report(
        today=date(2026, 8, 25), settled_path=settled_path,
        report_root=tmp_path / 'reports',
    )

    assert details.loc[0, '半全场（首/次）'].startswith('首平胜/次胜胜')
    assert details.loc[0, '模拟半全场']
    assert '本地跨联赛真实比分先验' in details.loc[0, '模拟模型来源']
    assert '低置信' in details.loc[0, '模拟模型来源']


def test_monte_carlo_hits_use_exact_market_rules(tmp_path):
    settled_path = tmp_path / 'settled.csv'
    pd.DataFrame([{
        'match_id': 1, 'match_date': '2026-08-24',
        'home_goals': 2, 'away_goals': 1, 'official_half_score': '0:0',
        'handicap_line': -1,
        'monte_carlo_top3_score': '2-1 12.0% / 1-1 10.0% / 2-0 9.0%',
        'monte_carlo_result': '胜 55.0%',
        'monte_carlo_handicap': '让平 40.0%',
        'monte_carlo_total': '2-3球 48.0%',
        'monte_carlo_half_full': '平胜 35.0% / 胜胜 25.0%',
    }]).to_csv(settled_path, index=False)

    details, _ = load_yesterday_hit_report(
        today=date(2026, 8, 25), settled_path=settled_path,
        report_root=tmp_path / 'reports',
    )

    for column in ('模拟Top3比分', '模拟胜负', '模拟让球', '模拟总进球', '模拟半全场'):
        assert details.loc[0, column].endswith('（命中）')
    for removed in ('命中项目', '蒙特风险', '模拟数据状态'):
        assert removed not in details.columns
