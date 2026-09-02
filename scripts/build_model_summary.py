import os

import pandas as pd

from src.database.model import ModelDatabase
from src.preprocessing.utils.target import TargetType


LEAGUES = ['英超', '英冠', '西甲', '德甲', '意甲', '法甲', '葡超', '瑞超', '日职', '韩职']
TARGET_NAMES = {
    TargetType.RESULT: '胜平负',
    TargetType.OVER_UNDER: '大小球',
    TargetType.SCORE: '准确比分',
    TargetType.HALF_RESULT: '半场胜平负',
    TargetType.HALF_FULL: '半全场',
}
ALGORITHM_NAMES = {
    'RandomForest': '随机森林',
    'LogisticRegressor': '标准化逻辑回归',
    'OptimizedEnsemble': '优化集成模型',
    'GoalDistributionModel': '泊松/Dixon-Coles进球分布',
    'ConditionalHalfFullModel': '半全场条件概率链',
    'MarketBlendResultModel': '赔率概率融合',
    'WeightedLogisticModel': '时间衰减逻辑回归',
}


if __name__ == '__main__':
    rows = []
    for league_id in LEAGUES:
        model_db = ModelDatabase(league_id)
        for model_id in model_db.get_model_ids():
            if '早期模型' in model_id:
                continue
            config = model_db.load_model_config(model_id)
            target_type = config['target_type']
            tuning = config.get('train', {}).get('tuning', {})
            algorithm = tuning.get(
                'algorithm',
                ALGORITHM_NAMES.get(config['cls'].__name__, config['cls'].__name__),
            )
            test_accuracy = tuning.get('test_accuracy')
            baseline = tuning.get('majority_baseline')
            p_value = tuning.get('mcnemar_p_value_vs_baseline')
            validated = (
                test_accuracy is not None
                and baseline is not None
                and test_accuracy > baseline
                and p_value is not None
                and p_value < 0.05
            )
            rows.append({
                '联赛': league_id,
                '预测类型': (
                    '让球胜负' if '让球胜负' in model_id else TARGET_NAMES[target_type]
                ),
                '模型': model_id,
                '保留算法': algorithm,
                '最近独立测试首选命中率': test_accuracy,
                '简单基线': baseline,
                '相对基线状态': '有稳定优势' if validated else '尚无稳定优势',
                '相对基线P值': p_value,
                'Top3命中率': tuning.get('top3_accuracy', tuning.get('top2_accuracy')),
                'Top5命中率': tuning.get('top5_accuracy'),
                '测试LogLoss': tuning.get('test_log_loss'),
                '高置信度门槛': tuning.get('selective_threshold'),
                '高置信度命中率': tuning.get('selective_accuracy'),
                '高置信度覆盖率': tuning.get('coverage'),
                '高置信度样本数': tuning.get('selective_samples'),
                '高置信度验证达标': tuning.get('selective_validated'),
                '测试方法': tuning.get('method'),
            })

    report = pd.DataFrame(rows).sort_values(['联赛', '预测类型'])
    os.makedirs('storage/reports', exist_ok=True)
    report.to_csv('storage/reports/最终模型表现汇总.csv', index=False)
    print(report.to_string(index=False))
