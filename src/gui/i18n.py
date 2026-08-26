from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QWidget,
)


ZH = {
    "File": "文件",
    "Tools": "工具",
    "Analysis": "分析",
    "Models": "模型",
    "Predict": "预测",
    "View": "视图",
    "Help": "帮助",
    "New League": "下载并新建联赛数据",
    "Load League": "打开已有联赛",
    "Delete League": "删除联赛及本地数据",
    "Close League": "关闭当前联赛",
    "Exit": "退出",
    "Data Analysis": "数据分析",
    "Descriptive Analysis": "查看数据概况与统计",
    "Distribution Analysis": "查看各项数据分布",
    "Variance Analysis": "检查低变化特征",
    "Correlation Analysis": "查看特征之间的相关性",
    "Boruta Analysis": "用 Boruta 筛选有效特征",
    "Coefficient Analysis": "查看特征对预测的影响",
    "Impurity Analysis": "查看树模型特征重要性",
    "Rules Extractor": "提取可解释的预测规则",
    "Train": "训练模型",
    "Evaluate": "回测模型表现",
    "Manage": "管理已训练模型",
    "Manage Models": "管理已训练模型",
    "Predict Offline": "手动输入比赛预测",
    "Predict Manual": "手动输入比赛预测",
    "Predict Fixtures": "获取并预测指定日期赛程",
    "Light": "浅色",
    "Dark": "深色",
    "Default": "默认",
    "Update": "检查更新",
    "Submit Bug": "报告问题",
    "Buy me a Coffee!": "赞助作者",
    "Welcome Notification": "欢迎使用",
    "OK": "确定",
    "Ok": "确定",
    "Cancel": "取消",
    "Yes": "是",
    "No": "否",
    "Close": "关闭",
    "Save": "保存",
    "Delete": "删除",
    "Download": "下载",
    "Export": "导出所选预测结果",
    "Find": "查找",
    "Copy": "复制",
    "Search": "搜索",
    "Create": "创建",
    "Open": "打开",
    "Apply": "应用当前设置",
    "Reset": "恢复默认设置",
    "Select All": "全选",
    "All": "全部",
    "New League Dialog": "新建联赛",
    "Load League Dialog": "打开联赛",
    "Delete League Dialog": "删除联赛",
    "Enter a unique league id...": "输入唯一的联赛名称…",
    "League:": "联赛：",
    "League ID:": "联赛名称：",
    "Season:": "赛季：",
    "Date": "日期",
    "Start Year:": "开始年份：",
    "Statistics:": "统计字段：",
    "Home": "主队",
    "Away": "客队",
    "Home Team": "主队",
    "Away Team": "客队",
    "Draw": "平局",
    "Result": "赛果",
    "Target:": "预测目标：",
    "Team:": "球队：",
    "Column:": "字段：",
    "Method:": "方法：",
    "Algorithm": "算法",
    "Model ID:": "模型名称：",
    "Enter a unique model id...": "输入唯一的模型名称…",
    "Features": "特征",
    "Feature:": "特征：",
    "Train Samples": "训练样本",
    "Eval Samples": "评估样本",
    "Cross Validation": "执行交叉验证",
    "Sliding Cross Validation": "执行按时间滚动回测",
    "Hyperparameter Tuning": "超参数调优",
    "Tune": "自动搜索较优参数",
    "Evaluate Models": "回测所选模型",
    "Delete Model": "删除所选模型",
    "Logistic Regression": "逻辑回归",
    "Decision Tree": "决策树",
    "Random Forest": "随机森林",
    "Extreme Boosting (XGBoost)": "极端梯度提升（XGBoost）",
    "K-Nearest Neighbors (KNN)": "K近邻（KNN）",
    "Naive Bayes": "朴素贝叶斯",
    "Support Vector Machine (SVM)": "支持向量机（SVM）",
    "Deep Neural Network (DNN)": "深度神经网络（DNN）",
    "Accuracy": "准确率",
    "Precision": "精确率",
    "Recall": "召回率",
    "Correct": "正确",
    "Classification Metrics": "分类指标",
    "Fixture Date:": "赛程日期：",
    "Fixtures Dialog": "未来赛程预测",
    "Prediction Dialog": "比赛预测",
    "Home Odd...": "主胜赔率…",
    "Away Odd...": "客胜赔率…",
    "Draw Odd...": "平局赔率…",
    "Export to CSV": "导出为 CSV",
    "Export Completed!": "导出完成！",
    "Export Failed": "导出失败",
    "Success": "成功",
    "Error": "错误",
    "Invalid Odds": "赔率无效",
    "Insufficient Data": "数据不足",
    "None Selected": "尚未选择",
    "Hide Missing": "隐藏表格中的缺失值",
    "Auto-Resizing": "自动调整列宽",
    "Entire Table": "整张表格",
    "Copied to clipboard.": "已复制到剪贴板。",
    "Descriptive Statistics Analysis": "描述性统计分析",
    "Descriptions": "查看数据概况与统计",
    "Distributions": "查看各项数据分布",
    "Variances": "检查低变化特征",
    "Correlations": "查看特征之间的相关性",
    "Boruta Selections": "用 Boruta 筛选有效特征",
    "Coefficients": "线性模型系数分析",
    "Rule Extraction": "提取可解释的预测规则",
    "Feature Variance": "特征方差",
    "Feature Correlations": "特征相关性",
    "Interpretability": "模型解释",
    "Class Imbalance": "类别不平衡",
    "Advanced ML Topics": "高级机器学习",
    "Discriminant Analysis (LDA/QDA)": "判别分析（LDA/QDA）",
    "Theme": "界面主题",
    "Open League": "切换联赛",
    "Colormap:": "图表配色：",
    "No Existing Leagues.": "没有已有联赛",
    "There are no existing leagues to delete.": "当前没有可以删除的联赛。",
    "There are no existing leagues to load.": "当前没有可以打开的联赛。",
    "League is Open": "联赛正在使用",
    "Update League": "打开前更新最新比赛数据",
    "League Failed": "联赛创建失败",
    "Odd Difference": "赔率范围无效",
    "League Exists": "联赛名称已存在",
    "vs": "对阵",
    "Apply Clustering": "应用聚类分组",
    "Boundary Plot": "决策边界图",
    "x-Feature:": "横轴特征：",
    "y-Feature:": "纵轴特征：",
    "Waterfall Plot": "单场贡献瀑布图",
    "Match Index:": "比赛序号：",
    "Shap Bar Plot": "SHAP 特征贡献图",
    "No features selected.": "尚未选择任何特征。",
    "No features or target selected.": "尚未选择特征或预测类别。",
    "No target selected.": "尚未选择预测类别。",
    "Model Creation Failed": "模型创建失败",
    "Model Exists": "模型名称已存在",
    "No Tunable Parameters": "没有选择自动调参项目",
    "No tunable parameters have been selected. Proceeding without tuning.": "没有勾选需要自动搜索的参数，将使用当前设置继续训练。",
    "Objective:": "优化指标：",
    "Hyperparameter Plot Error": "调参结果图生成失败",
    "Evaluation Instructions": "模型回测说明",
    "Dataset:": "回测数据范围：",
    "Percentile Filters": "预测概率筛选",
    "Odd Range:": "赔率范围：",
    "There are no existing models to evaluate.": "当前联赛没有可供回测的模型。",
    "There are no existing models to delete.": "当前联赛没有可以删除的模型。",
    "There are no existing models to predict fixtures.": "当前联赛没有可以预测赛程的模型。",
    "Filters:": "筛选条件：",
    "Some matches have been dropped due to insufficient historical data.": "部分比赛因历史数据不足而被跳过。",
    "Select the matches (rows) you want to export.": "请先选中需要导出的比赛行。",
    "Home Missing": "缺少主队",
    "Away Missing": "缺少客队",
    "Same Teams": "主客队不能相同",
    "Please select both home and away teams.": "请选择主队和客队。",
    "Home and away teams must be different.": "主队和客队不能相同。",
    "Importance Bar Plot": "特征重要性条形图",
    "Tree-Rules Plot": "决策树规则图",
    "Tree Depth:": "树的深度：",
    "Model Visualization Plot": "模型结构图",
    "Feature Attention Bar Plot": "特征关注度图",
    "Coefficients Bar Plot": "特征影响系数图",
    "No feature is selected.": "尚未选择特征。",
    "Exact match": "完全匹配",
    "0 matches": "找到 0 项",
    "Find:": "查找：",
    "Slow Operation": "查找可能较慢",
    "Learn": "中文帮助与学习资料",
    "Machine Learning": "机器学习基础",
    "Machine Learning (ML)": "什么是机器学习",
    "ML vs Statistics": "机器学习与传统统计的区别",
    "Supervised Learning": "监督学习入门",
    "Classification": "分类预测入门",
    "ML Models / Algorithms": "预测模型与算法说明",
    "K-Nearest Neighbors": "K近邻算法说明",
    "Linear Discriminant Analysis (LDA)": "线性判别分析（LDA）说明",
    "Support Vector Machines (SVM)": "支持向量机（SVM）说明",
    "Deep Neural Networks (DNN)": "深度神经网络（DNN）说明",
    "Feature Normalization": "为什么需要数据标准化",
    "DNN Improvements": "神经网络优化方法",
    "Model Interpretability / Explainability": "如何解释模型预测",
    "Variable Distribution": "数据分布说明",
    "Regression Coefficients": "回归系数说明",
    "Impurity": "树模型不纯度说明",
    "Boruta Algorithm": "Boruta 特征筛选说明",
    "SMOTE Algorithm": "SMOTE 样本平衡说明",
    "NearMiss Algorithm": "NearMiss 欠采样说明",
    "Partial Dependence Plot": "局部依赖图说明",
    "Select League:": "选择联赛：",
    "ID:": "联赛名称：",
    "League Filters": "联赛筛选",
    "Odd 1:": "主胜赔率：",
    "Odd X:": "平局赔率：",
    "Odd 2:": "客胜赔率：",
    "Match History Window:": "近期场次窗口：",
    "Goal-Difference Margin:": "净胜球阈值：",
    "Trials:": "调优次数：",
    "Normalizer:": "标准化：",
    "Sampler:": "采样方法：",
    "Calibration:": "概率校准：",
    "Cross Validation:": "交叉验证：",
    "Sliding Cross Validation:": "滑动交叉验证：",
    "Predict": "预测",
    "Analyze": "开始分析",
    "Store Filter": "保存当前筛选条件",
    "Delete Filter": "删除已保存筛选条件",
    "Seasonal Metrics": "按赛季查看回测指标",
    "View Metrics": "查看训练与回测指标",
    "Plot Boundaries": "绘制模型决策边界",
    "Plot Partial Dependence": "绘制特征局部依赖图",
    "Plot Waterfall": "绘制单场预测贡献瀑布图",
    "Plot Shap Values": "绘制 SHAP 特征贡献图",
    "Plot Feature Impurity": "绘制树模型特征重要性",
    "Visualize Model": "显示模型结构",
    "Plot Feature Attention": "绘制神经网络特征关注度",
    "Plot Coefficients": "绘制特征影响系数",
    "Manual Prediction Dialog": "手动比赛预测",
    "Prediction": "预测结果",
    "Result (1/X/2)": "胜平负（主胜/平/客胜）",
    "U/O-2.5": "大小球 2.5",
    "Half/Full Result": "半全场（9种结果）",
    "Exact Score": "准确比分（含6+）",
    "Half-time Result": "半场胜平负",
    "Score Prediction": "比分预测",
    "HTR": "半场赛果",
    "No Existing Models.": "没有可用模型",
    "There are no existing models to predict.": "当前联赛没有已训练的模型。",
    "Eval Samples (%):": "评估样本比例（%）：",
    "Model ID:": "模型名称：",
    "Class Weight": "类别权重",
    "Estimators (Trees)": "决策树数量",
    "Criterion": "划分准则",
    "Max Depth:": "最大深度：",
    "Min Samples Leaf": "叶节点最少样本",
    "Min Samples Split": "分裂最少样本",
    "Max Features": "最大特征数",
    "Calibrate Probabilities": "校准预测概率",
    "Standard": "标准化",
    "Min-Max": "最小-最大归一化",
    "Max-Abs": "最大绝对值归一化",
    "None": "无",
    "Logistic Regression Trainer": "逻辑回归模型训练",
    "Decision Tree Trainer": "决策树模型训练",
    "Random Forest Trainer": "随机森林模型训练",
    "XGBoost Trainer": "XGBoost 模型训练",
    "Discriminant Analysis Regression Trainer": "判别分析模型训练",
    "KNN Trainer": "K近邻模型训练",
    "Naive Bayes Trainer": "朴素贝叶斯模型训练",
    "SVM Trainer": "支持向量机模型训练",
    "Deep Neural Network Trainer": "深度神经网络模型训练",
    "Penalty": "正则化方式",
    "Sampling Strategy": "类别不平衡处理",
    "Objective": "调参优化指标",
    "Model Evaluation": "训练前验证设置",
    "Min Samples/Leaf": "叶节点最少样本数",
    "Min Samples/Split": "节点拆分最少样本数",
    "Max Depth": "树的最大深度",
    "Oracle Approximating Shrinkage (OAS)": "OAS 协方差收缩",
    "Decision Boundaries": "分类边界类型",
    "Neighbors": "邻居数量",
    "Neighbor Weights": "邻居权重方式",
    "Distance Metric": "距离计算方式",
    "Kernel": "核函数",
    "Degree": "多项式次数",
    "Gamma": "Gamma 参数",
    "Minimum Child Weight": "子节点最小权重",
    "Learning Rate": "学习率",
    "Lambda Regularization": "Lambda 正则化",
    "Alpha Regularization": "Alpha 正则化",
    "Hidden Layers": "隐藏层数量",
    "Hidden Units": "每层神经元数量",
    "Activation": "激活函数",
    "Variable Selection Network (VSN)": "变量选择网络（VSN）",
    "Layer Normalization": "层归一化",
    "Batch Normalization": "批归一化",
    "Dropout Rate": "随机失活比例",
    "Odd Noise Factor": "赔率噪声系数",
    "Optimizer": "优化器",
    "Lookahead": "Lookahead 优化",
    "Label Smoothing Factor": "标签平滑系数",
    "Batch Size": "每批训练样本数",
    "Epochs": "训练轮数",
    "Early Stopping Epochs": "提前停止等待轮数",
    "Learning Rate Decay": "降低学习率等待轮数",
    "Decay Factor": "学习率衰减系数",
    "Training Instructions": "模型训练说明",
    "Training Results": "模型训练结果",
    "Cross Validation Results": "交叉验证结果",
    "Sliding Cross Validation Results": "按时间滚动回测结果",
}


LEAGUE_ZH = {
    "Premier-League": "英格兰超级联赛",
    "Championship": "英格兰冠军联赛",
    "League-1": "英格兰甲级联赛",
    "League-2": "英格兰乙级联赛",
    "Premiership": "苏格兰超级联赛",
    "Bundesliga-1": "德国甲级联赛",
    "Bundesliga-2": "德国乙级联赛",
    "La-Liga": "西班牙甲级联赛",
    "Segunda-Division": "西班牙乙级联赛",
    "Serie-A": "意大利甲级联赛",
    "Serie-B": "意大利乙级联赛",
    "Ligue-1": "法国甲级联赛",
    "Ligue-2": "法国乙级联赛",
    "Eredivisie": "荷兰甲级联赛",
    "Jupiler-League": "比利时甲级联赛",
    "Primeira-Liga": "葡萄牙超级联赛",
    "Super-Lig": "土耳其超级联赛",
    "Super-League": "希腊超级联赛",
    "Allsvenskan": "瑞典超级联赛",
    "Eliteserien": "挪威超级联赛",
    "Veikkausliiga": "芬兰超级联赛",
    "Ekstraklasa": "波兰甲级联赛",
    "Liga-1": "罗马尼亚甲级联赛",
    "Premier-Division": "爱尔兰超级联赛",
    "MLS": "美国职业足球大联盟",
    "Liga-MX": "墨西哥超级联赛",
    "J-1": "日本甲级联赛",
    "Primera-Division": "阿根廷甲级联赛",
    "Super-Liga": "丹麦超级联赛",
}


TEAM_ZH = {
    "Arsenal": "阿森纳",
    "Aston Villa": "阿斯顿维拉",
    "Birmingham": "伯明翰城",
    "Blackburn": "布莱克本流浪者",
    "Blackpool": "布莱克浦",
    "Bolton": "博尔顿",
    "Bournemouth": "伯恩茅斯",
    "Brentford": "布伦特福德",
    "Brighton": "布莱顿",
    "Burnley": "伯恩利",
    "Cardiff": "卡迪夫城",
    "Charlton": "查尔顿竞技",
    "Chelsea": "切尔西",
    "Coventry": "考文垂",
    "Crystal Palace": "水晶宫",
    "Derby": "德比郡",
    "Everton": "埃弗顿",
    "Fulham": "富勒姆",
    "Huddersfield": "哈德斯菲尔德",
    "Hull": "赫尔城",
    "Ipswich": "伊普斯维奇",
    "Leeds": "利兹联",
    "Leicester": "莱斯特城",
    "Liverpool": "利物浦",
    "Luton": "卢顿",
    "Man City": "曼彻斯特城",
    "Man United": "曼彻斯特联",
    "Middlesbrough": "米德尔斯堡",
    "Newcastle": "纽卡斯尔联",
    "Norwich": "诺维奇城",
    "Nott'm Forest": "诺丁汉森林",
    "Portsmouth": "朴茨茅斯",
    "QPR": "女王公园巡游者",
    "Reading": "雷丁",
    "Sheffield United": "谢菲尔德联",
    "Southampton": "南安普顿",
    "Stoke": "斯托克城",
    "Sunderland": "桑德兰",
    "Swansea": "斯旺西城",
    "Tottenham": "托特纳姆热刺",
    "Watford": "沃特福德",
    "West Brom": "西布朗维奇",
    "West Ham": "西汉姆联",
    "Wigan": "维冈竞技",
    "Wolves": "伍尔弗汉普顿流浪者",
}

TEAM_ZH.update({
    # 德甲
    "Aachen": "亚琛", "Augsburg": "奥格斯堡", "Bayern Munich": "拜仁慕尼黑",
    "Bielefeld": "比勒费尔德", "Bochum": "波鸿", "Braunschweig": "不伦瑞克",
    "Cottbus": "科特布斯", "Darmstadt": "达姆施塔特", "Dortmund": "多特蒙德",
    "Duisburg": "杜伊斯堡", "Ein Frankfurt": "法兰克福", "FC Koln": "科隆",
    "Fortuna Dusseldorf": "杜塞尔多夫", "Freiburg": "弗赖堡", "Greuther Furth": "菲尔特",
    "Hamburg": "汉堡", "Hannover": "汉诺威96", "Hansa Rostock": "汉莎罗斯托克",
    "Heidenheim": "海登海姆", "Hertha": "柏林赫塔", "Hoffenheim": "霍芬海姆",
    "Holstein Kiel": "荷尔斯泰因基尔", "Ingolstadt": "因戈尔施塔特",
    "Kaiserslautern": "凯泽斯劳滕", "Karlsruhe": "卡尔斯鲁厄",
    "Leverkusen": "勒沃库森", "M'gladbach": "门兴格拉德巴赫", "Mainz": "美因茨",
    "Nurnberg": "纽伦堡", "Paderborn": "帕德博恩", "RB Leipzig": "RB莱比锡",
    "Schalke 04": "沙尔克04", "St Pauli": "圣保利", "Stuttgart": "斯图加特",
    "Union Berlin": "柏林联合", "Werder Bremen": "云达不来梅", "Wolfsburg": "沃尔夫斯堡",

    # 意甲
    "Ascoli": "阿斯科利", "Atalanta": "亚特兰大", "Bari": "巴里",
    "Benevento": "贝内文托", "Bologna": "博洛尼亚", "Brescia": "布雷西亚",
    "Cagliari": "卡利亚里", "Carpi": "卡尔皮", "Catania": "卡塔尼亚",
    "Cesena": "切塞纳", "Chievo": "切沃", "Como": "科莫",
    "Cremonese": "克雷莫纳", "Crotone": "克罗托内", "Empoli": "恩波利",
    "Fiorentina": "佛罗伦萨", "Frosinone": "弗罗西诺内", "Genoa": "热那亚",
    "Inter": "国际米兰", "Juventus": "尤文图斯", "Lazio": "拉齐奥",
    "Lecce": "莱切", "Livorno": "利沃诺", "Messina": "墨西拿",
    "Milan": "AC米兰", "Monza": "蒙扎", "Napoli": "那不勒斯",
    "Novara": "诺瓦拉", "Palermo": "巴勒莫", "Parma": "帕尔马",
    "Pescara": "佩斯卡拉", "Pisa": "比萨", "Reggina": "雷吉纳",
    "Roma": "罗马", "Salernitana": "萨勒尼塔纳", "Sampdoria": "桑普多利亚",
    "Sassuolo": "萨索洛", "Siena": "锡耶纳", "Spal": "斯帕尔",
    "Spezia": "斯佩齐亚", "Torino": "都灵", "Treviso": "特雷维索",
    "Udinese": "乌迪内斯", "Venezia": "威尼斯", "Verona": "维罗纳",

    # 法甲
    "Ajaccio": "阿雅克肖", "Ajaccio GFCO": "阿雅克肖加泽莱克", "Amiens": "亚眠",
    "Angers": "昂热", "Arles": "阿尔勒", "Auxerre": "欧塞尔",
    "Bastia": "巴斯蒂亚", "Bordeaux": "波尔多", "Boulogne": "布洛涅",
    "Brest": "布雷斯特", "Caen": "卡昂", "Clermont": "克莱蒙",
    "Dijon": "第戎", "Evian Thonon Gaillard": "伊维恩", "Grenoble": "格勒诺布尔",
    "Guingamp": "甘冈", "Le Havre": "勒阿弗尔", "Le Mans": "勒芒",
    "Lens": "朗斯", "Lille": "里尔", "Lorient": "洛里昂",
    "Lyon": "里昂", "Marseille": "马赛", "Metz": "梅斯",
    "Monaco": "摩纳哥", "Montpellier": "蒙彼利埃", "Nancy": "南锡",
    "Nantes": "南特", "Nice": "尼斯", "Nimes": "尼姆",
    "Paris FC": "巴黎FC", "Paris SG": "巴黎圣日耳曼", "Reims": "兰斯",
    "Rennes": "雷恩", "Sedan": "色当", "Sochaux": "索肖",
    "St Etienne": "圣埃蒂安", "Strasbourg": "斯特拉斯堡", "Toulouse": "图卢兹",
    "Troyes": "特鲁瓦", "Valenciennes": "瓦朗谢讷",

    # 西甲
    "Alaves": "阿拉维斯", "Almeria": "阿尔梅里亚", "Ath Bilbao": "毕尔巴鄂竞技",
    "Ath Madrid": "马德里竞技", "Barcelona": "巴塞罗那", "Betis": "皇家贝蒂斯",
    "Cadiz": "加的斯", "Celta": "塞尔塔", "Cordoba": "科尔多瓦",
    "Eibar": "埃瓦尔", "Elche": "埃尔切", "Espanol": "西班牙人",
    "Getafe": "赫塔菲", "Gimnastic": "塔拉戈纳体操", "Girona": "赫罗纳",
    "Granada": "格拉纳达", "Hercules": "大力神", "Huesca": "韦斯卡",
    "La Coruna": "拉科鲁尼亚", "Las Palmas": "拉斯帕尔马斯", "Leganes": "莱加内斯",
    "Levante": "莱万特", "Malaga": "马拉加", "Mallorca": "马略卡",
    "Murcia": "穆尔西亚", "Numancia": "努曼西亚", "Osasuna": "奥萨苏纳",
    "Oviedo": "皇家奥维耶多", "Real Madrid": "皇家马德里", "Recreativo": "韦尔瓦",
    "Santander": "桑坦德竞技", "Sevilla": "塞维利亚", "Sociedad": "皇家社会",
    "Sp Gijon": "希洪竞技", "Tenerife": "特内里费", "Valencia": "瓦伦西亚",
    "Valladolid": "巴拉多利德", "Vallecano": "巴列卡诺", "Villarreal": "比利亚雷亚尔",
    "Xerez": "赫雷斯", "Zaragoza": "萨拉戈萨",
})


COLUMN_ZH = {
    "Date": "比赛日期",
    "Season": "赛季",
    "Week": "轮次",
    "Home": "主队",
    "Away": "客队",
    "HG": "主队进球",
    "AG": "客队进球",
    "Result": "赛果",
    "HTR": "半场赛果",
    "Result-U/O": "大小球赛果",
    "1": "主胜赔率",
    "X": "平局赔率",
    "2": "客胜赔率",
    "HST": "主队射正",
    "AST": "客队射正",
    "HC": "主队角球",
    "AC": "客队角球",
    "HW": "主队近期胜场",
    "AW": "客队近期胜场",
    "HL": "主队近期负场",
    "AL": "客队近期负场",
    "HGF": "主队近期进球",
    "AGF": "客队近期进球",
    "HAGF": "双方近期进球差",
    "HGA": "主队近期失球",
    "AGA": "客队近期失球",
    "HAGA": "双方近期失球差",
    "HGD": "主队近期净胜球",
    "AGD": "客队近期净胜球",
    "HAGD": "双方近期净胜球差",
    "HWGD": "主队大胜场次",
    "AWGD": "客队大胜场次",
    "HAWGD": "双方大胜场次差",
    "HLGD": "主队大负场次",
    "ALGD": "客队大负场次",
    "HALGD": "双方大负场次差",
    "HW%": "主队胜率",
    "HL%": "主队负率",
    "AW%": "客队胜率",
    "AL%": "客队负率",
    "HSTF": "主队近期射正",
    "ASTF": "客队近期射正",
    "HCF": "主队近期角球",
    "ACF": "客队近期角球",
    "Predicted": "预测结果",
    "Prob(1)": "主胜概率",
    "Prob(X)": "平局概率",
    "Prob(2)": "客胜概率",
    "Prob(U)": "小球概率",
    "Prob(O)": "大球概率",
    "data": "数据集",
    "Model": "模型名称",
    "Model Type": "模型类型",
    "Fold": "第几折",
    "Start Date": "开始日期",
    "End Date": "结束日期",
    "Samples": "样本数",
}


def team_zh(name: str) -> str:
    if name in TEAM_ZH:
        return TEAM_ZH[name]
    from src.services.team_names import chinese_team_name_any
    return chinese_team_name_any(name)


def team_en(name: str) -> str:
    """Resolve an English or Chinese display name to its model-side name."""
    clean = str(name).strip()
    if clean in TEAM_ZH:
        return clean
    folded = clean.casefold()
    for english, chinese in TEAM_ZH.items():
        if folded in {english.casefold(), chinese.casefold()}:
            return english
    return clean


def column_zh(name: str) -> str:
    return COLUMN_ZH.get(name, ZH.get(name, name))


def cell_zh(column: str, value: str) -> str:
    if column in {"Home", "Away"}:
        return team_zh(value)
    if column in {"Result", "HTR"}:
        return {"H": "主胜", "D": "平局", "A": "客胜"}.get(value, value)
    if column in {"Result-U/O", "Predicted"}:
        return {"U": "小球", "O": "大球", "H": "主胜", "D": "平局", "A": "客胜"}.get(value, value)
    if column == "data":
        return {"train": "训练集", "eval": "独立测试集"}.get(value, value)
    return value


PREFIX_ZH = {
    "Thank you for using ProphitBet-v2.": (
        "感谢使用 ProphitBet-v2。\n\n"
        "这是一款开源、非营利应用，预测仅供数据分析参考。"
        "作者不对任何损失负责，请理性使用。\n\n"
        "点击“确定”后将自动进入新建联赛。"
    ),
    "A league already exists with id:": "该联赛名称已经存在：",
    "Failed to create league": "创建联赛失败",
    "Failed to create model": "创建模型失败",
    "Adding league data to table": "正在把联赛数据加入表格",
    "Initializing league": "正在初始化联赛",
    "Deleting league": "正在删除联赛",
    "Deleting model": "正在删除模型",
    "Evaluating matches": "正在评估比赛",
    "Computing Shap Values": "正在计算 SHAP 解释值，这可能需要一些时间",
    "Could not export data": "无法导出数据",
    "Failed to parse fixtures": "获取赛程失败，请确认所选日期正确",
    "Download and create a new league": "从网络下载历史比赛数据，并创建一个新的本地联赛",
    "Load an existing league": "打开已经下载到本机的联赛数据",
    "Delete an existing league from the database": "删除所选联赛及其本地数据；当前已打开的联赛不能删除",
    "Close current league and clear league table": "关闭当前联赛并清空主界面的数据表",
    "Copy selected values to clipboard": "把表格中选中的内容复制到剪贴板",
    "Search team name or a specific value in the table": "在当前表格中查找球队名称或指定数值",
    "Hide all missing values from the table": "隐藏表格中的空值或缺失值",
    "Analyze statistic properties of the features": "查看各字段的数量、均值、标准差和取值范围",
    "Analyze the distribution of the features": "查看各项比赛数据的分布形态",
    "Analyze the variance of each feature": "检查变化过小、可能对预测帮助有限的特征",
    "Analyze the correlation between the features": "查看不同特征之间是否存在较强相关关系",
    "Analyze the importance of each feature using Boruta": "使用 Boruta 方法筛选对预测真正有帮助的特征",
    "Analyze the importance of each feature using Linear Regression": "使用线性模型系数查看各特征对预测方向和强度的影响",
    "Analyze the importance of each feature using Impurity": "使用树模型的不纯度下降查看特征重要性",
    "Extract rules that can explain the outcome of a match": "从模型中提取较容易理解的比赛预测规则",
    "Train ML models to predict the outcome of matches": "使用历史比赛数据训练并保存预测模型",
    "Evaluate models in the selected evaluation matches": "用留出的历史比赛回测模型，查看准确率等指标",
    "Delete models for the current league": "查看或删除当前联赛已经训练好的模型",
    "Analyze & Explain the predictions of models": "分析模型为何得出某个预测结果",
    "Explain/Interpret": "解释所选",
    "Predict the outcome of a match manually": "手动选择主客队和赔率，生成比赛预测",
    "Download and predict an entire fixture": "获取指定日期的整轮赛程，并批量生成预测",
    "Select the model hyperparameters and press": (
        "先填写一个不重复的模型名称，再选择预测目标。初次使用可保留默认参数，"
        "然后点击“开始训练并保存模型”。“自动搜索较优参数”会明显增加训练时间，建议熟悉后再启用。"
    ),
    "Most recent samples": "把最近一部分比赛留作模型回测，不参与训练",
    "Whether to enable/disable hyperparameter tuning": "是否自动尝试多组参数；初次训练建议关闭，可大幅缩短等待时间",
    "Number of hyperparameter tuning iterations": "自动调参尝试次数；次数越多，耗时越长",
    "Tuning objective": "自动调参时要尽量提高的指标",
    "Select feature normalization type": "选择数值特征的标准化方式",
    "Select feature sampler type": "处理不同赛果样本数量不均衡的方法",
    "Whether to calibrate model output probabilities": "让模型输出的胜平负概率更接近真实发生频率",
    "Whether to apply cross validation": "训练前进行多组交叉验证，会增加耗时",
    "Whether to apply sliding cross validation": "按比赛时间顺序进行滚动回测，更贴近真实预测但耗时更长",
    "This might take some time": "正在计算，可能需要几分钟，请不要关闭窗口",
    "It might take some time": "正在自动调参，可能需要较长时间，请不要关闭窗口",
    "A league is currently open.": "当前已有联赛打开。继续后将关闭当前联赛，是否切换？",
    "If checked, the league will be updated before loading": "勾选后，会先联网更新该联赛的最新比赛数据再打开",
    "Number of N previous matches to compute the stats": "用于计算近期状态的历史场次数，通常选择 3 或 4 场",
    "The number of goals that results in early payouts": "用于“大胜/大负”统计的净胜球阈值，通常选择 2 或 3 球",
    "Select the target type, model and the dataset you wish to evaluate": "选择预测目标、模型和回测数据范围；赔率与概率筛选只会改变参与统计的比赛，不会修改模型输出",
    "Plots the attention per feature": "绘制各项特征的关注权重；仅在启用 VSN 时可用",
    "Copy the selected elements": "复制表格中选中的内容",
    "Whether to enable auto-resizing": "是否自动调整表头宽度；大表格中可能降低速度",
    "Export DataFrame to": "把当前表格导出为 CSV；若文件已存在则追加数据",
    "Search a specific value in the table": "在表格或指定字段中查找数值",
    "Searching in the entire table": "对整张表执行模糊查找可能较慢，是否继续？",
}


OPTION_ZH = {
    "None": "无",
    "Yes": "是",
    "No": "否",
    "Standard": "标准化（推荐）",
    "Min-Max": "缩放到 0～1",
    "Max-Abs": "按最大绝对值缩放",
    "SVM-Smote (Oversampling)": "SVM-SMOTE 增加少数类样本",
    "Near-Miss (Undersampling)": "NearMiss 减少多数类样本",
    "Hardness Threshold (Undersampling)": "难度阈值欠采样",
    "Accuracy": "准确率",
    "F1": "F1 综合指标",
    "Precision": "精确率",
    "Recall": "召回率",
    "Gini": "基尼系数",
    "Entropy": "信息熵",
    "Log-Loss": "对数损失",
    "SQRT": "平方根数量",
    "Log2": "以 2 为底的对数数量",
    "Linear": "线性",
    "Quadratic": "二次曲线",
    "Uniform": "所有邻居等权重",
    "Distance": "距离越近权重越高",
    "Manhattan": "曼哈顿距离",
    "Euclidean": "欧氏距离",
    "Gaussian": "高斯朴素贝叶斯",
    "Multinomial": "多项式朴素贝叶斯",
    "Complement": "补集朴素贝叶斯",
    "RBF": "径向基核（推荐）",
    "Polynomial": "多项式核",
    "Sigmoid": "Sigmoid 核",
}


def option_zh(text: str) -> str:
    return OPTION_ZH.get(text, ZH.get(text, text))


def translate(text: str) -> str:
    clean = text.strip()
    if clean in LEAGUE_ZH:
        return text.replace(clean, LEAGUE_ZH[clean])
    if clean in COLUMN_ZH:
        return text.replace(clean, COLUMN_ZH[clean])
    if clean in ZH:
        return text.replace(clean, ZH[clean])
    for english, chinese in PREFIX_ZH.items():
        if clean.startswith(english):
            return chinese + clean[len(english):]
    return text


def translate_widget(widget: QWidget) -> None:
    # Keep native window titles ASCII-only.  WSLg forwards those titles to the
    # Windows shell, where CJK fallback is unreliable on some systems and was
    # rendered as replacement glyphs.  Widget content and menus remain fully
    # translated below.

    tooltip = widget.toolTip()
    translated_tooltip = translate(tooltip)
    if translated_tooltip != tooltip:
        widget.setToolTip(translated_tooltip)

    if isinstance(widget, QAbstractButton):
        text = widget.text()
        translated = translate(text)
        if translated != text:
            widget.setText(translated)
    elif isinstance(widget, QLabel):
        text = widget.text()
        translated = translate(text)
        if translated != text:
            widget.setText(translated)
    elif isinstance(widget, QGroupBox):
        title = widget.title()
        translated = translate(title)
        if translated != title:
            widget.setTitle(translated)
    elif isinstance(widget, QLineEdit):
        placeholder = widget.placeholderText()
        translated = translate(placeholder)
        if translated != placeholder:
            widget.setPlaceholderText(translated)
    elif isinstance(widget, QComboBox):
        for index in range(widget.count()):
            text = widget.itemText(index)
            translated = option_zh(text)
            translated = translate(translated)
            if translated != text:
                widget.setItemText(index, translated)
    for action in widget.findChildren(QAction):
        text = action.text()
        translated = translate(text)
        if translated != text:
            action.setText(translated)
        tooltip = action.toolTip()
        translated_tooltip = translate(tooltip)
        if translated_tooltip != tooltip:
            action.setToolTip(translated_tooltip)
        status_tip = action.statusTip()
        translated_status_tip = translate(status_tip)
        if translated_status_tip != status_tip:
            action.setStatusTip(translated_status_tip)

    for menu in widget.findChildren(QMenu):
        title = menu.title()
        translated = translate(title)
        if translated != title:
            menu.setTitle(translated)


def install_live_translation(app: QApplication) -> QTimer:
    timer = QTimer(app)

    def translate_all() -> None:
        for widget in app.allWidgets():
            translate_widget(widget)

    timer.timeout.connect(translate_all)
    # Newly opened dialogs do not need four full widget-tree scans per second.
    # A one-second interval keeps translations effectively immediate while
    # reducing idle CPU use on WSLg.
    timer.start(1000)
    QTimer.singleShot(0, translate_all)
    return timer
