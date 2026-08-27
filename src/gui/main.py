import pandas as pd
import qdarktheme
import sys
import webbrowser
from typing import Optional
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow, QMenu, QMessageBox
from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.gui.windows import analysis
from src.gui.windows import leagues
from src.gui.windows import models
from src.gui.windows.sporttery import SportteryPredictionsDialog
from src.gui.widgets.tables import DataFrameTable
from src.gui.utils.taskrunner import ThreadWorker
from src.network.leagues.league import League
from src.services.league_sync import sync_five_leagues, sync_is_due
from src.services.team_names import translate_fixture_columns
from src.version import PRODUCT_TITLE


LEAGUE_WINDOW_TITLES = {
    '英超': 'Premier League',
    '西甲': 'La Liga',
    '德甲': 'Bundesliga',
    '意甲': 'Serie A',
    '法甲': 'Ligue 1',
    '瑞超': 'Allsvenskan',
    '葡超': 'Primeira Liga',
    '日职': 'J1 League',
    '韩职': 'K League 1',
}


class MainWindow(QMainWindow):
    """ The main app window. It includes the menu with all functionalities and the league table. """

    def __init__(self, app: QApplication):
        super().__init__()

        self._app = app

        self._title = PRODUCT_TITLE
        self._width = 800
        self._height = 600

        # Declaring functional placeholders.
        self._league_df = None
        self._league = None
        self._model_db = None

        self._league_db = LeagueDatabase()
        self._current_theme = 'Default'

        # Declaring widget placeholders.
        self._action_new = None
        self._action_load = None
        self._menu_analysis = None
        self._menu_models = None
        self._menu_predictions = None
        self._table = None
        self._boruta_analyzer = None
        self._coefficients_analyzer = None
        self._correlation_analyzer = None
        self._description_analyzer = None
        self._distributions_analyzer = None
        self._impurity_analyzer = None
        self._rules_analyzer = None
        self._variance_analyzer = None
        self._sync_thread = None
        self._sync_worker = None
        self._sync_manual = False
        self._fixtures_dialog = None
        self._sporttery_dialog = None

        # Declaring QDark and QT themes.
        self._qdark_theme_mapper = {'Default': 'auto', 'Light': 'light', 'Dark': 'dark'}

        # Initializing main window.
        self._initialize_window()
        self._add_widgets()

        QTimer.singleShot(200, self._show_welcome_message)

    def _initialize_window(self):
        """ Initializes main window. """

        self.setWindowTitle(self._title)
        self.resize(self._width, self._height)
        self.statusBar()

    def _exec_owned_dialog(self, dialog: QDialog):
        """Run a modal dialog as an owned WSLg window and bring it forward."""
        flags = dialog.windowFlags()
        if sys.platform == 'darwin':
            # A parented WindowModal QDialog is presented like a macOS sheet and
            # may lose its native traffic-light controls.  Keep it owned by the
            # main window, but make it a regular top-level window with a native
            # title bar so users can close, minimize, or zoom it normally.
            flags |= (
                Qt.WindowType.Window
                | Qt.WindowType.WindowTitleHint
                | Qt.WindowType.WindowMinimizeButtonHint
                | Qt.WindowType.WindowMaximizeButtonHint
                | Qt.WindowType.WindowCloseButtonHint
            )
        dialog.setParent(self, flags)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        QTimer.singleShot(0, dialog.raise_)
        QTimer.singleShot(0, dialog.activateWindow)
        return dialog.exec()

    def _add_widgets(self):
        """ Adds menubar and match table. """

        # --- Menu Bar ---
        menubar = self.menuBar()
        # Keep menus inside the app window on macOS so they are always visible.
        menubar.setNativeMenuBar(False)

        # --- File Menu ---
        menu_file = menubar.addMenu('File')
        self._add_file_menus(menu_file=menu_file)

        # --- Tools Menu ---
        menu_tools = menubar.addMenu('Tools')
        self._add_tool_menus(menu_tools=menu_tools)

        # --- Analysis Menu ---
        self._menu_analysis = menubar.addMenu('Analysis')
        self._add_analysis_menus(menu_analysis=self._menu_analysis)

        # --- Train Menu ---
        self._menu_models = menubar.addMenu('Models')
        self._add_model_menus(menu_model=self._menu_models)

        # --- Predict Menu ---
        self._menu_predictions = menubar.addMenu('Predict')
        self._add_predict_menus(menu_predictions=self._menu_predictions)

        # --- View Menu ---
        menu_view = menubar.addMenu('View')
        self._add_view_menus(menu_view=menu_view)

        # --- Help Menu ---
        menu_help = menubar.addMenu('Help')
        self._add_help_menus(menu_help=menu_help)

        self._set_league_menus_state(enabled=False)

    def _add_file_menus(self, menu_file: QMenu):
        """ Adds all league-related menus. """

        def close_league_permission() -> bool:
            """ Requests permission to close the current league. Returns whether use has agreed or not. """

            if self._league_df is None:
                return True

            result = QMessageBox.warning(
                self,
                'Open League',
                'A league is currently open. If you proceed, the current league will close. Do you want to proceed?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes
            return result

        def exit_app():
            """ Exists the main app.  """

            if not close_league_permission():
                return

            self._app.quit()

        def new_league():
            """ Opens a new league dialog. """

            if not close_league_permission():
                return

            new_league_dialog = leagues.NewLeagueWindow(league_db=self._league_db)
            self._exec_owned_dialog(new_league_dialog)
            self._load_new_league(df=new_league_dialog.league_df, league=new_league_dialog.league)

        def load_league():
            """ Loads a load league dialog. """

            if not close_league_permission():
                return

            current_league_id = None if self._league is None else self._league.league_id
            load_league_dialog = leagues.LoadLeagueWindow(league_db=self._league_db, current_league_id=current_league_id)
            self._exec_owned_dialog(load_league_dialog)
            self._load_new_league(df=load_league_dialog.league_df, league=load_league_dialog.league)

        def delete_league():
            """ Deletes an existing league. Cannot delete current league. """

            current_league_id = None if self._league is None else self._league.league_id
            self._exec_owned_dialog(
                leagues.DeleteLeagueWindow(
                    league_db=self._league_db,
                    current_league_id=current_league_id,
                )
            )

            if len(self._league_db.get_league_ids()) == 0:
                self._action_load.setEnabled(False)

        def close_league():
            self._clear_table()
            self._league_df = self._league = self._model_db = None

            self.setWindowTitle(self._title)
            self._set_league_menus_state(enabled=False)

        action_new = QAction('New League', self)
        self._action_new = action_new
        action_new.setStatusTip('Download and create a new league')
        action_new.setShortcut(QKeySequence('Ctrl+n'))
        action_new.triggered.connect(new_league)

        self._action_load = QAction('Load League', self)
        self._action_load.setStatusTip('Load an existing league')
        self._action_load.setShortcut(QKeySequence('Ctrl+o'))

        if len(self._league_db.get_league_ids()) == 0:
            self._action_load.setEnabled(False)

        self._action_load.triggered.connect(load_league)

        action_delete = QAction('Delete League', self)
        action_delete.setStatusTip('Delete an existing league from the database')
        action_delete.setShortcut(QKeySequence('Ctrl+d'))
        action_delete.triggered.connect(delete_league)

        self._action_close = QAction('Close League', self)
        self._action_close.setEnabled(False)
        self._action_close.setStatusTip('Close current league and clear league table')
        self._action_close.triggered.connect(close_league)

        action_exit = QAction('Exit', self)
        action_exit.triggered.connect(exit_app)

        menu_file.addAction(action_new)
        menu_file.addAction(self._action_load)
        menu_file.addAction(action_delete)
        menu_file.addSeparator()
        menu_file.addAction(self._action_close)
        menu_file.addSeparator()
        menu_file.addAction(action_exit)

    def _add_tool_menus(self, menu_tools: QMenu):
        """ Adds table functions menus. """

        def copy():
            if self._table is not None:
                self._table.copy_selection()

        def find():
            if self._table is not None:
                self._table.open_find_dialog()

        def hide(checked):
            self._table.hide_missing(hide=checked)

        action_copy = QAction('Copy', self)
        action_copy.setStatusTip('Copy selected values to clipboard.')
        action_copy.setShortcut(QKeySequence('Ctrl+c'))
        action_copy.triggered.connect(copy)
        menu_tools.addAction(action_copy)

        action_find = QAction('Find', self)
        action_find.setStatusTip('Search team name or a specific value in the table')
        action_find.setShortcut(QKeySequence('Ctrl+f'))
        action_find.triggered.connect(find)
        menu_tools.addAction(action_find)

        action_hide = QAction('Hide Missing', self)
        action_hide.setCheckable(True)
        action_hide.setStatusTip('Hide all missing values from the table.')
        action_hide.setShortcut(QKeySequence('Ctrl+h'))
        action_hide.triggered.connect(hide)
        menu_tools.addSeparator()
        menu_tools.addAction(action_hide)

        action_sync = QAction('同步全部联赛新数据', self)
        action_sync.setStatusTip('同步五大联赛、瑞超、葡超、日职和韩职，并增量重训已有模型')
        action_sync.triggered.connect(lambda _: self._start_league_sync(manual=True))
        menu_tools.addSeparator()
        menu_tools.addAction(action_sync)

    def _add_analysis_menus(self, menu_analysis: QMenu):
        """ Adds analysis tools options. """

        def open_boruta():
            self._boruta_analyzer = analysis.BorutaAnalyzerWindow(df=self._league_df)
            self._boruta_analyzer.show()

        def open_distributions():
            self._distributions_analyzer = analysis.DistributionAnalyzerWindow(df=self._league_df)
            self._distributions_analyzer.show()

        def open_coefficients():
            self._coefficients_analyzer = analysis.CoefficientsAnalyzerWindow(df=self._league_df)
            self._coefficients_analyzer.show()

        def open_correlations():
            self._correlation_analyzer = analysis.CorrelationsAnalyzerWindow(df=self._league_df)
            self._correlation_analyzer.show()

        def open_descriptive():
            self._description_analyzer = analysis.DescriptiveAnalyzerWindow(df=self._league_df)
            self._description_analyzer.show()

        def open_variances():
            self._variance_analyzer = analysis.VarianceAnalyzerWindow(df=self._league_df)
            self._variance_analyzer.show()

        def open_impurity():
            self._impurity_analyzer = analysis.ImpurityAnalyzerWindow(df=self._league_df)
            self._impurity_analyzer.show()

        def open_rules():
            self._rules_analyzer = analysis.RulesAnalyzerWindow(df=self._league_df)
            self._rules_analyzer.show()

        action_descriptions = QAction('Descriptions', self)
        action_descriptions.setStatusTip('Analyze statistic properties of the features')
        action_descriptions.triggered.connect(open_descriptive)

        action_distributions = QAction('Distributions', self)
        action_distributions.setStatusTip('Analyze the distribution of the features')
        action_distributions.triggered.connect(open_distributions)

        action_variances = QAction('Variances', self)
        action_variances.setStatusTip('Analyze the variance of each feature')
        action_variances.triggered.connect(open_variances)

        action_correlation = QAction('Correlations', self)
        action_correlation.setStatusTip('Analyze the correlation between the features')
        action_correlation.triggered.connect(open_correlations)

        action_boruta = QAction('Boruta Selections', self)
        action_boruta.setStatusTip('Analyze the importance of each feature using Boruta')
        action_boruta.triggered.connect(open_boruta)

        action_coefficients = QAction('Coefficients', self)
        action_coefficients.setStatusTip('Analyze the importance of each feature using Linear Regression')
        action_coefficients.triggered.connect(open_coefficients)

        action_impurity = QAction('Impurity Analysis', self)
        action_impurity.setStatusTip('Analyze the importance of each feature using Impurity')
        action_impurity.triggered.connect(open_impurity)

        action_rules = QAction('Rule Extraction', self)
        action_rules.setStatusTip('Extract rules that can explain the outcome of a match')
        action_rules.triggered.connect(open_rules)

        menu_analysis.addAction(action_descriptions)
        menu_analysis.addAction(action_distributions)
        menu_analysis.addAction(action_variances)
        menu_analysis.addAction(action_correlation)
        menu_analysis.addSeparator()
        menu_analysis.addAction(action_boruta)
        menu_analysis.addAction(action_coefficients)
        menu_analysis.addAction(action_impurity)
        menu_analysis.addSeparator()
        menu_analysis.addAction(action_rules)

    def _add_model_menus(self, menu_model: QMenu):
        """ Adds model train, eval and analysis menus. """

        # Creating train menus.
        menu_train = QMenu('Train', self)
        menu_train.setStatusTip('Train ML models to predict the outcome of matches')

        action_train_logistic = QAction('Logistic Regression', self)
        action_train_logistic.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.LogisticRegressionTrainerDialog(df=self._league_df, model_db=self._model_db))
        )
        action_train_discriminant = QAction('Discriminant Analysis (LDA/QDA)', self)
        action_train_discriminant.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.DiscriminantTrainerDialog(df=self._league_df, model_db=self._model_db))
        )
        action_train_tree = QAction('Decision Tree', self)
        action_train_tree.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.DecisionTreeTrainerDialog(df=self._league_df, model_db=self._model_db))
        )
        action_train_forest = QAction('Random Forest', self)
        action_train_forest.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.RandomForestTrainerDialog(df=self._league_df, model_db=self._model_db))
        )
        action_train_xgboost = QAction('Extreme Boosting (XGBoost)', self)
        action_train_xgboost.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.ExtremeBoostingTrainerDialog(df=self._league_df, model_db=self._model_db))
        )
        action_train_knn = QAction('K-Nearest Neighbors (KNN)', self)
        action_train_knn.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.KNNTrainerDialog(df=self._league_df, model_db=self._model_db))
        )
        action_train_nb = QAction('Naive Bayes', self)
        action_train_nb.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.NaiveBayesTrainerDialog(df=self._league_df, model_db=self._model_db))
        )
        action_train_svm = QAction('Support Vector Machine (SVM)', self)
        action_train_svm.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.SVMTrainerDialog(df=self._league_df, model_db=self._model_db))
        )
        action_train_dnn = QAction('Deep Neural Network (DNN)', self)
        action_train_dnn.triggered.connect(
            lambda _: self._exec_owned_dialog(models.trainers.NeuralNetworkTrainerDialog(df=self._league_df, model_db=self._model_db))
        )

        menu_train.addAction(action_train_logistic)
        menu_train.addAction(action_train_discriminant)
        menu_train.addAction(action_train_tree)
        menu_train.addAction(action_train_forest)
        menu_train.addAction(action_train_xgboost)
        menu_train.addAction(action_train_knn)
        menu_train.addAction(action_train_nb)
        menu_train.addAction(action_train_svm)
        menu_train.addAction(action_train_dnn)

        # Creating evaluation menus.
        action_eval = QAction('Evaluate', self)
        action_eval.setStatusTip('Evaluate models in the selected evaluation matches')
        action_eval.triggered.connect(lambda _: self._exec_owned_dialog(models.EvaluatorDialog(df=self._league_df, model_db=self._model_db)))

        # Creating evaluation menus.
        action_model_manager = QAction('Manage Models', self)
        action_model_manager.setStatusTip('Delete models for the current league.')
        action_model_manager.triggered.connect(lambda _: self._exec_owned_dialog(models.ModelManagerDialogWindow(model_db=self._model_db)))

        # Creating analysis menus.
        menu_interpret = QMenu('Interpretability', self)
        menu_interpret.setStatusTip('Analyze & Explain the predictions of models.')

        action_analysis_logistic = QAction('Logistic Regression', self)
        action_analysis_logistic.setStatusTip('Explain/Interpret Logistic Regression models.')
        action_analysis_logistic.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.LogisticExplainerDialog(df=self._league_df, model_db=self._model_db)))

        action_analysis_lda = QAction('Discriminant Analysis (LDA/QDA)', self)
        action_analysis_lda.setStatusTip('Explain/Interpret Discriminant models.')
        action_analysis_lda.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.DiscriminantExplainerDialog(df=self._league_df, model_db=self._model_db)))

        action_analysis_tree = QAction('Decision Tree', self)
        action_analysis_tree.setStatusTip('Explain/Interpret Decision Tree models.')
        action_analysis_tree.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.DecisionTreeExplainerDialog(df=self._league_df, model_db=self._model_db)))

        action_analysis_forest = QAction('Random Forest', self)
        action_analysis_forest.setStatusTip('Explain/Interpret Random Forest models.')
        action_analysis_forest.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.RandomForestExplainerDialog(df=self._league_df, model_db=self._model_db)))

        action_analysis_xgboost = QAction('Extreme Boosting (XGBoost)', self)
        action_analysis_xgboost.setStatusTip('Explain/Interpret XGBoost models.')
        action_analysis_xgboost.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.ExtremeBoostingTreeExplainerDialog(df=self._league_df, model_db=self._model_db)))

        action_analysis_knn = QAction('K-Nearest Neighbors (KNN)', self)
        action_analysis_knn.setStatusTip('Explain/Interpret KNN models.')
        action_analysis_knn.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.KNNExplainerDialog(df=self._league_df, model_db=self._model_db)))

        action_analysis_nb = QAction('Naive Bayes', self)
        action_analysis_nb.setStatusTip('Explain/Interpret Naive Bayes models.')
        action_analysis_nb.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.NaiveBayesExplainerDialog(df=self._league_df, model_db=self._model_db)))

        action_analysis_svm = QAction('Support Vector Machine (SVM)', self)
        action_analysis_svm.setStatusTip('Explain/Interpret SVM Regression models.')
        action_analysis_svm.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.SVMExplainerDialog(df=self._league_df, model_db=self._model_db)))

        action_analysis_dnn = QAction('Deep Neural Network (DNN)', self)
        action_analysis_dnn.setStatusTip('Explain/Interpret Neural Networks.')
        action_analysis_dnn.triggered.connect(lambda _: self._exec_owned_dialog(models.explainers.NeuralNetworkExplainerDialog(df=self._league_df, model_db=self._model_db)))

        menu_interpret.addAction(action_analysis_logistic)
        menu_interpret.addAction(action_analysis_lda)
        menu_interpret.addAction(action_analysis_tree)
        menu_interpret.addAction(action_analysis_forest)
        menu_interpret.addAction(action_analysis_xgboost)
        menu_interpret.addAction(action_analysis_knn)
        menu_interpret.addAction(action_analysis_nb)
        menu_interpret.addAction(action_analysis_svm)
        menu_interpret.addAction(action_analysis_dnn)

        menu_model.addMenu(menu_train)
        menu_model.addAction(action_eval)
        menu_model.addAction(action_model_manager)
        menu_model.addSeparator()
        menu_model.addMenu(menu_interpret)

    def _add_view_menus(self, menu_view: QMenu):
        """ Adds view menus. """

        def apply_qdark_theme(theme_name: str):
            """ Applies a QDark theme to the app windows and widgets. """

            # Abort operation of the same theme is selected.
            if theme_name == self._current_theme:
                return

            self._current_theme = theme_name

            theme_mode = self._qdark_theme_mapper[theme_name]
            qdarktheme.setup_theme(theme_mode)

            self.statusBar().showMessage(f'Theme: {theme_name}')

        # Creating theme menus.
        menu_theme = QMenu('Theme', self)
        menu_view.addMenu(menu_theme)

        group = QActionGroup(self)
        group.setExclusive(True)
        for name in self._qdark_theme_mapper.keys():
            action = QAction(name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, n=name: apply_qdark_theme(theme_name=n))
            group.addAction(action)
            menu_theme.addAction(action)

    def _add_predict_menus(self, menu_predictions: QMenu):
        """ Adds prediction menu options. """

        action_offline = QAction('Predict Manual', self)
        action_offline.setStatusTip('Predict the outcome of a match manually')
        action_offline.triggered.connect(
            lambda _: self._exec_owned_dialog(models.PredictorDialog(
                df=self._league_df,
                model_db=self._model_db,
                league_db=self._league_db,
            ))
        )

        action_fixtures = QAction('Predict Fixtures', self)
        action_fixtures.setStatusTip('Download and predict an entire fixture')
        action_fixtures.triggered.connect(self._open_fixture_predictions)

        action_sporttery = QAction('今日竞彩预测', self)
        action_sporttery.setStatusTip('同步官方竞彩场次并查看胜平负、让球、比分和爆冷比分')
        action_sporttery.triggered.connect(self._open_sporttery_predictions)

        menu_predictions.addAction(action_offline)
        menu_predictions.addAction(action_fixtures)
        menu_predictions.addSeparator()
        menu_predictions.addAction(action_sporttery)

    def _open_fixture_predictions(self):
        """Open the fixture dialog without a nested modal event loop.

        Parentless ``exec()`` dialogs can open behind the WSLg main window,
        leaving the visible main window disabled and looking frozen.  Keeping
        the dialog owned by the main window also prevents premature cleanup.
        """
        if self._fixtures_dialog is not None and self._fixtures_dialog.isVisible():
            self._fixtures_dialog.raise_()
            self._fixtures_dialog.activateWindow()
            return

        self._fixtures_dialog = models.FixturesDialog(
            df=self._league_df,
            model_db=self._model_db,
            league=self._league,
            parent=self,
        )
        self._fixtures_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._fixtures_dialog.show()
        self._fixtures_dialog.raise_()
        self._fixtures_dialog.activateWindow()

    def _open_sporttery_predictions(self):
        self._sporttery_dialog = SportteryPredictionsDialog(self)
        self._exec_owned_dialog(self._sporttery_dialog)

    def _add_help_menus(self, menu_help: QMenu):
        """ Adds help/tutorial and donation menus. """

        def open_url(url: str):
            webbrowser.open(url=url)

        def show_chinese_training_guide():
            QMessageBox.information(
                self,
                '中文模型训练快速指南',
                '推荐第一次这样设置：\n\n'
                '1. 打开“模型 → 训练并保存模型 → 逻辑回归”。\n'
                '2. 把“模型名称”改成一个新名称，例如：胜平负模型2。\n'
                '3. 预测目标选择“胜平负（主胜/平/客胜）”。\n'
                '4. 评估样本比例保留 20%。\n'
                '5. 关闭“自动搜索较优参数”、交叉验证和按时间滚动回测，可快速完成首次训练。\n'
                '6. 其余选项保持默认，点击“开始训练并保存模型”。\n\n'
                '训练完成后，到“预测 → 手动输入比赛预测”选择刚保存的模型。'
            )

        action_chinese_guide = QAction('中文模型训练快速指南', self)
        action_chinese_guide.triggered.connect(show_chinese_training_guide)
        menu_help.addAction(action_chinese_guide)
        menu_help.addSeparator()

        # Creating tutorial menu.
        lessons_menu = QMenu('Learn', self)

        # Adding tutorial menus.
        menu_ml = QMenu('Machine Learning', self)
        action_ml = QAction('Machine Learning (ML)', self)
        action_ml.triggered.connect(lambda _: open_url(url='https://www.ibm.com/think/topics/machine-learning'))

        action_ml_stats = QAction('ML vs Statistics', self)
        action_ml_stats.triggered.connect(
            lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/difference-between-statistical-model-and-machine-learning/')
        )

        action_supervised = QAction('Supervised Learning', self)
        action_supervised.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/'))

        action_classification = QAction('Classification', self)
        action_classification.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/getting-started-with-classification/'))

        action_metrics = QAction('Classification Metrics', self)
        action_metrics.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/'))

        menu_ml.addAction(action_ml)
        menu_ml.addAction(action_ml_stats)
        menu_ml.addAction(action_supervised)
        menu_ml.addAction(action_classification)
        menu_ml.addAction(action_metrics)
        lessons_menu.addMenu(menu_ml)

        menu_algorithms = QMenu('ML Models / Algorithms', self)
        action_knn = QAction('K-Nearest Neighbors', self)
        action_knn.triggered.connect(lambda _: open_url(url='https://www.ibm.com/think/topics/knn'))

        action_nb = QAction('Naive Bayes', self)
        action_nb.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/'))

        action_logistic = QAction('Logistic Regression', self)
        action_logistic.triggered.connect(lambda _: open_url(url='https://www.ibm.com/think/topics/logistic-regression'))

        action_tree = QAction('Decision Tree', self)
        action_tree.triggered.connect(lambda _: open_url(url='https://www.ibm.com/think/topics/decision-trees'))

        action_forest = QAction('Random Forest', self)
        action_forest.triggered.connect(lambda _: open_url(url='https://builtin.com/data-science/random-forest-algorithm'))

        action_xgboost = QAction('Extreme Boosting (XGBoost)', self)
        action_xgboost.triggered.connect(lambda _: open_url(url='https://www.ibm.com/think/topics/xgboost'))

        action_lda = QAction('Linear Discriminant Analysis (LDA)', self)
        action_lda.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/ml-linear-discriminant-analysis/'))

        action_svm = QAction('Support Vector Machines (SVM)', self)
        action_svm.triggered.connect(lambda _: open_url(url='https://spotintelligence.com/2024/05/06/support-vector-machines-svm/'))

        action_nn = QAction('Deep Neural Networks (DNN)', self)
        action_nn.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/neural-networks-a-beginners-guide/'))

        action_normalization = QAction('Feature Normalization', self)
        action_normalization.triggered.connect(
            lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/Feature-Engineering-Scaling-Normalization-and-Standardization/')
        )

        action_dnn_improvements = QAction('DNN Improvements', self)
        action_dnn_improvements.triggered.connect(lambda _: open_url(url='https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks'))

        action_interpretability = QAction('Model Interpretability / Explainability', self)
        action_interpretability.triggered.connect(lambda _: open_url(url='https://christophm.github.io/interpretable-ml-book/interpretability.html'))

        menu_algorithms.addAction(action_knn)
        menu_algorithms.addAction(action_nb)
        menu_algorithms.addAction(action_logistic)
        menu_algorithms.addAction(action_tree)
        menu_algorithms.addAction(action_forest)
        menu_algorithms.addAction(action_xgboost)
        menu_algorithms.addAction(action_lda)
        menu_algorithms.addAction(action_svm)
        menu_algorithms.addAction(action_nn)
        menu_algorithms.addSeparator()
        menu_algorithms.addAction(action_normalization)
        menu_algorithms.addAction(action_dnn_improvements)
        menu_algorithms.addSeparator()
        menu_algorithms.addAction(action_interpretability)
        lessons_menu.addMenu(menu_algorithms)

        menu_data = QMenu('Data Analysis', self)
        action_distribution = QAction('Variable Distribution', self)
        action_distribution.triggered.connect(lambda _: open_url(url='https://www.linkedin.com/pulse/understanding-data-distributions-examples-applications-r-s-8gkhc/'))

        action_correlation = QAction('Feature Correlations', self)
        action_correlation.triggered.connect(
            lambda _: open_url(url='https://users.sussex.ac.uk/~grahamh/RM1web/Eight%20things%20you%20need%20to%20know%20about%20interpreting%20correlations.pdf')
        )

        action_variance = QAction('Feature Variance', self)
        action_variance.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/maths/variance/'))

        action_coefficients = QAction('Regression Coefficients', self)
        action_coefficients.triggered.connect(lambda _: open_url(url='https://articles.outlier.org/coefficient-regression'))

        action_impurity = QAction('Impurity', self)
        action_impurity.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/machine-learning/gini-impurity-and-entropy-in-decision-tree-ml/'))

        action_boruta = QAction('Boruta Algorithm', self)
        action_boruta.triggered.connect(lambda _: open_url(url='https://www.blog.trainindata.com/is-boruta-dead/'))

        menu_data.addAction(action_distribution)
        menu_data.addAction(action_correlation)
        menu_data.addAction(action_variance)
        menu_data.addAction(action_coefficients)
        menu_data.addAction(action_impurity)
        menu_data.addAction(action_boruta)
        lessons_menu.addMenu(menu_data)

        menu_advanced = QMenu('Advanced ML Topics', self)
        action_imbalances = QAction('Class Imbalance', self)
        action_imbalances.triggered.connect(
            lambda _: open_url(url='https://isi-web.org/sites/default/files/2024-02/Handling-Data-Imbalance-in-Machine-Learning.pdf')
        )

        action_smote = QAction('SMOTE Algorithm', self)
        action_smote.triggered.connect(lambda _: open_url(url='https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/'))

        action_nearmiss = QAction('NearMiss Algorithm', self)
        action_nearmiss.triggered.connect(lambda _: open_url(url='https://www.linkedin.com/pulse/under-sampling-method-kaamil-ahmed/'))

        action_pdp = QAction('Partial Dependence Plot', self)
        action_pdp.triggered.connect(lambda _: open_url(url='https://www.geeksforgeeks.org/deep-learning/partial-dependence-plot-from-an-xgboost-model-in-r/'))

        action_shap = QAction('SHAP', self)
        shap_url = 'https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html'
        action_shap.triggered.connect(lambda _: open_url(url=shap_url))

        menu_advanced.addAction(action_imbalances)
        menu_advanced.addAction(action_smote)
        menu_advanced.addAction(action_nearmiss)
        menu_advanced.addSeparator()
        menu_advanced.addAction(action_pdp)
        menu_advanced.addAction(action_shap)
        lessons_menu.addMenu(menu_advanced)

        # Adding rest of the menus.
        action_update = QAction('Update', self)
        action_update.triggered.connect(lambda _: open_url(url='https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/tree/main'))

        action_bug = QAction('Submit Bug', self)
        action_bug.triggered.connect(lambda _: open_url(url='https://github.com/kochlisGit/ProphitBet-Soccer-Bets-Predictor/issues/new'))

        action_donation = QAction('Buy me a Coffee!', self)
        action_donation.triggered.connect(lambda _: open_url(url='https://www.paypal.com/donate/?hosted_button_id=AK3SEFDGVAWFE'))

        menu_help.addMenu(lessons_menu)
        menu_help.addAction(action_update)
        menu_help.addAction(action_bug)
        menu_help.addAction(action_donation)

    def _show_welcome_message(self):
        """ Pops-up a welcome message box. """

        # A modal welcome box can appear behind the WSLg window and make the
        # freshly launched app look frozen.  Keep the reminder visible without
        # blocking startup or automated league loading.
        self.statusBar().showMessage(
            f'欢迎使用 {PRODUCT_TITLE}。预测仅供数据分析参考，请理性使用。',
            10000,
        )
        league_ids = self._league_db.get_league_ids()
        if len(league_ids) == 0 and self._action_new is not None:
            self._action_new.trigger()
        elif len(league_ids) > 0 and self._league is None:
            # The half/full-time model currently belongs to the Premier League.
            # Prefer it at startup instead of the alphabetically first Chinese id.
            league_id = '英超' if '英超' in league_ids else league_ids[0]
            league = self._league_db.index[league_id]
            df = self._league_db.load_league(league_id=league_id)
            self._load_new_league(df=df, league=league)
            if sync_is_due():
                QTimer.singleShot(1000, lambda: self._start_league_sync(manual=False))

    def _start_league_sync(self, manual: bool = False):
        """Run all supported-league synchronization without freezing the GUI."""
        if self._sync_thread is not None and self._sync_thread.isRunning():
            if manual:
                QMessageBox.information(self, '数据同步', '全部联赛正在后台同步，请稍候。')
            return
        self._sync_manual = manual
        self.statusBar().showMessage('正在后台同步全部联赛新比赛…')
        self._sync_thread = QThread(self)
        self._sync_worker = ThreadWorker(sync_five_leagues)
        self._sync_worker.moveToThread(self._sync_thread)
        self._sync_thread.started.connect(self._sync_worker.run)
        self._sync_worker.finished.connect(self._on_sync_finished)
        self._sync_worker.error.connect(self._on_sync_error)
        self._sync_worker.finished.connect(self._sync_thread.quit)
        self._sync_worker.error.connect(self._sync_thread.quit)
        self._sync_thread.finished.connect(self._cleanup_sync)
        self._sync_thread.start()

    def _on_sync_finished(self, results: dict):
        added = sum(item.get('added', 0) for item in results.values())
        retrained = sum(item.get('models_retrained', 0) for item in results.values())
        message = f'同步完成：新增 {added} 场比赛，重新训练 {retrained} 个模型。'
        self.statusBar().showMessage(message, 15000)
        if (
                self._league is not None
                and results.get(self._league.league_id, {}).get('added', 0)
        ):
            fresh = self._league_db.load_league(self._league.league_id)
            league = self._league
            self._load_new_league(fresh, league)
        if self._sync_manual:
            details = '\n'.join(
                (
                    f'{league}：失败，{item.get("error", "未知错误")}'
                    if item.get('status') == 'rejected'
                    else f'{league}：新增 {item.get("added", 0)} 场，'
                         f'重训 {item.get("models_retrained", 0)} 个模型，'
                         f'最新 {item.get("latest_date", "-")}'
                )
                for league, item in results.items()
            )
            QMessageBox.information(self, '数据同步完成', f'{message}\n\n{details}')

    def _on_sync_error(self, message: str):
        self.statusBar().showMessage(f'自动同步失败：{message}', 15000)
        if self._sync_manual:
            QMessageBox.critical(self, '数据同步失败', message)

    def _cleanup_sync(self):
        if self._sync_worker is not None:
            self._sync_worker.deleteLater()
        if self._sync_thread is not None:
            self._sync_thread.deleteLater()
        self._sync_worker = None
        self._sync_thread = None

    def _load_new_league(self, df: Optional[pd.DataFrame], league: Optional[League]):
        """ Load league matches into the table and stores current matches, league id. """

        if df is None:
            return

        if self._league_df is not None:
            self._clear_table()

        # Add matches to the league table.
        display_df = translate_fixture_columns(df, league.league_id)
        self._table = DataFrameTable(parent=self, df=display_df, supports_sorting=True, supports_query_search=True)
        self.setCentralWidget(self._table)

        # Enable league menus and store data.
        league_title = LEAGUE_WINDOW_TITLES.get(league.league_id, 'League')
        self.setWindowTitle(f'{self._title} - {league_title}')
        self._set_league_menus_state(enabled=True)
        self._action_load.setEnabled(True)
        # Partial HTR coverage is expected for newly completed matches. It is
        # only required by half-full models and must not hide fresh data from
        # result, handicap, totals, or score workflows.
        required = df.drop(columns=['HTR'], errors='ignore').columns
        self._league_df = df.dropna(subset=required)
        self._league = league
        self._model_db = ModelDatabase(league_id=self._league.league_id)

    def _clear_table(self):
        """ Removes all league data from the table. """

        self._table.setParent(None)      # disconnects from its parent/layout
        self._table.deleteLater()        # schedules the object for deletion
        self.setCentralWidget(None)
        self._table = None

    def _set_league_menus_state(self, enabled: bool):
        """ Enables/Disables league functions. """

        self._action_close.setEnabled(enabled)
        self._menu_analysis.setEnabled(enabled)
        self._menu_models.setEnabled(enabled)
        self._menu_predictions.setEnabled(enabled)
