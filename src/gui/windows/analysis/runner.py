from typing import Callable

from PySide6.QtWidgets import QMessageBox, QWidget

from src.gui.utils.taskrunner import TaskRunnerDialog
from src.gui.widgets.plot import PlotWindow


def run_plot_analysis(parent: QWidget, title: str, task_fn: Callable):
    """Run CPU-heavy analysis away from the Qt UI thread and display its plot."""
    runner = TaskRunnerDialog(
        parent=parent,
        title=title,
        info="正在后台分析数据，请稍候…",
        task_fn=task_fn,
    )
    ax = runner.run()
    if runner.error_message is not None or ax is None:
        QMessageBox.critical(
            parent,
            "分析失败",
            runner.error_message or "分析没有返回结果。",
        )
        return None
    window = PlotWindow(ax=ax, parent=parent, title=title)
    window.show()
    return window

