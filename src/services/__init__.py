# -*- coding: utf-8 -*-
"""
===================================
服务层模块初始化
===================================

职责：
1. 声明可导出的服务类（延迟导入，避免启动时拉入 LLM 等重依赖）

使用方式：
    直接从子模块导入，例如:
    from src.services.history_service import HistoryService
"""


def __getattr__(name: str):
    """延迟导入：仅在通过 src.services.X 访问时才加载对应子模块。"""
    _lazy_map = {
        "AnalysisService": "src.services.analysis_service",
        "BacktestService": "src.services.backtest_service",
        "HistoryService": "src.services.history_service",
        "KronosService": "src.services.kronos_service",
        "SectorAnalysisService": "src.services.sector_analysis_service",
        "SectorAnalysisReportSender": "src.services.sector_analysis_report_sender",
        "SectorAnalysisScheduler": "src.services.sector_analysis_scheduler",
        "StockService": "src.services.stock_service",
        "TaskService": "src.services.task_service",
        "create_sector_analysis_report_sender": "src.services.sector_analysis_report_sender",
        "create_sector_analysis_scheduler": "src.services.sector_analysis_scheduler",
        "get_task_service": "src.services.task_service",
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module 'src.services' has no attribute {name!r}")


__all__ = [
    "AnalysisService",
    "BacktestService",
    "HistoryService",
    "KronosService",
    "SectorAnalysisService",
    "SectorAnalysisReportSender",
    "SectorAnalysisScheduler",
    "StockService",
    "TaskService",
    "create_sector_analysis_report_sender",
    "create_sector_analysis_scheduler",
    "get_task_service",
]
