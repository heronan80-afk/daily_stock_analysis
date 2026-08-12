# -*- coding: utf-8 -*-
"""
===================================
A股板块分析调度器
===================================

职责：
1. 运行板块分析任务
2. 发送板块分析报告
3. 处理调度异常

依赖：
- src.services.sector_analysis_service.SectorAnalysisService
- src.services.sector_analysis_report_sender.SectorAnalysisReportSender
"""

import logging
from typing import Optional

from src.config import Config, get_config
from src.services.sector_analysis_service import SectorAnalysisService
from src.services.sector_analysis_report_sender import SectorAnalysisReportSender, create_sector_analysis_report_sender

logger = logging.getLogger(__name__)


class SectorAnalysisScheduler:
    """
    板块分析调度器

    负责运行板块分析任务并发送报告
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化调度器

        Args:
            config: 系统配置（可选，默认使用 get_config()）
        """
        self.config = config or get_config()
        self.analysis_service = SectorAnalysisService()
        self.report_sender = create_sector_analysis_report_sender()
        logger.info("板块分析调度器初始化完成")

    def run_task(self) -> bool:
        """
        运行板块分析任务并发送报告

        Returns:
            任务是否成功完成
        """
        logger.info("开始执行板块分析调度任务")

        try:
            # 运行板块分析
            report_content = self.analysis_service.run_analysis()

            if report_content:
                logger.info("板块分析完成，开始发送报告")
                # 发送报告
                success = self.report_sender.send_report(report_content)
                if success:
                    logger.info("板块分析报告发送成功")
                else:
                    logger.error("板块分析报告发送失败")
                return success
            else:
                logger.info("板块分析未生成报告")
                return True  # 非交易日等情况不视为失败

        except Exception as e:
            logger.error("板块分析调度任务失败: %s", e, exc_info=True)
            # 发送错误通知
            error_message = (
                f"# 板块分析任务执行失败\n"
                f"## 错误信息\n"
                f"任务执行过程中发生错误：\n"
                f"```\n{str(e)}\n```\n"
                f"## 建议\n"
                f"请检查系统日志以获取更多详细信息。"
            )
            try:
                self.report_sender.send_notification(error_message)
            except Exception as notify_error:
                logger.error("发送错误通知失败: %s", notify_error)
            return False

    def run_task_wrapper(self):
        """
        任务包装函数，用于调度器直接调用（无参数）
        """
        self.run_task()


def create_sector_analysis_scheduler() -> SectorAnalysisScheduler:
    """
    创建板块分析调度器实例

    Returns:
        SectorAnalysisScheduler 实例
    """
    return SectorAnalysisScheduler(get_config())