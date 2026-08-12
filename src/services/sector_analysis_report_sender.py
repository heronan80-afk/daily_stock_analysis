# -*- coding: utf-8 -*-
"""
===================================
A股板块分析报告发送器
===================================

职责：
1. 使用项目的通知系统发送板块分析报告
2. 支持飞书应用机器人和飞书 Webhook 两种发送方式
3. 自动识别可用的通知通道
4. 对过长的报告进行分片发送

依赖：
- src.notification_sender.feishu_sender.FeishuSender
"""

import logging
from typing import Optional

from src.config import Config, get_config
from src.notification_sender.feishu_sender import FeishuSender

logger = logging.getLogger(__name__)


class SectorAnalysisReportSender:
    """
    板块分析报告发送器

    使用项目的通知系统发送板块分析报告
    支持飞书应用机器人和飞书 Webhook 两种发送方式
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化发送器

        Args:
            config: 系统配置（可选，默认使用 get_config()）
        """
        self.config = config or get_config()
        self.feishu_sender = FeishuSender(self.config)
        logger.info("板块分析报告发送器初始化完成")

    def send_report(self, report_content: str) -> bool:
        """
        发送板块分析报告

        支持以下发送方式：
        1. 飞书应用机器人（优先）
        2. 飞书 Webhook（备选）

        Args:
            report_content: 报告内容（Markdown 格式）

        Returns:
            发送是否成功
        """
        if not report_content:
            logger.error("报告内容不能为空")
            return False

        logger.info("开始发送板块分析报告")

        # 发送报告到配置的聊天 ID
        success = False
        if self.config.feishu_chat_id:
            logger.info("使用飞书应用机器人发送报告")
            success = self._send_via_feishu_app_bot(report_content)
        else:
            logger.warning("FEISHU_CHAT_ID 未配置，尝试使用其他通知方式")
            success = self._send_via_feishu_webhook(report_content)

        if success:
            logger.info("板块分析报告发送成功")
        else:
            logger.error("板块分析报告发送失败")

        return success

    def _send_via_feishu_app_bot(self, report_content: str) -> bool:
        """
        使用飞书应用机器人发送报告

        Args:
            report_content: 报告内容（Markdown 格式）

        Returns:
            发送是否成功
        """
        try:
            logger.info("配置的飞书聊天 ID: %s", self.config.feishu_chat_id)
            return self.feishu_sender.send_to_feishu(report_content)
        except Exception as e:
            logger.error("飞书应用机器人发送失败: %s", e)
            return False

    def _send_via_feishu_webhook(self, report_content: str) -> bool:
        """
        使用飞书 Webhook 发送报告

        Args:
            report_content: 报告内容（Markdown 格式）

        Returns:
            发送是否成功
        """
        try:
            logger.info("使用飞书 Webhook 发送报告")
            return self.feishu_sender.send_to_feishu(report_content)
        except Exception as e:
            logger.error("飞书 Webhook 发送失败: %s", e)
            return False

    def send_notification(self, notification_content: str) -> bool:
        """
        发送通知消息

        Args:
            notification_content: 通知内容（Markdown 格式）

        Returns:
            发送是否成功
        """
        return self.send_report(notification_content)


def create_sector_analysis_report_sender() -> SectorAnalysisReportSender:
    """
    创建板块分析报告发送器实例

    Returns:
        SectorAnalysisReportSender 实例
    """
    return SectorAnalysisReportSender(get_config())