# -*- coding: utf-8 -*-
"""
===================================
A股板块分析服务 - 专门用于板块数据分析
===================================

职责：
1. 管理板块配置和股票列表
2. 执行板块数据分析
3. 整合双数据源验证
4. 生成板块分析报告（含 Kronos 时序预测信号）
5. 支持定时任务执行

支持的板块：
- PCB 板块
- 液冷板块
- 光模块板块
- AI算力板块
- 游戏板块
"""

from typing import Any, Dict, List, Optional, Tuple

import datetime
import json
import re
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# 板块配置 - 来自 a-share-5-sector 定时任务
SECTOR_CONFIG = [
    {
        "板块": "PCB 板块",
        "股票": ["生益科技", "深南电路", "沪电股份"],
        "代码": ["600183.SH", "002913.SZ", "002463.SZ"],
        "关键词": ["PCB 板块 最新研报", "PCB 行业 市场动态", "PCB 相关政策新闻"]
    },
    {
        "板块": "液冷板块",
        "股票": ["英维克", "申菱环境", "高澜股份"],
        "代码": ["002837.SZ", "301018.SZ", "300499.SZ"],
        "关键词": ["液冷技术 最新应用", "液冷服务器 市场需求", "数据中心液冷 行业发展"]
    },
    {
        "板块": "光模块板块",
        "股票": ["中际旭创", "新易盛", "天孚通信"],
        "代码": ["300308.SZ", "300502.SZ", "300394.SZ"],
        "关键词": ["光模块 行业报告", "800G 光模块 进展", "光通信产业链 动态"]
    },
    {
        "板块": "AI算力板块",
        "股票": ["海光信息", "昆仑万维", "中科曙光"],
        "代码": ["688041.SH", "300418.SZ", "603019.SH"],
        "关键词": ["AI算力 最新新闻", "服务器芯片 市场分析", "数据中心建设 政策"]
    },
    {
        "板块": "游戏板块",
        "股票": ["迅游科技", "宝通科技", "中青宝"],
        "代码": ["300467.SZ", "300031.SZ", "300052.SZ"],
        "关键词": ["游戏板块 最新研报", "游戏行业 市场动态", "游戏版号 政策新闻"]
    }
]

# 指数配置
INDEX_CONFIG = [
    {"名称": "上证指数", "代码": "000001.SH"},
    {"名称": "深证成指", "代码": "399001.SZ"},
    {"名称": "创业板指", "代码": "399006.SZ"}
]


class SectorAnalysisService:
    """A股板块分析服务"""

    def __init__(self) -> None:
        """初始化板块分析服务"""
        self.sector_config = SECTOR_CONFIG
        self.index_config = INDEX_CONFIG

    def is_trading_day(self) -> bool:
        """判断今天是否为交易日（周末不交易）"""
        today = datetime.datetime.now()
        return today.weekday() < 5

    # --- Wind MCP 数据获取方法 ---
    # TODO: 这些方法依赖子进程调用 MCP 技能，shell 路径为硬编码。
    #       后续可改为通过统一数据接口获取。

    def call_wind_mcp(self, server_type: str, tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """调用 Wind MCP 技能

        使用参数列表形式调用子进程，避免 shell=True 导致的命令注入风险。
        对 server_type 和 tool_name 做白名单校验。
        """
        if not re.fullmatch(r'[a-zA-Z0-9_]+', server_type):
            logger.error("非法的 server_type: %s", server_type)
            return None
        if not re.fullmatch(r'[a-zA-Z0-9_]+', tool_name):
            logger.error("非法的 tool_name: %s", tool_name)
            return None
        script_dir = os.path.expanduser('~/.agents/skills/wind-mcp-skill')
        if not os.path.isdir(script_dir):
            logger.error("Wind MCP 技能目录不存在: %s", script_dir)
            return None
        try:
            result = subprocess.check_output(
                ['node', 'scripts/cli.mjs', 'call', server_type, tool_name, json.dumps(params)],
                cwd=script_dir,
                text=True,
                timeout=120,
            )
            return json.loads(result)
        except subprocess.TimeoutExpired:
            logger.error("调用 Wind MCP 超时")
            return None
        except Exception as e:
            logger.error("调用 Wind MCP 技能失败: %s", e)
            return None

    def get_stock_price_indicators(self, windcode: str) -> Optional[Dict[str, Any]]:
        """获取股票价格指标"""
        params = {
            "windcode": windcode,
            "indexes": "中文简称,最新成交价,涨跌幅,成交量,成交额,换手率,市盈率(TTM),总市值2,当日主力净流入额"
        }
        return self.call_wind_mcp("stock_data", "get_stock_price_indicators", params)

    def get_index_price_indicators(self, windcode: str) -> Optional[Dict[str, Any]]:
        """获取指数价格指标"""
        params = {
            "windcode": windcode,
            "indexes": "中文简称,最新成交价,涨跌幅,成交量,成交额"
        }
        return self.call_wind_mcp("index_data", "get_index_price_indicators", params)

    # --- mx-data 技能调用方法 ---

    def call_mx_data(self, query: str) -> Optional[str]:
        """调用 mx-data 技能

        使用参数列表形式调用子进程，避免 shell=True 导致的命令注入风险。
        """
        script_dir = os.path.expanduser('~/.openclaw/skills/mx-data')
        if not os.path.isdir(script_dir):
            logger.error("mx-data 技能目录不存在: %s", script_dir)
            return None
        try:
            return subprocess.check_output(
                ['python3', 'mx_data.py', query],
                cwd=script_dir,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            logger.error("调用 mx-data 超时")
            return None
        except Exception as e:
            logger.error("调用 mx-data 技能失败: %s", e)
            return None

    # --- mx-search 技能调用方法 ---

    def call_mx_search(self, query: str) -> Optional[str]:
        """调用 mx-search 技能

        使用参数列表形式调用子进程，避免 shell=True 导致的命令注入风险。
        """
        script_dir = os.path.expanduser('~/.openclaw/skills/mx-search')
        if not os.path.isdir(script_dir):
            logger.error("mx-search 技能目录不存在: %s", script_dir)
            return None
        try:
            return subprocess.check_output(
                ['python3', 'mx_search.py', query],
                cwd=script_dir,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            logger.error("调用 mx-search 超时")
            return None
        except Exception as e:
            logger.error("调用 mx-search 技能失败: %s", e)
            return None

    def get_sector_stock_data(self) -> List[Dict[str, Any]]:
        """获取所有板块的股票数据"""
        all_stock_data = []

        for sector in self.sector_config:
            sector_data = {"板块": sector["板块"], "股票数据": []}
            logger.info("处理 %s...", sector["板块"])

            for stock_name, stock_code in zip(sector["股票"], sector["代码"]):
                logger.info("获取 %s(%s) 数据...", stock_name, stock_code)

                # 使用 Wind MCP 获取数据
                wind_result = self.get_stock_price_indicators(stock_code)
                if wind_result and 'content' in wind_result and len(wind_result['content']) > 0:
                    text = wind_result['content'][0].get('text', '')
                    try:
                        data = json.loads(text)
                        if data.get("error") is None and "data" in data and "rows" in data["data"]:
                            for row in data["data"]["rows"]:
                                stock_info = {
                                    "股票名称": row[0],
                                    "最新成交价": row[1],
                                    "涨跌幅": row[2],
                                    "成交量": row[3],
                                    "成交额": row[4],
                                    "换手率": row[5],
                                    "市盈率(TTM)": row[6],
                                    "总市值2": row[7],
                                    "当日主力净流入额": row[8],
                                    "Wind代码": row[9]
                                }
                                sector_data["股票数据"].append(stock_info)
                                logger.info("Wind MCP 数据获取成功: %s", stock_name)
                    except Exception as e:
                        logger.error("解析 Wind MCP 数据失败: %s", e)

                # 使用 mx-data 获取数据作为补充
                mx_result = self.call_mx_data(f"{stock_name} 最新价 涨跌幅 成交量")
                if mx_result:
                    logger.info("mx-data 数据获取成功: %s", stock_name)

            all_stock_data.append(sector_data)

        return all_stock_data

    def get_index_data(self) -> List[Dict[str, Any]]:
        """获取指数数据"""
        index_data = []

        for index in self.index_config:
            logger.info("获取 %s(%s) 数据...", index["名称"], index["代码"])
            result = self.get_index_price_indicators(index["代码"])

            if result and 'content' in result and len(result['content']) > 0:
                text = result['content'][0].get('text', '')
                try:
                    data = json.loads(text)
                    if data.get("error") is None and "data" in data and "rows" in data["data"]:
                        for row in data["data"]["rows"]:
                            index_info = {
                                "名称": row[0],
                                "最新成交价": row[1],
                                "涨跌幅": row[2],
                                "成交量": row[3],
                                "成交额": row[4],
                                "Wind代码": row[5]
                            }
                            index_data.append(index_info)
                            logger.info("成功获取 %s 数据", index["名称"])
                except Exception as e:
                    logger.error("解析指数数据失败: %s", e)

        return index_data

    def get_news_search_results(self) -> List[Dict[str, Any]]:
        """获取资讯搜索结果"""
        search_results = []

        for search_config in self.sector_config:
            search_data = {"板块": search_config["板块"], "关键词": search_config["关键词"], "搜索结果": ""}
            for keyword in search_config["关键词"]:
                logger.info("搜索: %s", keyword)
                search_result = self.call_mx_search(keyword)
                if search_result:
                    search_data["搜索结果"] += search_result
            search_results.append(search_data)

        return search_results

    def analyze_data(self, all_stock_data: List[Dict[str, Any]], index_data: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
        """分析板块数据"""
        logger.info("分析板块数据...")

        avg_change: Dict[str, float] = {}
        best_stock: Dict[str, Any] = {}
        worst_stock: Dict[str, Any] = {}

        for sector in all_stock_data:
            if "股票数据" in sector and len(sector["股票数据"]) > 0:
                changes = []
                for stock in sector["股票数据"]:
                    if stock.get("涨跌幅"):
                        try:
                            changes.append(float(stock["涨跌幅"]))
                        except Exception:
                            pass

                if changes:
                    avg_change[sector["板块"]] = sum(changes) / len(changes)

                    # 找到板块中涨跌幅最大和最小的股票
                    sorted_stocks = sorted(
                        sector["股票数据"],
                        key=lambda x: float(x["涨跌幅"]) if x.get("涨跌幅") else -1000,
                        reverse=True
                    )
                    best_stock[sector["板块"]] = sorted_stocks[0]

                    sorted_stocks = sorted(
                        sector["股票数据"],
                        key=lambda x: float(x["涨跌幅"]) if x.get("涨跌幅") else 1000
                    )
                    worst_stock[sector["板块"]] = sorted_stocks[0]

        return avg_change, best_stock, worst_stock

    def _run_kronos_predictions(self, report_date: str) -> Dict[str, Dict[str, Any]]:
        """对每个板块内的股票执行 Kronos 预测，按板块汇总。

        Args:
            report_date: 报告日期，格式 'YYYY-MM-DD'

        Returns:
            {sector_name: {
                "total": N, "bullish": N, "bearish": N, "neutral": N,
                "details": [{name, code, direction, confidence, pct_change}, ...]
            }}
            若 Kronos 不可用或无结果，返回空字典。
        """
        try:
            from src.services.kronos_service import KronosService
            ks = KronosService()
            if not ks.is_available:
                logger.info("Kronos 模型不可用，跳过 Kronos 预测")
                return {}
        except Exception as e:
            logger.warning("KronosService 初始化失败: %s", e)
            return {}

        logger.info("开始执行 Kronos 板块预测...")
        results: Dict[str, Dict[str, Any]] = {}

        for sector in self.sector_config:
            sector_name = sector["板块"]
            sector_codes = sector["代码"]
            sector_results: List[Dict[str, Any]] = []

            for stock_name, stock_code in zip(sector["股票"], sector_codes):
                # 剥离交易所后缀（.SH/.SZ）得到 6 位代码
                kronos_code = stock_code.split(".")[0]
                try:
                    pred = ks.predict(kronos_code, report_date)
                    if pred and pred.available:
                        sector_results.append({
                            "name": stock_name,
                            "code": kronos_code,
                            "direction": pred.direction,
                            "confidence": pred.direction_confidence,
                            "pct_change": pred.pred_pct_change,
                        })
                        logger.debug("Kronos %s(%s): %s", stock_name, kronos_code, pred.direction)
                except Exception as e:
                    logger.debug("Kronos 预测失败 %s(%s): %s", stock_name, kronos_code, e)
                    continue

            if sector_results:
                bullish = sum(1 for r in sector_results if r["direction"] == "看多")
                bearish = sum(1 for r in sector_results if r["direction"] == "看空")
                neutral = sum(1 for r in sector_results if r["direction"] == "震荡")
                results[sector_name] = {
                    "total": len(sector_results),
                    "bullish": bullish,
                    "bearish": bearish,
                    "neutral": neutral,
                    "details": sector_results,
                }
                logger.info("Kronos %s: %d看多, %d看空, %d震荡",
                           sector_name, bullish, bearish, neutral)

        return results

    def generate_report(
        self,
        all_stock_data: List[Dict[str, Any]],
        index_data: List[Dict[str, Any]],
        analysis_results: Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]],
        search_results: List[Dict[str, Any]],
        kronos_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """生成板块分析报告"""
        logger.info("生成板块分析报告...")

        today = datetime.datetime.now()
        report_content: List[str] = []

        report_content.append("# A股每日板块分析报告")
        report_content.append(f"## 报告日期: {today.strftime('%Y-%m-%d %H:%M')}")

        report_content.append("\n## 大盘分析")
        for index in index_data:
            report_content.append(f"- **{index['名称']}**: {index['最新成交价']} ({index['涨跌幅']}%)")

        report_content.append("\n## 板块分析")
        avg_change, best_stock, worst_stock = analysis_results
        for sector in all_stock_data:
            report_content.append(f"### {sector['板块']}")
            if sector['板块'] in avg_change:
                report_content.append(f"平均涨跌幅: {avg_change[sector['板块']]:.2f}%")

            report_content.append("\n股票详情:")
            for stock in sector["股票数据"]:
                report_content.append(f"- **{stock['股票名称']}**: {stock['最新成交价']} ({stock['涨跌幅']}%)")

        report_content.append("\n## 表现最好的股票")
        for sector, stock in best_stock.items():
            report_content.append(f"- **{sector}**: {stock['股票名称']} ({stock['涨跌幅']}%)")

        report_content.append("\n## 表现最差的股票")
        for sector, stock in worst_stock.items():
            report_content.append(f"- **{sector}**: {stock['股票名称']} ({stock['涨跌幅']}%)")

        report_content.append("\n## 资讯摘要")
        for search in search_results:
            report_content.append(f"### {search['板块']}")
            report_content.append(f"搜索关键词: {', '.join(search['关键词'])}")
            if search.get("搜索结果"):
                report_content.append("搜索结果预览:")
                preview = search['搜索结果'][:2000]
                report_content.append(preview + ("..." if len(search['搜索结果']) > 2000 else ""))

        # Kronos 时序预测信号（新增）
        if kronos_results:
            report_content.append("\n## Kronos 时序预测信号")
            for sector in self.sector_config:
                sector_name = sector["板块"]
                if sector_name not in kronos_results:
                    continue
                kr = kronos_results[sector_name]
                if not kr["details"]:
                    continue

                report_content.append(f"\n### {sector_name}")
                report_content.append("| 股票 | 方向 | 置信度 | 预测涨跌幅 |")
                report_content.append("|------|------|--------|-----------|")
                for d in kr["details"]:
                    report_content.append(
                        f"| {d['name']} | {d['direction']} | {d['confidence']:.1f}% | {d['pct_change']:+.2f}% |"
                    )
                report_content.append(
                    f"板块信号汇总: {kr['bullish']}看多, {kr['bearish']}看空, {kr['neutral']}震荡"
                )

        return "\n".join(report_content)

    def run_analysis(self, report_date: Optional[str] = None) -> Optional[str]:
        """运行完整的板块分析流程。

        Args:
            report_date: 报告日期，格式 'YYYY-MM-DD'。默认使用当前日期。

        Returns:
            生成的报告内容（Markdown），非交易日返回 None
        """
        if report_date is None:
            report_date = datetime.datetime.now().strftime("%Y-%m-%d")

        logger.info("开始执行板块分析（日期: %s）...", report_date)

        # 1. 确认是否为交易日
        if not self.is_trading_day():
            logger.info("今天是周末，不是交易日")
            return None

        # 2. 获取股票行情数据
        all_stock_data = self.get_sector_stock_data()

        # 3. 获取大盘指数数据
        index_data = self.get_index_data()

        # 4. 分析数据
        analysis_results = self.analyze_data(all_stock_data, index_data)

        # 5. 资讯搜索
        search_results = self.get_news_search_results()

        # 6. Kronos 时序预测
        kronos_results = self._run_kronos_predictions(report_date)

        # 7. 生成综合分析报告
        report = self.generate_report(
            all_stock_data, index_data, analysis_results, search_results,
            kronos_results=kronos_results,
        )

        logger.info("板块分析执行完成")
        return report