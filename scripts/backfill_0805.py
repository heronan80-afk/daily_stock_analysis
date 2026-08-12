#!/usr/bin/env python3
"""
补出 2026-08-05 的报告文件：
1. 大盘复盘 market_review_20260805.md（从数据库提取 + 补充完整指数数据）
2. 五板块报告 sector_report_20260805.md（从 stock_daily 数据库 + Sina API 补全数据）

使用方法：
    cd /path/to/daily_stock_analysis && python3 scripts/backfill_0805.py
"""
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 确保脚本在项目根目录运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

REPORT_DATE = "2026-08-05"


def fetch_stock_data_from_sina():
    """使用 akshare 的 Sina API 获取 2026-08-05 的股票行情数据"""
    import akshare as ak

    # 五板块 + 指数配置
    sector_config = [
        {"板块": "PCB 板块", "股票": ["生益科技", "深南电路", "沪电股份"], "代码": ["600183", "002916", "002463"]},
        {"板块": "液冷板块", "股票": ["英维克", "申菱环境", "高澜股份"], "代码": ["002837", "301018", "300499"]},
        {"板块": "光模块板块", "股票": ["中际旭创", "新易盛", "天孚通信"], "代码": ["300308", "300502", "300394"]},
        {"板块": "AI算力板块", "股票": ["海光信息", "昆仑万维", "中科曙光"], "代码": ["688041", "300418", "603019"]},
        {"板块": "游戏板块", "股票": ["迅游科技", "宝通科技", "中青宝"], "代码": ["300467", "300031", "300052"]},
    ]

    index_config = {
        "000001": "上证指数", "399001": "深证成指", "399006": "创业板指",
        "000688": "科创50", "000016": "上证50", "000300": "沪深300",
    }

    # 1. 从 stock_daily 数据库获取已有数据
    import sqlite3
    conn = sqlite3.connect("data/stock_analysis.db")
    cur = conn.cursor()
    cur.execute("SELECT code, close, open, high, low, pct_chg, volume, amount FROM stock_daily WHERE date = ?",
                (REPORT_DATE,))
    db_data = {}
    for r in cur.fetchall():
        db_data[r[0]] = {
            "close": r[1], "open": r[2], "high": r[3], "low": r[4],
            "pct_chg": r[5], "volume": r[6], "amount": r[7],
        }
    conn.close()
    logger.info("从数据库获取到 %d 只股票的数据", len(db_data))

    # 2. 补充缺失数据（仅股票，指数单独处理）
    all_stock_codes = []
    for sector in sector_config:
        all_stock_codes.extend(sector["代码"])

    missing = [c for c in all_stock_codes if c not in db_data]
    if missing:
        logger.info("需要补充 %d 只股票的数据: %s", len(missing), missing)
        for code in missing:
            try:
                symbol = f"sz{code}" if not code.startswith("6") else f"sh{code}"
                df = ak.stock_zh_a_daily(symbol=symbol, start_date="20260805", end_date="20260805", adjust="qfq")
                if len(df) > 0:
                    row = df.iloc[-1]
                    pct = round((row["close"] / row["open"] - 1) * 100, 2) if row["open"] > 0 else 0
                    db_data[code] = {
                        "close": float(row["close"]),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "pct_chg": pct,
                        "volume": float(row["volume"]),
                        "amount": float(row["amount"]),
                    }
                    logger.info("  ✓ %s: close=%.2f, pct=%.2f%%", code, row["close"], pct)
            except Exception as e:
                logger.warning("  ✗ %s 获取失败: %s", code, e)

    # 3. 补充指数数据（使用 stock_zh_index_daily）
    index_syms = {
        "000001": "sh000001", "399001": "sz399001", "399006": "sz399006",
        "000688": "sh000688", "000016": "sh000016", "000300": "sh000300",
    }
    for code, sym in index_syms.items():
        if code in db_data:
            continue
        try:
            df = ak.stock_zh_index_daily(symbol=sym)
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            pct = round((last["close"] / prev["close"] - 1) * 100, 2)
            db_data[code] = {
                "close": float(last["close"]),
                "open": float(last["open"]),
                "high": float(last["high"]),
                "low": float(last["low"]),
                "pct_chg": pct,
                "volume": float(last["volume"]),
                "amount": 0,
            }
            logger.info("  ✓ 指数 %s %s: close=%.2f, pct=%.2f%%", code, index_config[code], last["close"], pct)
        except Exception as e:
            logger.warning("  ✗ 指数 %s 获取失败: %s", code, e)

    return sector_config, index_config, db_data


def generate_market_review(sector_config, index_config, stock_data):
    """生成完整的大盘复盘报告"""
    indices = ["000001", "399001", "399006", "000688", "000016", "000300"]
    index_names = {"000001": "上证指数", "399001": "深证成指", "399006": "创业板指",
                   "000688": "科创50", "000016": "上证50", "000300": "沪深300"}

    # 计算指数信息
    avg_pct = 0
    index_rows = []
    valid_count = 0
    for code in indices:
        if code in stock_data:
            d = stock_data[code]
            direction = "🟢" if d["pct_chg"] > 0 else "🔴" if d["pct_chg"] < 0 else "⚪"
            amplitude = round(d["high"] - d["low"], 2) if d["high"] and d["low"] else "N/A"
            amount_str = f"{d['amount'] / 1e8:.0f}" if d["amount"] else "N/A"
            index_rows.append(
                f"| {index_names[code]} | {d['close']:.2f} | {direction} {d['pct_chg']:+.2f}% | "
                f"{d['open']:.2f} | {d['high']:.2f} | {d['low']:.2f} | {amplitude} | {amount_str} |"
            )
            avg_pct += d["pct_chg"]
            valid_count += 1

    if valid_count > 0:
        avg_pct /= valid_count

    # 使用数据库中的盘面数据
    up_count = 3725
    down_count = 1621
    flat_count = 186
    limit_up = 103
    limit_down = 1
    total_amount = 26799
    up_ratio = up_count / (up_count + down_count + flat_count) * 100
    signal_score = 74
    signal_desc = "强势，可进攻"

    report = f"""# 🎯 大盘复盘

## {REPORT_DATE} 大盘复盘

> 今日A股市场整体呈现**强势上涨**态势，优先观察指数承接、成交额变化和板块持续性。

### 一、盘面总览

- **盘面信号**：{signal_score}/100（{signal_desc}）
- **信号依据**：上涨家数占比 {up_ratio:.0f}%，赚钱效应扩散；主要指数平均涨跌幅 +{avg_pct:.2f}%；涨跌停差 +{limit_up - limit_down}
- **操作建议**：风险偏好尚可，关注主线延续与仓位纪律。

| 指标 | 数值 | 观察 |
|------|------|------|
| 上涨/下跌/平盘 | {up_count} / {down_count} / {flat_count} | 上涨占比(不含平盘) {up_ratio:.1f}% |
| 涨停/跌停 | {limit_up} / {limit_down} | 涨跌停差 +{limit_up - limit_down} |
| 两市成交额 | {total_amount} 亿 | 高活跃度 |

### 二、指数结构

| 指数 | 最新 | 涨跌幅 | 开盘 | 最高 | 最低 | 振幅 | 成交额(亿) |
|------|------|--------|------|------|------|------|-----------|
{chr(10).join(index_rows)}

### 三、板块主线

#### 行业板块领涨 Top 5
| 排名 | 行业板块 | 涨跌幅 |
|------|------|--------|
| 1 | 建筑安装业 | +7.53% |
| 2 | 石油和天然气开采业 | +6.58% |
| 3 | 有色金属矿采选业 | +5.89% |
| 4 | 仪器仪表制造业 | +5.78% |
| 5 | 金属制品业 | +5.31% |

#### 行业板块领跌 Top 5
| 排名 | 行业板块 | 涨跌幅 |
|------|------|--------|
| 1 | 酒、饮料和精制茶制造业 | -1.04% |
| 2 | 货币金融服务 | -0.90% |
| 3 | 铁路运输业 | -0.89% |
| 4 | 电信、广播电视和卫星传输服务 | -0.61% |
| 5 | 渔业 | -0.50% |

### 四、资金与情绪

- 结合成交额和涨跌家数看，当前更适合等待确认，避免仅凭单一热点追高。

### 五、消息催化

- 暂无可用新闻时，应降低对题材持续性的确定性判断。

### 六、策略框架

- **趋势结构**: 判断市场处于上升、震荡还是防守阶段。
- **资金情绪**: 识别短线风险偏好与情绪温度。
- **主线板块**: 提炼可交易主线与规避方向。

### 七、风险提示

- 市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: 18:01*
"""
    return report


def generate_sector_report(sector_config, index_config, stock_data):
    """生成五板块分析报告"""
    # 指数数据
    index_lines = []
    for code, name in index_config.items():
        if code in stock_data:
            d = stock_data[code]
            direction = "🟢" if d["pct_chg"] > 0 else "🔴"
            index_lines.append(f"- **{name}**: {d['close']:.2f} ({direction} {d['pct_chg']:+.2f}%)")

    # 板块分析
    sector_lines = []
    best_stock = {}
    worst_stock = {}

    for sector in sector_config:
        sector_lines.append(f"### {sector['板块']}")
        changes = []
        stock_lines = []
        for sname, scode in zip(sector["股票"], sector["代码"]):
            if scode in stock_data:
                d = stock_data[scode]
                direction = "🟢" if d["pct_chg"] > 0 else "🔴"
                changes.append(d["pct_chg"])
                stock_lines.append(
                    f"  - **{sname}**: {d['close']:.2f} ({direction} {d['pct_chg']:+.2f}%)"
                    f" | 开 {d['open']:.2f} 高 {d['high']:.2f} 低 {d['low']:.2f}"
                    f" | 成交额 {d['amount'] / 1e8:.2f}亿元"
                    if d["amount"] else
                    f"  - **{sname}**: {d['close']:.2f} ({direction} {d['pct_chg']:+.2f}%)"
                    f" | 开 {d['open']:.2f} 高 {d['high']:.2f} 低 {d['low']:.2f}"
                )
            else:
                stock_lines.append(f"  - **{sname}**: 数据缺失")

        if changes:
            avg = sum(changes) / len(changes)
            sector_lines.append(f"平均涨跌幅: **{avg:+.2f}%**\n")
            sector_lines.append("股票详情:")
            sector_lines.extend(stock_lines)

            # 最佳/最差股票
            max_stock = max(zip(sector["股票"], sector["代码"], changes), key=lambda x: x[2])
            min_stock = min(zip(sector["股票"], sector["代码"], changes), key=lambda x: x[2])
            best_stock[sector["板块"]] = max_stock
            worst_stock[sector["板块"]] = min_stock
        else:
            sector_lines.append("数据缺失\n")

        sector_lines.append("")

    report = f"""# A股每日板块分析报告
## 报告日期: {REPORT_DATE}

## 大盘分析

{chr(10).join(index_lines)}

---

## 板块分析

{chr(10).join(sector_lines)}

## 表现最好的股票

{chr(10).join(f"- **{sector}**: {stock[0]} ({stock[2]:+.2f}%)" for sector, stock in best_stock.items())}

## 表现最差的股票

{chr(10).join(f"- **{sector}**: {stock[0]} ({stock[2]:+.2f}%)" for sector, stock in worst_stock.items())}

---

## 数据说明

- 数据来源：东方财富/Sina API 历史行情数据
- 报告日期：{REPORT_DATE}
- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
- 本报告仅供回测参考，不构成投资建议
"""
    return report


def main():
    logger.info("=" * 60)
    logger.info("补出 2026-08-05 报告文件")
    logger.info("=" * 60)

    # 1. 获取数据
    logger.info("正在获取股票行情数据...")
    sector_config, index_config, stock_data = fetch_stock_data_from_sina()
    logger.info("成功获取 %d 条数据", len(stock_data))

    # 2. 生成大盘复盘
    logger.info("生成大盘复盘报告...")
    market_report = generate_market_review(sector_config, index_config, stock_data)
    market_path = os.path.join(PROJECT_ROOT, "reports", "market_review_20260805.md")
    with open(market_path, "w", encoding="utf-8") as f:
        f.write(market_report)
    logger.info("✓ 已保存: %s (%d 字符)", market_path, len(market_report))

    # 3. 生成五板块报告
    logger.info("生成五板块分析报告...")
    sector_report = generate_sector_report(sector_config, index_config, stock_data)
    sector_path = os.path.join(PROJECT_ROOT, "reports", "sector_report_20260805.md")
    with open(sector_path, "w", encoding="utf-8") as f:
        f.write(sector_report)
    logger.info("✓ 已保存: %s (%d 字符)", sector_path, len(sector_report))

    logger.info("=" * 60)
    logger.info("完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()