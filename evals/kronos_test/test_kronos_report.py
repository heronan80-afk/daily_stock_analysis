#!/usr/bin/env python3
"""Kronos 回测：用历史报告中的股票测试 Kronos 预测能力"""
import sys, os, json
import pandas as pd
import numpy as np

sys.path.insert(0, '/tmp/Kronos')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from data_provider.tencent_fetcher import TencentFetcher
from model import Kronos, KronosTokenizer, KronosPredictor

REPORT_DATE = "2026-07-10"
LOOKBACK = 400
PRED_LEN = 15
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

STOCKS = {
    "300502": "新易盛",
    "603019": "中科曙光",
    "300308": "中际旭创",
    "600183": "生益科技",
    "300418": "昆仑万维",
    "002463": "沪电股份",
    "300031": "宝通科技",
}

def fetch_data(fetcher, symbol):
    df = fetcher.get_daily_data(symbol, start_date="2024-01-01", end_date="2026-07-31")
    cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[cols].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df

def run_prediction(predictor, df, report_date, pred_len=20):
    report_dt = pd.to_datetime(report_date)
    before = df[df["date"] <= report_dt].reset_index(drop=True)
    after = df[df["date"] > report_dt].reset_index(drop=True)
    
    if len(before) < LOOKBACK:
        raise ValueError(f"历史数据不足: {len(before)} < {LOOKBACK}")
    
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    x_df = before.iloc[-LOOKBACK:][price_cols].reset_index(drop=True)
    x_ts = before.iloc[-LOOKBACK:]['date'].reset_index(drop=True)
    
    actual_len = min(len(after), pred_len)
    y_ts = after.iloc[:actual_len]['date'].reset_index(drop=True)
    
    pred_df = predictor.predict(
        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
        pred_len=actual_len, T=1.0, top_p=0.9, sample_count=1, verbose=False
    )
    
    return pred_df, after.iloc[:actual_len].reset_index(drop=True), before

def calc_metrics(actual, pred):
    n = min(len(actual), len(pred))
    a, p = actual[:n], pred[:n]
    mape = float(np.mean(np.abs((a - p) / a)) * 100)
    a_ret, p_ret = np.diff(a), np.diff(p)
    dir_acc = float(np.mean(np.sign(a_ret) == np.sign(p_ret)) * 100) if len(a_ret) > 0 else 0.0
    a_total = float((a[-1] - a[0]) / a[0] * 100)
    p_total = float((p[-1] - p[0]) / p[0] * 100)
    total_dir = bool(np.sign(a_total) == np.sign(p_total))
    return {
        "mape": round(mape, 2),
        "direction_accuracy": round(dir_acc, 1),
        "actual_total_return_pct": round(a_total, 2),
        "pred_total_return_pct": round(p_total, 2),
        "total_direction_correct": total_dir,
        "pred_days": n
    }

def main():
    print("=" * 70)
    print(f"Kronos 回测 - 报告日 {REPORT_DATE}, 预测 {PRED_LEN} 天")
    print("=" * 70)
    
    print("\n🤗 加载 Kronos-small 模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", token=False)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", token=False)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   ✅ 参数: {params:.1f}M")
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    print(f"   设备: {device}")
    
    fetcher = TencentFetcher()
    results = {}
    
    for symbol, name in STOCKS.items():
        print(f"\n{'='*60}")
        print(f"📊 {name} ({symbol})")
        print("-" * 60)
        try:
            df = fetch_data(fetcher, symbol)
            print(f"   数据: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')} ({len(df)}天)")
            
            pred_df, actual_df, before = run_prediction(predictor, df, REPORT_DATE, PRED_LEN)
            last_price = float(before.iloc[-1]['close'])
            print(f"   报告日收盘价: {last_price:.2f}")
            print(f"   预测天数: {len(pred_df)}")
            
            m = calc_metrics(actual_df['close'].values, pred_df['close'].values[:len(actual_df)])
            print(f"\n   📐 结果:")
            print(f"      MAPE: {m['mape']}%")
            print(f"      日间方向准确率: {m['direction_accuracy']}%")
            print(f"      实际涨跌: {m['actual_total_return_pct']:+.2f}%")
            print(f"      预测涨跌: {m['pred_total_return_pct']:+.2f}%")
            print(f"      趋势方向: {'✅ 正确' if m['total_direction_correct'] else '❌ 错误'}")
            
            out = pd.DataFrame({
                'date': actual_df['date'].values,
                'actual_close': actual_df['close'].values,
                'pred_close': pred_df['close'].values[:len(actual_df)],
            })
            out.to_csv(os.path.join(OUTPUT_DIR, f"{symbol}_{name}_result.csv"), index=False)
            
            results[symbol] = {"name": name, "last_price": round(last_price, 2), **m}
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            import traceback; traceback.print_exc()
            results[symbol] = {"name": name, "error": str(e)}
    
    print("\n" + "=" * 70)
    print("📋 汇总")
    print("=" * 70)
    print(f"{'股票':<8} {'MAPE':>8} {'方向准确率':>10} {'实际涨跌':>10} {'预测涨跌':>10} {'趋势':>6}")
    print("-" * 70)
    valid = {k: v for k, v in results.items() if "error" not in v}
    for s, r in valid.items():
        mark = "✅" if r["total_direction_correct"] else "❌"
        print(f"{r['name']:<6} {r['mape']:>7.2f}% {r['direction_accuracy']:>9.1f}% {r['actual_total_return_pct']:>+9.2f}% {r['pred_total_return_pct']:>+9.2f}% {mark:>4}")
    if valid:
        avg_mape = np.mean([r['mape'] for r in valid.values()])
        avg_dir = np.mean([r['direction_accuracy'] for r in valid.values()])
        n_dir = sum(1 for r in valid.values() if r['total_direction_correct'])
        print("-" * 70)
        print(f"{'平均':<6} {avg_mape:>7.2f}% {avg_dir:>9.1f}% {'':>10} {'':>10} {n_dir}/{len(valid):>4}")
    
    summary = {
        "report_date": REPORT_DATE,
        "lookback": LOOKBACK,
        "pred_len": PRED_LEN,
        "model": "Kronos-small",
        "device": device,
        "results": results
    }
    with open(os.path.join(OUTPUT_DIR, "kronos_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果: {OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
