import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_e_score(df: pd.DataFrame) -> pd.DataFrame:
    # 需要的列，缺失则用 0（保守）
    cols = ["SDS", "TCR", "SCR", "RD"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    sdf = df.copy()

    # 防御性处理：把无穷/NA 清成 0
    sdf["SDS"] = pd.to_numeric(sdf["SDS"], errors="coerce").fillna(0.0)
    sdf["TCR"] = pd.to_numeric(sdf["TCR"], errors="coerce").fillna(0.0)
    sdf["SCR"] = pd.to_numeric(sdf["SCR"], errors="coerce").fillna(0.0)
    sdf["RD"]  = pd.to_numeric(sdf["RD"],  errors="coerce").fillna(0.0)

    # E_score = (SDS * (SCR + RD)) / (1 + TCR)
    denom = 1.0 + sdf["TCR"].clip(lower=0.0)  # 防止负值/0
    sdf["E_score"] = (sdf["SDS"] * (sdf["SCR"] + sdf["RD"])) / denom.replace(0, 1.0)

    return sdf


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["SDS", "TCR", "SCR", "RD", "E_score"]
    g = df.groupby("model_name")[keep].mean().sort_values("E_score", ascending=False)
    return g.round(4)


def fit_linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """最简单的线性趋势：y = ax + b 的 a。若点不足或方差为零返回 NaN。"""
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan")
    try:
        a, b = np.polyfit(x, y, 1)
        return float(a)
    except Exception:
        return float("nan")


def compute_emergent_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    每个模型的 EGR（Emergent Growth Rate）：E_score 对 task_id 的线性斜率。
    """
    rows = []
    for model, sub in df.groupby("model_name"):
        sub = sub.dropna(subset=["task_id", "E_score"])
        x = sub["task_id"].values.astype(float)
        y = sub["E_score"].values.astype(float)
        slope = fit_linear_slope(x, y)
        rows.append({"model_name": model, "EGR_slope": slope})
    out = pd.DataFrame(rows).sort_values("EGR_slope", ascending=False)
    return out.round(6)


def plot_emergence(df: pd.DataFrame, out_png: Path, out_svg: Path):
    """
    画 E_score vs task_id 折线图（按模型）。
    注意：不指定颜色/风格，满足通用要求。
    """
    plt.figure(figsize=(8, 5), dpi=140)

    # 为了连线更清晰，按 task_id 排序后再画
    for model, sub in df.groupby("model_name"):
        sub = sub.copy().sort_values("task_id")
        plt.plot(sub["task_id"].values, sub["E_score"].values, marker="o", label=str(model))

    plt.xlabel("Task ID")
    plt.ylabel("Emergent Reasoning Score (E_score)")
    plt.title("Reasoning Emergence by Task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_svg)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./out_parsed/ccbench_parsed.csv", help="输入指标表 CSV（含 SDS/TCR/SCR/RD 等列）")
    ap.add_argument("--outdir", default="out_charts", help="输出目录")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # 确保必要列存在
    if "task_id" not in df.columns or "model_name" not in df.columns:
        raise ValueError("CSV 需要包含 'task_id' 和 'model_name' 列。")

    # 计算 E_score
    df2 = compute_e_score(df)

    # —— 保存逐样本 E_score（便于复查）——
    cols_export = [
        "id","task_id","model_name","task_category",
        "SDS","TCR","SCR","RD","E_score",
        "conv_time_sec","active_time_sec","avg_turn_time","active_ratio",
        "total_input_tokens","total_output_tokens","total_tokens",
        "tool_calls_dataset","tool_failures_dataset","failure_rate_dataset",
    ]
    cols_export = [c for c in cols_export if c in df2.columns]
    per_task_path = out_dir / "per_task_scores.csv"
    df2[cols_export].to_csv(per_task_path, index=False, encoding="utf-8-sig")

    # —— 各模型均值汇总 —— 
    summary = summarize_by_model(df2)
    summary_path = out_dir / "summary_by_model.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    # —— Emergent Growth Rate（线性斜率）——
    egr = compute_emergent_growth_rate(df2)
    egr_path = out_dir / "emergence_slope.csv"
    egr.to_csv(egr_path, index=False, encoding="utf-8-sig")

    # —— 画趋势图并保存 —— 
    plot_emergence(
        df2.dropna(subset=["task_id", "E_score"]),
        out_png = out_dir / "emergence_trend.png",
        out_svg = out_dir / "emergence_trend.svg",
    )

    print("✅ 已生成文件：")
    print(f"- {summary_path}")
    print(f"- {per_task_path}")
    print(f"- {egr_path}")
    print(f"- {out_dir / 'emergence_trend.png'}")
    print(f"- {out_dir / 'emergence_trend.svg'}")


if __name__ == "__main__":
    main()