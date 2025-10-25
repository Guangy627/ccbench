#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- 通用工具 ----------

def pick_series(df: pd.DataFrame, base: str, prefer_weighted: bool) -> pd.Series:
    """
    优先取 weighted 列（如 SDS_w），否则回退到非加权（SDS），再无则补 0.0。
    """
    wcol = f"{base}_w"
    if prefer_weighted and (wcol in df.columns):
        return pd.to_numeric(df[wcol], errors="coerce").fillna(0.0)
    if base in df.columns:
        return pd.to_numeric(df[base], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def compute_e_score(df: pd.DataFrame, prefer_weighted: bool) -> pd.DataFrame:
    """
    E_score = (SDS * (SCR + RD)) / (1 + TCR)
    这里的 SDS/TCR/SCR/RD 会优先取 *_w 列。
    """
    sdf = df.copy()

    # 选取列（加权优先）
    sds = pick_series(sdf, "SDS", prefer_weighted)
    tcr = pick_series(sdf, "TCR", prefer_weighted)
    scr = pick_series(sdf, "SCR", prefer_weighted)
    rd  = pick_series(sdf, "RD",  prefer_weighted)

    # 保存实际使用的“规范列”，方便后续统一处理与导出
    sdf["SDS_used"] = sds
    sdf["TCR_used"] = tcr.clip(lower=0.0)  # 防御性裁剪
    sdf["SCR_used"] = scr
    sdf["RD_used"]  = rd

    denom = 1.0 + sdf["TCR_used"].replace([np.inf, -np.inf], 0.0).clip(lower=0.0)
    denom = denom.replace(0.0, 1.0)
    sdf["E_score"] = (sdf["SDS_used"] * (sdf["SCR_used"] + sdf["RD_used"])) / denom

    # 记录到底用了哪些原始列，便于复现
    used_cols = {
        "SDS_source": "SDS_w" if (prefer_weighted and "SDS_w" in df.columns) else ("SDS" if "SDS" in df.columns else "NA"),
        "TCR_source": "TCR_w" if (prefer_weighted and "TCR_w" in df.columns) else ("TCR" if "TCR" in df.columns else "NA"),
        "SCR_source": "SCR_w" if (prefer_weighted and "SCR_w" in df.columns) else ("SCR" if "SCR" in df.columns else "NA"),
        "RD_source":  "RD_w"  if (prefer_weighted and "RD_w"  in df.columns) else ("RD"  if "RD"  in df.columns else "NA"),
    }
    for k, v in used_cols.items():
        sdf[k] = v

    return sdf


def summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["SDS_used", "TCR_used", "SCR_used", "RD_used", "E_score"]
    g = df.groupby("model_name")[keep].mean().sort_values("E_score", ascending=False)
    return g.round(4)


def fit_linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan")
    try:
        a, _ = np.polyfit(x, y, 1)
        return float(a)
    except Exception:
        return float("nan")


def compute_emergent_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df.groupby("model_name"):
        sub = sub.dropna(subset=["task_id", "E_score"]).copy()
        # 确保按 task_id 排序
        sub = sub.sort_values("task_id")
        x = pd.to_numeric(sub["task_id"], errors="coerce").astype(float).values
        y = pd.to_numeric(sub["E_score"], errors="coerce").astype(float).values
        slope = fit_linear_slope(x, y)
        rows.append({"model_name": model, "EGR_slope": slope})
    out = pd.DataFrame(rows).sort_values("EGR_slope", ascending=False)
    return out.round(6)


def plot_emergence(df: pd.DataFrame, out_png: Path, out_svg: Path):
    plt.figure(figsize=(8, 5), dpi=140)
    for model, sub in df.groupby("model_name"):
        sub = sub.copy().dropna(subset=["task_id", "E_score"]).sort_values("task_id")
        plt.plot(sub["task_id"].values, sub["E_score"].values, marker="o", label=str(model))
    plt.xlabel("Task ID")
    plt.ylabel("Emergent Reasoning Score (E_score)")
    plt.title("Reasoning Emergence by Task")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_svg)
    plt.close()


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="./out_parsed/ccbench_parsed_weighted.csv",  # 默认直接读加权结果
        help="输入指标表 CSV（建议使用 *_weighted.csv；需含 SDS/TCR/SCR/RD 或其 *_w 列）"
    )
    ap.add_argument("--outdir", default="out_charts", help="输出目录")
    ap.add_argument("--prefer-weighted", action="store_true", default=True,
                    help="优先使用 *_w 列进行计算（默认开启）")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "task_id" not in df.columns or "model_name" not in df.columns:
        raise ValueError("CSV 需要包含 'task_id' 和 'model_name' 列。")

    # 计算 E_score（优先加权）
    df2 = compute_e_score(df, prefer_weighted=args.prefer_weighted)

    # —— 保存逐样本 E_score（便于复查）——
    cols_export = [
        "id","task_id","model_name","task_category",
        # 实际使用的规范列
        "SDS_used","TCR_used","SCR_used","RD_used","E_score",
        # 源列记录（可帮助审稿人复现）
        "SDS_source","TCR_source","SCR_source","RD_source",
        # 你原有的过程指标，若存在就导出
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
    print(f"（列使用来源：SDS={df2['SDS_source'].iat[0]}, TCR={df2['TCR_source'].iat[0]}, SCR={df2['SCR_source'].iat[0]}, RD={df2['RD_source'].iat[0]}）")


if __name__ == "__main__":
    main()
