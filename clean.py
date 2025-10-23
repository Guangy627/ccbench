#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# ============ 配置 ============

MODELS = ("DeepSeek-V3.1-Terminus", "Claude-Sonnet-4")

# 关键词词表（可按需扩展）
PLAN_WORDS    = r"(plan|step|steps|roadmap|first|then|next|i will|we will|strategy|outline|pipeline|procedure)"
VERIFY_WORDS  = r"(check|verify|validate|test|assert|unit test|print|plot|visualize|compare|diff|sanity|inspect|metrics|evaluate)"
FIX_WORDS     = r"(fix|bug|error|exception|retry|patch|adjust|correct|repair|resolve|refactor|hotfix|rollback|re-run|rerun)"
REFLECT_WORDS = r"(wait|hmm|let's re|i made a mistake|rethink|double[- ]check|revisit|re-evaluate|consider|oops)"
CODE_HINTS    = r"(def |class |return |for |while |if |elif |import |from |try:|except |print\(|plt\.|pd\.|np\.|read_csv|groupby|agg|merge|join|hist|bar|line|scatter|corr|describe)"

# 自然语言类与工具类事件类型（含常见别名）
ASSISTANT_TYPES = {
    "message", "summary", "text", "analysis", "write", "edit",
    "assistant", "assistantmessage", "assistant_message"
}
TOOL_TYPES = {"bash","tool","tool_use","tool_call","tool_result","run","exec","command","shell"}

# 提取文本时优先尝试的键
TEXT_KEYS_PRIMARY = ["text", "content", "message", "summary", "thought", "analysis"]
TEXT_KEYS_TOOL    = ["stdout", "stderr", "output", "result", "details", "preview"]

# ============ 工具函数 ============

def parse_ts(ts: Any) -> Optional[datetime]:
    """解析 ISO8601 时间戳字符串为 datetime（容错 Z 结尾）。"""
    try:
        if ts is None or (isinstance(ts, float) and pd.isna(ts)):
            return None
        t = str(ts).replace("Z", "+00:00")
        return datetime.fromisoformat(t)
    except Exception:
        return None

def split_sentences(text: str) -> List[str]:
    sents = re.split(r"[。\.\!\?;\n]+", text or "")
    return [s.strip() for s in sents if s and s.strip()]

def density_ratio(text: str, pattern_words: str) -> float:
    sents = split_sentences(text)
    if not sents:
        return 0.0
    p = re.compile(pattern_words, re.IGNORECASE)
    hit = sum(1 for s in sents if p.search(s))
    return hit / len(sents)

def strategic_density_score(text: str) -> float:
    """SDS：命中 计划/验证/修复/代码 线索的句子占比。"""
    sents = split_sentences(text)
    if not sents:
        return 0.0
    p_any = re.compile("|".join([PLAN_WORDS, VERIFY_WORDS, FIX_WORDS, CODE_HINTS]), re.IGNORECASE)
    hit = sum(1 for s in sents if p_any.search(s))
    return hit / len(sents)

def tcr_semantic_compression(text: str) -> float:
    """
    TCR 近似：句子去重后 / 原句数（严格去重）。
    如需“语义去重”，可引入 TF-IDF + 余弦相似度阈值。
    """
    sents = split_sentences(text)
    if not sents:
        return 1.0
    uniq = list(dict.fromkeys(sents))
    return len(uniq) / len(sents)

# ---------- 角色/类型标准化 & 文本抽取 ----------

def normalize_role(role_raw: str, mtype_raw: str) -> str:
    r = (role_raw or "").strip().lower()
    t = (mtype_raw or "").strip().lower()

    # 当 role 缺失/无效，用 type 来推断
    if r in {"", "external", None}:
        if t in {"assistant", "summary"}:
            return "assistant"
        if t == "user":
            return "user"
        if t == "system":
            return "system"
        # 兜底
        return "assistant"

    # 有 role 时的常规映射
    if r in {"assistant","ai","bot","model","claude","glm","deepseek"}:
        return "assistant"
    if r in {"user","human","evaluator","tester"}:
        return "user"
    if r == "system":
        return "system"
    return r


def normalize_type(v: str) -> str:
    if not v:
        return "message"
    vv = str(v).lower()
    if vv in {"assistantmessage","assistant_message"}:
        return "assistantmessage"
    return vv

def extract_text_from_message(msg) -> str:
    """
    尽可能从 message 对象里提取可读文本：
    - 优先 TEXT_KEYS_PRIMARY
    - 次选 TEXT_KEYS_TOOL（tool_result/运行输出）
    - 对 list/dict 递归拼接
    """
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg

    if isinstance(msg, dict):
        # 1) 主文本键
        for k in TEXT_KEYS_PRIMARY:
            if k in msg and isinstance(msg[k], (str, int, float)):
                return str(msg[k])
        # 2) 嵌套主文本
        for k in TEXT_KEYS_PRIMARY:
            if k in msg and isinstance(msg[k], (dict, list)):
                t = extract_text_from_message(msg[k])
                if t:
                    return t
        # 3) 工具输出键
        for k in TEXT_KEYS_TOOL:
            if k in msg and isinstance(msg[k], (str, int, float)):
                return str(msg[k])
        # 4) 兜底：拼接所有可视字段
        parts = []
        for k, v in msg.items():
            if isinstance(v, (str, int, float)):
                parts.append(str(v))
            elif isinstance(v, (dict, list)):
                t = extract_text_from_message(v)
                if t:
                    parts.append(t)
        return "\n".join(parts)

    if isinstance(msg, list):
        parts = [extract_text_from_message(x) for x in msg]
        return "\n".join([p for p in parts if p])

    return str(msg)

# ============ 轨迹解析 ============
def parse_trajectory(traj_str: str):
    data = json.loads(traj_str)
    events = []
    for m in data:
        role_raw = m.get("userType") or m.get("role") or ""
        mtype_raw = m.get("type") or "mz1essage"
        role = normalize_role(role_raw, mtype_raw)   # 👈 用上 mtype_raw
        mtype = normalize_type(mtype_raw)
        ts = m.get("timestamp")
        msg_obj = m.get("message")
        content = extract_text_from_message(msg_obj)
        events.append({
            "role": role,
            "type": mtype,
            "timestamp": ts,
            "dt": parse_ts(ts),
            "content": (content or "").strip()
        })
    return events


def concat_assistant_text(events) -> str:
    """
    更宽松的自然语言拼接逻辑：
    - 优先选择 role='assistant' 且 type in ASSISTANT_TYPES 的 content
    - 若整个过程中没有捕获到自然语言，再退而求其次取 tool_result 的可读文本
    """
    texts = []
    for ev in events:
        t = ev["type"]
        if ev["role"] == "assistant" and (t in ASSISTANT_TYPES or t not in TOOL_TYPES):
            if ev["content"]:
                texts.append(ev["content"])

    if texts:
        return "\n\n".join(texts).strip()

    # 兜底：如果确实没有 assistant 文本，放入工具输出片段，避免空文件
    tool_texts = [ev["content"] for ev in events if ev["type"] in TOOL_TYPES and ev["content"]]
    return "\n\n".join(tool_texts[:3]).strip()  # 只取前几段，防止太长

def final_assistant_text(events) -> str:
    """
    最后一段“最终输出”的更鲁棒近似：
    - 找到最后一个 assistant 自然语言事件
    - 若不存在，退回到最后一个 tool_result 的可读文本
    """
    for ev in reversed(events):
        t = ev["type"]
        if ev["role"] == "assistant" and (t in ASSISTANT_TYPES or t not in TOOL_TYPES):
            if ev["content"]:
                return ev["content"]
    # 兜底
    for ev in reversed(events):
        if ev["type"] in TOOL_TYPES and ev["content"]:
            return ev["content"]
    return ""

def tool_stats_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    事件层面的工具统计：调用次数、失败次数、成功率（启发式），
    依据 type 与 content（如 'exit code 0' / '"success": true' / 'error' / 'failed' 等）
    """
    calls = 0
    failures = 0
    success = 0
    for ev in events:
        t = ev["type"]
        tl = t.lower()
        if tl in TOOL_TYPES:
            c = (ev["content"] or "").lower()
            if tl in {"bash","tool","tool_use","tool_call","run","exec","command","shell"}:
                calls += 1
            # 成功/失败启发式
            if ("exit code 0" in c) or ('"success": true' in c) or ("success" in c and "fail" not in c):
                success += 1
            if ("error" in c) or ("failed" in c) or re.search(r"exit code\s*[1-9]\d*", c):
                failures += 1
    success_rate = success / max(1, calls) if calls else 1.0
    return {"tool_calls_events": calls, "tool_fail_events": failures, "tool_success_rate_events": success_rate}

def conversation_duration_seconds(events: List[Dict[str, Any]]) -> float:
    times = [ev["dt"] for ev in events if ev["dt"] is not None]
    if len(times) >= 2:
        return (max(times) - min(times)).total_seconds()
    return float("nan")

# ========= 改进版：活跃时长计算 =========

def active_duration_stats(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算模型的活跃时长指标：
    - active_time_sec: 累加 assistant 发言段的时间差（排除 evaluator 等待）
    - avg_turn_time: 平均单轮耗时（active_time / assistant_messages）
    - active_ratio: 活跃占比 = active_time / 总持续时间
    """
    dts = [ev["dt"] for ev in events]
    roles = [ev["role"] for ev in events]
    if not any(dts):
        return {"active_time_sec": float("nan"),
                "avg_turn_time": float("nan"),
                "active_ratio": float("nan")}

    total_time = conversation_duration_seconds(events)
    active_time = 0.0
    last_time = None
    last_role = None

    for i in range(len(events) - 1):
        if dts[i] and dts[i + 1]:
            delta = (dts[i + 1] - dts[i]).total_seconds()
            if roles[i].lower().startswith("assistant"):
                active_time += max(0.0, delta)

    # 计算平均轮次耗时
    assistant_msgs = sum(1 for ev in events if ev["role"].lower().startswith("assistant"))
    avg_turn = active_time / max(1, assistant_msgs)

    # 活跃率
    ratio = active_time / total_time if total_time and total_time > 0 else float("nan")

    return {
        "active_time_sec": round(active_time, 3),
        "avg_turn_time": round(avg_turn, 3),
        "active_ratio": round(ratio, 3),
    }
# ============ 指标计算（含可选“时间加权”） ============

def time_weights_by_msg(events: List[Dict[str, Any]]) -> List[float]:
    """为每条事件估计停留时长（与下一条事件的时间差）；缺失时长用 1 近似。"""
    dts = [ev["dt"] for ev in events]
    weights = []
    for i in range(len(events)):
        if dts[i] and i+1 < len(events) and dts[i+1]:
            w = (dts[i+1] - dts[i]).total_seconds()
            weights.append(max(0.0, w))
        else:
            weights.append(1.0)
    return weights

def density_ratio_weighted(events: List[Dict[str, Any]], pattern_words: str) -> float:
    """时间加权的句子命中比例（仅 assistant 自然语言消息）。"""
    p = re.compile(pattern_words, re.IGNORECASE)
    weights = time_weights_by_msg(events)
    wt_hits, wt_total = 0.0, 0.0
    for ev, w in zip(events, weights):
        t = ev["type"]
        if ev["role"] == "assistant" and t in ASSISTANT_TYPES:
            sents = split_sentences(ev["content"])
            if not sents:
                continue
            hit = sum(1 for s in sents if p.search(s))
            wt_hits += w * hit
            wt_total += w * len(sents)
    return (wt_hits / wt_total) if wt_total else 0.0

def strategic_density_weighted(events: List[Dict[str, Any]]) -> float:
    """时间加权版 SDS。"""
    p_any = re.compile("|".join([PLAN_WORDS, VERIFY_WORDS, FIX_WORDS, CODE_HINTS]), re.IGNORECASE)
    weights = time_weights_by_msg(events)
    wt_hits, wt_total = 0.0, 0.0
    for ev, w in zip(events, weights):
        t = ev["type"]
        if ev["role"] == "assistant" and t in ASSISTANT_TYPES:
            sents = split_sentences(ev["content"])
            if not sents:
                continue
            hit = sum(1 for s in sents if p_any.search(s))
            wt_hits += w * hit
            wt_total += w * len(sents)
    return (wt_hits / wt_total) if wt_total else 0.0

# ============ 调试辅助（可选） ============

def debug_unique_roles_types(df: pd.DataFrame, n: int = 5):
    roles, types = set(), set()
    for s in df["trajectory"].head(n):
        try:
            for m in json.loads(s):
                roles.add(str(m.get("userType") or m.get("role") or "").lower())
                types.add(str(m.get("type") or "").lower())
        except Exception:
            pass
    print("unique roles:", roles)
    print("unique types:", types)

# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="Parse CC-Bench trajectories and compute GSB/SDS/TCR/Reflexivity + tool/time stats.")
    parser.add_argument("--csv", required=True, help="Input CSV, e.g., ccbench_data_analysis_deepseek_claude.csv")
    parser.add_argument("--outdir", default="out_parsed", help="Output directory")
    parser.add_argument("--weighted", action="store_true", help="Enable time-weighted metrics")
    parser.add_argument("--debug", action="store_true", help="Print unique roles/types for the first few rows")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    if args.debug:
        debug_unique_roles_types(df, n=10)

    parsed_rows = []
    for _, row in df.iterrows():
        try:
            events = parse_trajectory(row["trajectory"])
            atext = concat_assistant_text(events)
            afinal = final_assistant_text(events)
            dur_s = conversation_duration_seconds(events)
            tool_ev = tool_stats_from_events(events)
            active_stats = active_duration_stats(events)

            # 非加权指标
            SDS = strategic_density_score(atext)
            TCR = tcr_semantic_compression(atext)
            SCR = density_ratio(atext, FIX_WORDS + "|" + VERIFY_WORDS)
            RD  = density_ratio(atext, REFLECT_WORDS)
            first_chunk = atext.split("\n\n")[0] if atext else ""
            G   = min(1.0, 0.3 + 0.7 * density_ratio(first_chunk, PLAN_WORDS))
            S   = density_ratio(atext, PLAN_WORDS)
            B   = max(density_ratio(atext, VERIFY_WORDS), density_ratio(atext, FIX_WORDS))

            row_dict = {
                "id": row.get("id"),
                "task_id": row.get("task_id"),
                "model_name": row.get("model_name"),
                "task_category": row.get("task_category"),
                #tool
                "user_messages": row.get("user_messages"),
                "assistant_messages": row.get("assistant_messages"),
                "total_input_tokens": row.get("total_input_tokens"),
                "total_output_tokens": row.get("total_output_tokens"),
                "total_tokens": row.get("total_tokens"),
                "tool_calls_dataset": row.get("tool_calls"),
                "tool_failures_dataset": row.get("tool_failures"),
                "failure_rate_dataset": row.get("failure_rate"),
                #event 
                "conv_time_sec": round(dur_s, 3) if pd.notna(dur_s) else None,
                "active_time_sec": active_stats["active_time_sec"],
                "avg_turn_time": active_stats["avg_turn_time"],
                "active_ratio": active_stats["active_ratio"],
                **tool_ev,
                #metrics
                "G": round(G, 3), "S": round(S, 3), "B": round(B, 3),
                "SDS": round(SDS, 3), "TCR": round(TCR, 3),
                "SCR": round(SCR, 3), "RD": round(RD, 3),
                "assistant_text_len": len(atext),
                "final_excerpt": afinal[:400]
            }

            if args.weighted:
                SDS_w = strategic_density_weighted(events)
                SCR_w = density_ratio_weighted(events, FIX_WORDS + "|" + VERIFY_WORDS)
                RD_w  = density_ratio_weighted(events, REFLECT_WORDS)
                S_w   = density_ratio_weighted(events, PLAN_WORDS)
                B_w   = max(density_ratio_weighted(events, VERIFY_WORDS),
                            density_ratio_weighted(events, FIX_WORDS))
                row_dict.update({
                    "G_w": round(G, 3),  
                    "S_w": round(S_w, 3),
                    "B_w": round(B_w, 3),
                    "SDS_w": round(SDS_w, 3),
                    "SCR_w": round(SCR_w, 3),
                    "RD_w": round(RD_w, 3),
                })

            parsed_rows.append(row_dict)

        except Exception as e:
            parsed_rows.append({
                "id": row.get("id"),
                "task_id": row.get("task_id"),
                "model_name": row.get("model_name"),
                "task_category": row.get("task_category"),
                "error": str(e)
            })

    parsed_df = pd.DataFrame(parsed_rows)

    # —— 输出文件名根据 weighted 与否自动切换 ——
    out_csv = out_dir / ("ccbench_parsed_weighted.csv" if args.weighted else "ccbench_parsed.csv")
    parsed_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


    # 自动挑一个两模型同时存在的 task_id，并导出两模型的 assistant 全文与 final 片段
    counts = parsed_df.groupby(["task_id", "model_name"]).size().unstack(fill_value=0)
    if set(MODELS).issubset(set(counts.columns)):
        both_ids = counts[(counts[MODELS[0]] > 0) & (counts[MODELS[1]] > 0)].index.tolist()
    else:
        both_ids = []
    if both_ids:
        # chosen_task = both_ids[0]
        for chosen_task in both_ids:
            print(f"[INFO] 选择 task_id = {chosen_task}（两模型均存在）")
            for model in MODELS:
                raw_row = df[(df["task_id"] == chosen_task) & (df["model_name"] == model)].iloc[0]
                events = parse_trajectory(raw_row["trajectory"])
                atext = concat_assistant_text(events)
                afinal = final_assistant_text(events)
                (out_dir / f"{chosen_task}_{model}_assistant_text.txt").write_text(atext)
                (out_dir / f"{chosen_task}_{model}_final.txt").write_text(afinal)

    print(f"[DONE] 导出指标表：{out_csv}")
    print(f"[DONE] 输出目录：{out_dir.resolve()}")

if __name__ == "__main__":
    main()
