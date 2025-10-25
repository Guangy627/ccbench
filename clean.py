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

REF_MODEL = "Claude-Sonnet-4"
W_SDS = 0.35
W_SCR = 0.35
W_E   = 0.20
W_TCR = 0.10  # 从 ΔM 中减去

TAU_POS = 0.02   # Good 阈值：ΔM > TAU_POS
TAU_NEG = -0.02  # Bad 阈值：ΔM < TAU_NEG
TAU_SAME = 0.02  # Same 区间：|ΔM| ≤ TAU_SAME
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

# >>> 新增：E_score（与你报告里的一致）
def e_score(SDS: float, TCR: float, SCR: float, RD: float) -> float:
    """
    Emergent reasoning score:
    E = (SDS * (SCR + RD)) / (1 + TCR)
    """
    try:
        return float((SDS * (SCR + RD)) / (1.0 + max(TCR, 0.0)))
    except Exception:
        return 0.0

# >>> 新增：计算 ΔM
def compute_delta_m(row_a: pd.Series, row_b: pd.Series) -> float:
    """
    以 row_a 相对 row_b 的差异为 ΔM：
    ΔM = w1*(SDS_a - SDS_b) + w2*(SCR_a - SCR_b) + w3*(E_a - E_b) - w4*(TCR_a - TCR_b)
    """
    SDS_a, SDS_b = row_a.get("SDS", 0.0), row_b.get("SDS", 0.0)
    SCR_a, SCR_b = row_a.get("SCR", 0.0), row_b.get("SCR", 0.0)
    TCR_a, TCR_b = row_a.get("TCR", 1.0), row_b.get("TCR", 1.0)
    RD_a,  RD_b  = row_a.get("RD", 0.0),  row_b.get("RD", 0.0)

    E_a = e_score(SDS_a, TCR_a, SCR_a, RD_a)
    E_b = e_score(SDS_b, TCR_b, SCR_b, RD_b)

    delta = (W_SDS * (SDS_a - SDS_b)
             + W_SCR * (SCR_a - SCR_b)
             + W_E   * (E_a   - E_b)
             - W_TCR * (TCR_a - TCR_b))
    return float(delta)

# >>> 新增：对齐两模型并打 G/S/B 标签
def assign_gsb_labels(parsed_df: pd.DataFrame,
                      model_a: str,
                      model_b: str,
                      ref_model: str = REF_MODEL) -> pd.DataFrame:
    """
    对每个 task_id 内，找出两个模型各一条记录，计算 ΔM，并给双方都打标签：
      - 若 ref_model == model_b，则 ΔM 表示 model_a - model_b
      - Good  : ΔM > TAU_POS
        Same  : |ΔM| ≤ TAU_SAME
        Bad   : ΔM < TAU_NEG
    返回：在原 df 基础上增加 'E_score', 'delta_M', 'GSB' 列
    """
    df = parsed_df.copy()

    # 先算每行的 E_score
    df["E_score"] = df.apply(lambda r: e_score(r.get("SDS", 0.0),
                                              r.get("TCR", 1.0),
                                              r.get("SCR", 0.0),
                                              r.get("RD", 0.0)), axis=1)

    df["delta_M"] = pd.NA
    df["GSB"] = pd.NA

    # 仅对同一个 task_id 的 (model_a, model_b) 成对比较
    for tid, g in df.groupby("task_id"):
        ga = g[g["model_name"] == model_a]
        gb = g[g["model_name"] == model_b]
        if ga.empty or gb.empty:
            continue
        # 取各自第一条（或可按需要做更严格的选择）
        ra = ga.iloc[0]
        rb = gb.iloc[0]

        # 统一：ΔM 计算“模型A相对模型B”
        delta = compute_delta_m(ra, rb)

        # 给两个模型各自打标签；标签含义是“相对参照模型（ref_model）”
        if ref_model == model_b:
            # ΔM = A - B，故 A 的优劣即由 ΔM 决定；B 取相反
            gsb_a = ("Good" if delta > TAU_POS else
                     "Bad"  if delta < TAU_NEG else "Same")
            gsb_b = ("Bad"  if delta > TAU_POS else
                     "Good" if delta < TAU_NEG else "Same")
            df.loc[ra.name, "delta_M"] = round(delta, 6)
            df.loc[rb.name, "delta_M"] = round(-delta, 6)
            df.loc[ra.name, "GSB"] = gsb_a
            df.loc[rb.name, "GSB"] = gsb_b
        else:
            # 若你的参照不是 model_b，可按需要调整逻辑；此处保持“相对 ref_model”定义
            # 这里给出一个对称处理：谁是参照，就计算“本行 - 参照”的 ΔM
            if ra["model_name"] == ref_model:
                delta_a = compute_delta_m(ra, rb)  # ref - other
                gsb_a = ("Good" if delta_a > TAU_POS else
                         "Bad"  if delta_a < TAU_NEG else "Same")
                gsb_b = ("Bad"  if delta_a > TAU_POS else
                         "Good" if delta_a < TAU_NEG else "Same")
                df.loc[ra.name, "delta_M"] = round(delta_a, 6)
                df.loc[rb.name, "delta_M"] = round(-delta_a, 6)
                df.loc[ra.name, "GSB"] = gsb_a
                df.loc[rb.name, "GSB"] = gsb_b
            else:
                delta_b = compute_delta_m(rb, ra)  # ref - other
                gsb_b = ("Good" if delta_b > TAU_POS else
                         "Bad"  if delta_b < TAU_NEG else "Same")
                gsb_a = ("Bad"  if delta_b > TAU_POS else
                         "Good" if delta_b < TAU_NEG else "Same")
                df.loc[rb.name, "delta_M"] = round(delta_b, 6)
                df.loc[ra.name, "delta_M"] = round(-delta_b, 6)
                df.loc[rb.name, "GSB"] = gsb_b
                df.loc[ra.name, "GSB"] = gsb_a

    return df


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
            # first_chunk = atext.split("\n\n")[0] if atext else ""
            # G   = min(1.0, 0.3 + 0.7 * density_ratio(first_chunk, PLAN_WORDS))
            # S   = density_ratio(atext, PLAN_WORDS)
            # B   = max(density_ratio(atext, VERIFY_WORDS), density_ratio(atext, FIX_WORDS))
            first_chunk = atext.split("\n\n")[0] if atext else ""
            G_goal   = min(1.0, 0.3 + 0.7 * density_ratio(first_chunk, PLAN_WORDS))
            S_plan   = density_ratio(atext, PLAN_WORDS)
            B_verify = max(density_ratio(atext, VERIFY_WORDS), density_ratio(atext, FIX_WORDS))

            row_dict = {
                "id": row.get("id"),
                "task_id": row.get("task_id"),
                "model_name": row.get("model_name"),
                "task_category": row.get("task_category"),
                # tool
                "user_messages": row.get("user_messages"),
                "assistant_messages": row.get("assistant_messages"),
                "total_input_tokens": row.get("total_input_tokens"),
                "total_output_tokens": row.get("total_output_tokens"),
                "total_tokens": row.get("total_tokens"),
                "tool_calls_dataset": row.get("tool_calls"),
                "tool_failures_dataset": row.get("tool_failures"),
                "failure_rate_dataset": row.get("failure_rate"),
                # event 
                "conv_time_sec": round(dur_s, 3) if pd.notna(dur_s) else None,
                "active_time_sec": active_stats["active_time_sec"],
                "avg_turn_time": active_stats["avg_turn_time"],
                "active_ratio": active_stats["active_ratio"],
                **tool_ev,
                # metrics（行为）
                "G_goal": round(G_goal, 3),
                "S_plan": round(S_plan, 3),
                "B_verify": round(B_verify, 3),
                "SDS": round(SDS, 3),
                "TCR": round(TCR, 3),
                "SCR": round(SCR, 3),
                "RD": round(RD, 3),
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
                # >>> 替换/改名：与非加权保持一致的命名
                row_dict.update({
                    "G_goal_w": round(G_goal, 3),  # 仍沿用非加权 G_goal 的定义（如果需要也可做早期阶段加权）
                    "S_plan_w": round(S_w, 3),
                    "B_verify_w": round(B_w, 3),
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

    # —— 先写一版“行为指标”文件 ——（保留你现有逻辑）
    out_csv = out_dir / ("ccbench_parsed_weighted.csv" if args.weighted else "ccbench_parsed.csv")
    parsed_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # —— 基于行为指标打 G/S/B 标签（Good/Same/Bad）——
    # 与 MODELS 中的两模型配对，参照 REF_MODEL 进行相对评价
    if len(MODELS) == 2:
        labeled_df = assign_gsb_labels(parsed_df, model_a=MODELS[0], model_b=MODELS[1], ref_model=REF_MODEL)
        # 带标签的结果文件
        out_csv_gsb = out_dir / ("ccbench_parsed_with_gsb_weighted.csv" if args.weighted else "ccbench_parsed_with_gsb.csv")
        labeled_df.to_csv(out_csv_gsb, index=False, encoding="utf-8-sig")
        print(f"[DONE] 导出带 GSB 标签的指标表：{out_csv_gsb}")
    else:
        print("[WARN] MODELS 不是两元素对（无法成对打 GSB 标签）；已仅导出行为指标表。")
    print(f"[DONE] 输出目录：{out_dir.resolve()}")

if __name__ == "__main__":
    main()
