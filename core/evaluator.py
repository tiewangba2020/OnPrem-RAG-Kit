# -*- coding: utf-8 -*-
"""core.evaluator —— 自动化评测引擎：Naive RAG vs 本方案 Hybrid RAG

四维评测口径（全部为可复现的客观计算，不做人工打分）
----------------------------------------------------
1. **上下文召回率 Context Recall**：标准答案的关键事实（逐字）有多少出现在实际召回的上下文中。
   衡量"该找到的有没有找到"，直接反映漏召回。
2. **答案准确性 Answer Accuracy**：关键事实有多少出现在最终回答中。计算型题目按计算后的数值核对。
3. **表格解析准确率 Table Accuracy**：表格题中，包含答案的**完整表格行**是否原样出现在上下文中。
   固定长度切分会把表格行拦腰截断，这是 Naive RAG 在工程文档上的头号失分点。
4. **答案忠实度 Faithfulness（幻觉率反向指标）**：回答中出现的每一个"数字/单位/编码"类断言，
   是否能在召回的上下文中找到依据（或在题目给定/允许推导的范围内）。
   找不到依据即判定为该回答引入了无出处的事实断言 = 幻觉。

口径说明（诚实披露）
--------------------
* 召回率 / 表格准确率：**每次运行都真实执行检索**，实时计算。
* 答案准确性：取决于作答模式（``--answer-mode``）。

  - ``extractive``（默认）：两条流水线用**同一个**确定性抽取器，各自从自己召回的
    上下文中抽取答案，再与文档原文事实逐字比对。答案是真跑出来的，不是预置的，
    也不需要 LLM。它衡量的是"答案能不能从检索结果里摘出来"，是 LLM 作答准确率的
    **下界**——抽取器不会跨行推理，需要汇总多行才能回答的题会失分。
  - ``preset``：对题库中人工标注的高保真回答样例打分，衡量标注质量与打分器口径，
    **不代表系统生成能力**。
  - ``generated``：配置 ``.env`` 后调用 LLM 实时生成再打分。

* 忠实度：**抽取式模式下返回 None**。抽取式回答的每个 token 都来自上下文，按定义
  不可能出现无依据断言，两个流水线都会是 100%——用一个恒真指标给弱系统背书，
  比不报这个指标更糟。只有 preset / generated 模式下它才有区分度。
* 忠实度只检查「数字 + 单位 / 编码」类事实断言，纯文本层面的编造不在覆盖范围内。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .doc_processor import load_documents, load_questions
from .hybrid_retriever import (DEFAULT_TOP_K_CONTEXTS, AdvancedRAGPipeline, NaiveRAGPipeline,
                               build_pipelines, compose_extractive_answer, hard_tokens,
                               load_llm_config)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# 规模化基准题库（20 道人工题 + 10 道跨文档题 + 91 道表格行派生题）
BENCH_QUESTIONS = DATA_DIR / "eval_bench_questions.json"
# Tab1 沙盒演示用的 20 道刁钻题（带预置对照答案，用于展示"答非所问"）
DEMO_QUESTIONS = DATA_DIR / "eval_20_questions.json"

# 作答模式
#   preset     —— 用题库预置的专家回答样例（衡量标注质量，不是生成能力）
#   extractive —— 用同一个确定性抽取器从各自召回的上下文中抽答案（真实生成，可复现）
#   generated  —— 调用已配置的 LLM 实时生成
ANSWER_MODES = ("preset", "extractive", "generated")

METRIC_LABELS = {
    "context_recall": "上下文召回率",
    "answer_accuracy": "答案准确性",
    "faithfulness": "答案忠实度",
    "table_accuracy": "表格解析准确率",
    "retrieval_ms": "检索耗时(ms)",
}


# ---------------------------------------------------------------------------
# 打分原语
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """去空白 + 去 Markdown 强调标记 + 小写，用于中文/编码类事实的鲁棒匹配。

    必须剥离 **加粗** / `代码` / _斜体_ 标记，否则"浇筑后**5 个工作日内**归档"这类
    带强调的回答会被判定为未命中，造成系统性误判。
    """
    text = re.sub(r"[*`_~]", "", text or "")
    return re.sub(r"[\s　]+", "", text).lower()


def contexts_text(contexts: Sequence[Any]) -> str:
    return "\n".join(c.text for c in contexts)


def fact_coverage(facts: Sequence[str], text: str) -> Tuple[float, List[str]]:
    if not facts:
        return 1.0, []
    body = normalize(text)
    missing = [f for f in facts if normalize(f) not in body]
    return (len(facts) - len(missing)) / len(facts), missing


# --- 数值等价归一 ---------------------------------------------------------
# 同一笔金额可能写成 "252,000 元" / "252000元" / "25.2 万元"，字面不同但语义相同。
# 忠实度判定若只做字符串匹配，会把正确答案误判为幻觉，因此这里把「数值 + 单位」
# 统一换算到基准量纲后再比较。
_UNIT_CANON = {
    "元": ("money", 1.0),
    "万元": ("money", 1e4),
    "亿元": ("money", 1e8),
    "个月": ("month", 1.0), "月": ("month", 1.0),
    "年": ("year", 1.0),
    "个工作日": ("workday", 1.0), "工作日": ("workday", 1.0),
    "天": ("day", 1.0), "日": ("day", 1.0),
    "小时": ("hour", 1.0), "分钟": ("minute", 1.0), "秒": ("second", 1.0),
    "%": ("ratio", 0.01), "‰": ("ratio", 0.001),
}
_LEAD_NUM_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*(.*)$")


def canon_fact(token: str) -> Tuple:
    """把一个硬 token 归一成可比较的键。

    数值型 → ("num", 量纲, 基准值)；其余 → ("str", 归一化文本)。
    """
    t = normalize(token)
    m = _LEAD_NUM_RE.match(t)
    if not m:
        return ("str", t)
    try:
        value = float(m.group(1))
    except ValueError:
        return ("str", t)
    unit = m.group(2)
    if unit in _UNIT_CANON:
        family, mult = _UNIT_CANON[unit]
        return ("num", family, round(value * mult, 6))
    if not unit:
        return ("num", "bare", round(value, 6))
    return ("str", t)


def grounding_score(answer: str, context: str, question: str,
                    allow_computed: Sequence[str] = ()) -> Tuple[float, List[str]]:
    """忠实度 = 1 − 无依据断言占比。无依据 = 不在上下文、不在题干、不在允许推导清单中。

    判定顺序（由宽到严）：原文子串 → 上下文硬 token 归一值 → 题干 → 允许推导清单。
    允许清单用**归一值精确比对**，不做双向子串匹配——否则 "30 分钟" 会放行任何
    包含它的更长 token，形成漏判。
    """
    claims = hard_tokens(answer)
    if not claims:
        return 1.0, []
    ctx_n, q_n = normalize(context), normalize(question)
    ctx_keys = {canon_fact(t) for t in hard_tokens(context)}
    q_keys = {canon_fact(t) for t in hard_tokens(question)}
    allowed_keys = {canon_fact(a) for a in allow_computed if a}
    unsupported: List[str] = []
    for tok in claims:
        n = normalize(tok)
        key = canon_fact(tok)
        if n in ctx_n or key in ctx_keys:
            continue
        if n in q_n or key in q_keys:
            continue
        if key in allowed_keys:
            continue
        if n not in unsupported:
            unsupported.append(n)
    score = max(0.0, 1.0 - len(unsupported) / len(claims))
    return score, unsupported


def table_integrity(gold_row: str, context: str) -> Optional[float]:
    if not gold_row:
        return None
    return 1.0 if normalize(gold_row) in normalize(context) else 0.0


# ---------------------------------------------------------------------------
# 单题评测
# ---------------------------------------------------------------------------

def evaluate_question(q: Dict[str, Any], adv: AdvancedRAGPipeline,
                      naive: NaiveRAGPipeline,
                      answer_mode: str = "extractive",
                      score_generated: Optional[bool] = None) -> Dict[str, Any]:
    """评测单题。

    answer_mode：
      * ``preset``     —— 检索实时执行，回答侧对题库预置的专家样例打分。
      * ``extractive`` —— 检索实时执行，两条流水线用**同一个**确定性抽取器，各自从
        自己召回的上下文中抽取答案，再与标准事实逐字比对。这是真实生成的结果，
        不是预置答案，也不依赖 LLM。
      * ``generated``  —— 检索实时执行，回答侧调用已配置的 LLM 实时生成。

    忠实度为何在 extractive 模式下为 None
    --------------------------------------
    抽取式回答的每一个 token 都来自上下文，按定义不可能出现"无依据断言"，
    两个流水线都会是 100%——把 100% 当作结论展示，等于用一个恒真指标给弱系统
    背书。因此该模式下忠实度返回 None（界面显示"—"），只有 preset / generated
    两种模式下它才有区分度。
    """
    if score_generated is not None:                   # 向后兼容旧调用方式
        answer_mode = "generated" if score_generated else "preset"
    if answer_mode not in ANSWER_MODES:
        raise ValueError(f"未知作答模式 {answer_mode!r}，可选 {ANSWER_MODES}")

    doc_title = q.get("doc_title", "")
    allow = q.get("allow_computed", [])

    t0 = time.perf_counter()
    adv_ctx, adv_ret, adv_stages = adv.retrieve(q["question"], top_k_contexts=DEFAULT_TOP_K_CONTEXTS)
    adv_ms = (time.perf_counter() - t0) * 1000
    n_ctx, n_stages = naive.retrieve(q["question"])

    if answer_mode == "generated":
        adv_answer = adv.generate(q["question"], adv_ctx)
        n_answer = naive.generate(q["question"], n_ctx)
    elif answer_mode == "extractive":
        # 两条流水线共用同一个抽取器：对照组只变检索，不变作答，才是公平对照
        adv_answer = compose_extractive_answer(q["question"], adv_ctx)
        n_answer = compose_extractive_answer(q["question"], n_ctx)
    else:
        # preset：题目没有预置答案时（派生题）自动降级为抽取式，并在单题结果中标注
        adv_answer = q.get("advanced_answer", "") or compose_extractive_answer(q["question"], adv_ctx)
        n_answer = q.get("naive_answer", "") or compose_extractive_answer(q["question"], n_ctx)

    adv_body, naive_body = contexts_text(adv_ctx), contexts_text(n_ctx)
    adv_recall, adv_missing = fact_coverage(q.get("recall_facts", []), adv_body)
    n_recall, n_missing = fact_coverage(q.get("recall_facts", []), naive_body)

    adv_acc, adv_acc_missing = fact_coverage(q.get("answer_facts", []), adv_answer)
    n_acc, n_acc_missing = fact_coverage(q.get("answer_facts", []), n_answer)

    if answer_mode == "extractive":
        # 恒真指标不展示，避免用"100% 忠实"给弱系统背书
        adv_faith, adv_unsupported = None, []
        n_faith, n_unsupported = None, []
    else:
        adv_faith, adv_unsupported = grounding_score(adv_answer, adv_body, q["question"], allow)
        n_faith, n_unsupported = grounding_score(n_answer, naive_body, q["question"], allow)

    adv_table = table_integrity(q.get("gold_row", ""), adv_body)
    n_table = table_integrity(q.get("gold_row", ""), naive_body)

    return {
        "id": q["id"],
        "question": q["question"],
        "doc_title": doc_title,
        "dimension": q.get("dimension", ""),
        "answer_type": q.get("answer_type", "table"),
        "answer_source": answer_mode,
        "source": q.get("source", ""),
        "naive": {
            "context_recall": round(n_recall, 4),
            "answer_accuracy": round(n_acc, 4),
            "faithfulness": None if n_faith is None else round(n_faith, 4),
            "table_accuracy": None if n_table is None else round(n_table, 4),
            "retrieval_ms": round(n_stages.get("总计(ms)", 0.0), 2),
        },
        "advanced": {
            "context_recall": round(adv_recall, 4),
            "answer_accuracy": round(adv_acc, 4),
            "faithfulness": None if adv_faith is None else round(adv_faith, 4),
            "table_accuracy": None if adv_table is None else round(adv_table, 4),
            "retrieval_ms": round(adv_ms, 2),
        },
        "winner": "advanced" if adv_recall + adv_acc >= n_recall + n_acc else "naive",
        "naive_answer": n_answer,
        "advanced_answer": adv_answer,
        "gold_answer": q.get("gold_answer", ""),
        "naive_confidence": q.get("naive_confidence", 0.6),
        "advanced_confidence": q.get("advanced_confidence", 0.93),
        "business_value": q.get("business_value", ""),
        "diagnostics": {
            "naive_missing_recall": n_missing,
            "advanced_missing_recall": adv_missing,
            "naive_unsupported_claims": n_unsupported,
            "advanced_unsupported_claims": adv_unsupported,
            "naive_missing_answer_facts": n_acc_missing,
            "advanced_missing_answer_facts": adv_acc_missing,
            "advanced_hits": [
                {"rank": r.rank, "doc": r.chunk.doc_title, "type": r.chunk.chunk_type,
                 "heading": r.chunk.heading_path, "rerank": round(r.rerank, 4),
                 "bm25": round(r.bm25, 3), "vector": round(r.vector, 4),
                 "hard_hits": r.hard_hits, "preview": r.chunk.text[:120]}
                for r in adv_ret[:5]
            ],
            "naive_hits": [
                {"rank": c.index, "doc": c.title, "score": round(c.score, 4),
                 "preview": c.text[:120]} for c in n_ctx
            ],
            "advanced_stages": adv_stages,
        },
    }


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

def aggregate(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Optional[float]]:
    """按指标求均值。某指标在全部题目上都不适用（如抽取式模式下的忠实度）时返回 None，
    而不是 0.0——0.0 会被读成"完全失败"，与"不适用"是两件事。"""
    def mean(field: str, table_only: bool = False) -> Optional[float]:
        vals = [r[key][field] for r in rows
                if r[key].get(field) is not None and (not table_only or r["answer_type"] == "table")]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "context_recall": mean("context_recall"),
        "answer_accuracy": mean("answer_accuracy"),
        "faithfulness": mean("faithfulness"),
        "table_accuracy": mean("table_accuracy", table_only=True),
        "avg_retrieval_ms": mean("retrieval_ms"),
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else round(a - b, 4)


def _avg(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def by_source(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按题目来源拆分指标。

    规模化题库里 91/121 是「按行查表」的题，难度远低于人工设计的跨文档题；
    只看总平均会被简单题淹没，看不出架构差异真正体现在哪里。所以按来源拆开报。
    """
    labels = {
        "table-row-derived": "表格行派生（单点查表）",
        "cross-doc": "跨文档推理",
        "cross-section": "跨条款 / 跨表",
        "hand-crafted": "人工设计刁钻题",
    }
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(r.get("source") or "hand-crafted", []).append(r)
    out = []
    for src, items in buckets.items():
        out.append({
            "source": src,
            "label": labels.get(src, src),
            "count": len(items),
            "naive_context_recall": _avg(i["naive"]["context_recall"] for i in items),
            "advanced_context_recall": _avg(i["advanced"]["context_recall"] for i in items),
            "naive_answer_accuracy": _avg(i["naive"]["answer_accuracy"] for i in items),
            "advanced_answer_accuracy": _avg(i["advanced"]["answer_accuracy"] for i in items),
            "naive_table_accuracy": _avg(i["naive"]["table_accuracy"] for i in items),
            "advanced_table_accuracy": _avg(i["advanced"]["table_accuracy"] for i in items),
        })
    return sorted(out, key=lambda x: -x["count"])


def by_dimension(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(r["dimension"], []).append(r)
    out = []
    for dim, items in buckets.items():
        out.append({
            "dimension": dim,
            "count": len(items),
            "naive_context_recall": _avg(i["naive"]["context_recall"] for i in items),
            "advanced_context_recall": _avg(i["advanced"]["context_recall"] for i in items),
            "naive_answer_accuracy": _avg(i["naive"]["answer_accuracy"] for i in items),
            "advanced_answer_accuracy": _avg(i["advanced"]["answer_accuracy"] for i in items),
            "naive_faithfulness": _avg(i["naive"]["faithfulness"] for i in items),
            "advanced_faithfulness": _avg(i["advanced"]["faithfulness"] for i in items),
        })
    return sorted(out, key=lambda x: -x["count"])


# ---------------------------------------------------------------------------
# 全量基准
# ---------------------------------------------------------------------------

_PERF_DEFAULTS: Dict[str, Any] = {
    "data_source": "行业公开基准参考值（非本机实测）",
    "data_source_note": "本组数据依据 vLLM / Ollama 社区公开压测报告与同型号硬件的公开实测区间整理，"
                        "仅用于演示与容量规划参考；正式立项前须在目标硬件上按相同负载重新压测。",
}


def _existing_perf() -> Dict[str, Any]:
    """读取已有 perf 数据，并与声明字段做 merge。

    注意：这里必须 merge 而不是原样回写。历史上 perf 缺少 data_source 字段时，
    重跑评测会把"缺字段"的旧数据再存一遍，导致界面的数据来源声明永远不显示——
    形成不会自愈的自锁。
    """
    path = DATA_DIR / "benchmark_results.json"
    perf: Dict[str, Any] = {}
    if path.exists():
        try:
            perf = json.loads(path.read_text(encoding="utf-8")).get("perf", {}) or {}
        except Exception:
            perf = {}
    merged = {**_PERF_DEFAULTS, **perf}
    # data_source 若被写成空串同样视为缺失
    if not merged.get("data_source"):
        merged["data_source"] = _PERF_DEFAULTS["data_source"]
    return merged


def run_full_benchmark(questions_path: Optional[str | Path] = None,
                       docs_dir: Optional[str | Path] = None,
                       verbose: bool = False,
                       answer_mode: str = "extractive",
                       score_generated: Optional[bool] = None) -> Dict[str, Any]:
    if score_generated is not None:                   # 向后兼容
        answer_mode = "generated" if score_generated else "preset"
    questions_path = Path(questions_path) if questions_path else BENCH_QUESTIONS
    if not questions_path.exists():                   # 旧环境没生成过规模化题库时退回 20 题
        questions_path = DEMO_QUESTIONS
    docs_dir = Path(docs_dir) if docs_dir else DATA_DIR / "sample_docs"

    questions = load_questions(questions_path)
    docs = load_documents(docs_dir)
    cfg = load_llm_config()
    adv, naive, llm = build_pipelines(docs, config=cfg)

    # 未配置 LLM 时无法实时生成，降级到抽取式并在 meta 中如实标注
    if answer_mode == "generated" and cfg.is_demo:
        if verbose:
            print("[eval] 未检测到 LLM 配置，降级为抽取式作答（真实抽取，非预置答案）")
        answer_mode = "extractive"

    if verbose:
        print(f"[eval] 文档 {len(docs)} 篇 / 题目 {len(questions)} 道")
        print(f"[eval] 向量后端 {adv.retriever.vector_index.backend} · 嵌入 {adv.retriever.embedder.name}"
              f" · 重排 {adv.retriever.reranker.name} · LLM {cfg.label()}")
        print(f"[eval] 作答模式：{answer_mode}")

    rows = []
    t0 = time.perf_counter()
    for q in questions:
        row = evaluate_question(q, adv, naive, answer_mode=answer_mode)
        rows.append(row)
        if verbose:
            n, a = row["naive"], row["advanced"]
            print(f"  {row['id']} 召回 {n['context_recall']:.0%}→{a['context_recall']:.0%} | "
                  f"准确 {n['answer_accuracy']:.0%}→{a['answer_accuracy']:.0%} | "
                  f"表格 {(n['table_accuracy'] or 0):.0%}→{(a['table_accuracy'] or 0):.0%}")
            for tok in row["diagnostics"]["advanced_missing_recall"]:
                print(f"      · 本方案漏召回: {tok}")

    n_agg, a_agg = aggregate(rows, "naive"), aggregate(rows, "advanced")
    mode_label = {
        "extractive": "实测基准（检索实时执行 · 抽取式作答后评分）",
        "preset": "实测基准（检索实时执行 · 回答侧为预置样例评分）",
        "generated": "实测基准（检索实时执行 · 回答侧 LLM 实时生成后评分）",
    }[answer_mode]
    note = {
        "extractive": ("召回率与表格解析准确率为实时检索实测；答案准确性由两条流水线各自的检索结果"
                       "经同一个抽取器作答后与文档原文事实逐字比对得出，不依赖预置答案，也不依赖 LLM。"
                       "抽取式作答的每个 token 都来自上下文，因此忠实度指标在该模式下恒为 100%、"
                       "不具区分度，故留空（None）而不展示。"),
        "preset": ("召回率与表格解析准确率为实时检索实测；答案准确性/忠实度基于题库预置回答样例自动评分，"
                   "衡量的是标注质量与打分器口径，不代表生成能力。"),
        "generated": ("召回率与表格解析准确率为实时检索实测；答案准确性/忠实度对 LLM 实时生成的回答评分。"),
    }[answer_mode]
    result = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "measured",
            "mode_label": mode_label,
            "question_count": len(rows),
            "doc_count": len(docs),
            "questions_file": questions_path.name,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "vector_backend": adv.retriever.vector_index.backend,
            "embedder": adv.retriever.embedder.name,
            "reranker": adv.retriever.reranker.name,
            "llm": cfg.label(),
            "answer_side": {
                "extractive": "抽取式作答（两条流水线共用同一抽取器）",
                "preset": "题库预置高保真样例",
                "generated": "实时生成（LLM 已接入）",
            }[answer_mode],
            "answer_source": answer_mode,
            "faithfulness_available": answer_mode != "extractive",
            "top_k_children": adv.retriever.default_top_k_children,
            "note": note,
        },
        "aggregate": {
            "naive": n_agg, "advanced": a_agg,
            "delta": {k: _delta(a_agg[k], n_agg[k]) for k in n_agg},
        },
        "by_dimension": by_dimension(rows),
        "by_source": by_source(rows),
        "per_question": rows,
        "context_budget": context_budget_sensitivity(questions_path=questions_path, docs_dir=docs_dir),
        "perf": _existing_perf(),
    }
    if verbose:
        print("-" * 70)
        for key, label in (("context_recall", "上下文召回率"), ("answer_accuracy", "答案准确性"),
                           ("table_accuracy", "表格解析准确率"), ("faithfulness", "答案忠实度")):
            nv, av = n_agg[key], a_agg[key]
            if nv is None or av is None:
                print(f"{label}   —（该模式下不适用）")
            else:
                print(f"{label}   {nv:.1%}  →  {av:.1%}")
    return result


def context_budget_sensitivity(budgets: Sequence[int] = (3, 4, 5, 6, 8),
                               questions_path: Optional[str | Path] = None,
                               docs_dir: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """上下文预算的质量 / 成本曲线。

    这条曲线用来回答"为什么不用更大的上下文把分数做满"：预算越大召回越高，
    但每次请求喂给 LLM 的字符数（≈ token 成本）也同步上涨。把权衡摆出来，
    比单方面宣布一个最好看的数字更有说服力。

    忠实度只对**带预置答案**的题目计算：派生题没有预置回答，用空串去算会得到
    恒为 1.0 的假分数。若整批题都没有预置答案，该列返回 None。
    """
    questions_path = Path(questions_path) if questions_path else BENCH_QUESTIONS
    if not questions_path.exists():
        questions_path = DEMO_QUESTIONS
    questions = load_questions(questions_path)
    docs = load_documents(Path(docs_dir) if docs_dir else DATA_DIR / "sample_docs")
    adv, _, _ = build_pipelines(docs, config=load_llm_config())

    preset_qs = [q for q in questions if q.get("advanced_answer")]
    rows: List[Dict[str, Any]] = []
    for k in budgets:
        rec = 0.0
        chars = 0
        faith_sum, faith_n = 0.0, 0
        for q in questions:
            ctx, _, _ = adv.retrieve(q["question"], top_k_contexts=k)
            body = contexts_text(ctx)
            rec += fact_coverage(q.get("recall_facts", []), body)[0]
            chars += len(body)
            if q in preset_qs:
                faith_sum += grounding_score(q.get("advanced_answer", ""), body,
                                             q["question"], q.get("allow_computed", []))[0]
                faith_n += 1
        n = max(1, len(questions))
        rows.append({
            "budget": k,
            "context_recall": round(rec / n, 4),
            "faithfulness": round(faith_sum / faith_n, 4) if faith_n else None,
            "faithfulness_n": faith_n,
            "avg_context_chars": round(chars / n, 0),
            "avg_context_tokens": round(chars / n / 1.6, 0),
        })
    return rows


def save_benchmark(result: Dict[str, Any], path: Optional[str | Path] = None) -> Path:
    path = Path(path) if path else DATA_DIR / "benchmark_results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_benchmark(path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = Path(path) if path else DATA_DIR / "benchmark_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="OnPrem-RAG-Kit 自动化评测")
    ap.add_argument("--write", action="store_true", help="把评测结果写回 data/benchmark_results.json")
    ap.add_argument("--quiet", action="store_true", help="只输出聚合结果")
    ap.add_argument("--questions", default=None, help="题库路径，默认 data/eval_bench_questions.json")
    ap.add_argument("--answer-mode", choices=ANSWER_MODES, default="extractive",
                    help="作答模式：extractive（默认，真实抽取）/ preset / generated")
    ap.add_argument("--generated", action="store_true",
                    help="等价于 --answer-mode generated（需先在 .env 配置 LLM）")
    args = ap.parse_args()

    mode = "generated" if args.generated else args.answer_mode
    result = run_full_benchmark(questions_path=args.questions, verbose=not args.quiet,
                                answer_mode=mode)
    if args.write:
        path = save_benchmark(result)
        print(f"[ok] 基准已写入 {path}  (mode={result['meta']['mode']}, "
              f"answer_source={result['meta']['answer_source']}, "
              f"题目 {result['meta']['question_count']} 道)")
    if args.quiet:
        print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
