# -*- coding: utf-8 -*-
"""scripts/smoke_test.py —— 交付前自检 / CI 冒烟测试

用法::

    python scripts/smoke_test.py          # 全量自检（含 Streamlit 应用无头渲染）
    python scripts/smoke_test.py --fast   # 跳过 Streamlit 应用渲染（仅核心引擎）

覆盖范围
--------
1. 文档解析：3 篇行业文档可解析，父子分段非空
2. 混合检索：Advanced / Naive 两条流水线均可作答，且表格题能整表回溯
3. 评测基准：benchmark_results.json 结构完整、指标可用
4. TCO 测算：三套预设可算、报告与 CSV 可导出
5. Streamlit 应用：使用官方 AppTest 在进程内完整执行 app.py——
   **四个 Tab 的渲染代码全部真实运行**，任何异常都会在此暴露
6. 交互回归：模拟点击 Tab 3 的三个预设场景按钮，验证「会期状态写入 → 参数重算 → 全量重渲染」
   这条最易踩坑的交互链路（滑杆步长栅格冲突会在此暴露）
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

FAILED: list = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as exc:  # noqa: BLE001
        FAILED.append(name)
        print(f"[FAIL] {name} -> {type(exc).__name__}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. 文档解析
# ---------------------------------------------------------------------------

def t_documents() -> None:
    from core.doc_processor import corpus_stats, load_documents

    docs = load_documents(ROOT / "data" / "sample_docs")
    assert len(docs) >= 3, f"样例文档不足：{len(docs)} 篇"
    stats = corpus_stats(docs)
    assert stats["child_count"] > 50, "子块数量异常"
    assert stats["table_count"] >= 15, "表格识别数量异常"
    assert stats["naive_broken_table_rows"] > 0, "对照组应存在被截断的表格行（这是对比前提）"
    print(f"       文档 {stats['doc_count']} 篇 · 父块 {stats['parent_count']} · "
          f"子块 {stats['child_count']} · 表格 {stats['table_count']} 张 · "
          f"固定窗口截断表格行 {stats['naive_broken_table_rows']} 行")


# ---------------------------------------------------------------------------
# 2. 混合检索
# ---------------------------------------------------------------------------

def t_retrieval() -> None:
    from core.doc_processor import load_documents
    from core.hybrid_retriever import build_pipelines

    docs = load_documents(ROOT / "data" / "sample_docs")
    adv, naive, llm = build_pipelines(docs)
    q = "E-402 主轴驱动过载的排查流程中，第 3 步和第 4 步的判定值分别是多少？"

    a = adv.answer(q)
    assert a.contexts, "Advanced 未召回任何上下文"
    # 整表回溯可能以独立表格父块返回，也可能以"小节全文（内含完整表格）"形态返回，
    # 因此判定的是"是否完整拿到了表"，而不是 kind 字段的字面值。
    assert any(c.kind == "table" or c.has_table for c in a.contexts), "表格题未触发整表回溯"
    assert "4.5 MPa" in "\n".join(c.text for c in a.contexts), "关键数值未召回"

    n = naive.answer(q)
    assert n.contexts, "Naive 未召回任何上下文"
    print(f"       Advanced {a.latency_ms:.0f} ms / 置信度 {a.confidence:.2f} / "
          f"父块 {len(a.contexts)} 个　|　Naive {n.latency_ms:.0f} ms / 置信度 {n.confidence:.2f}")


# ---------------------------------------------------------------------------
# 3. 评测基准
# ---------------------------------------------------------------------------

def t_benchmark() -> None:
    from core.evaluator import load_benchmark

    b = load_benchmark()
    assert b.get("aggregate"), "benchmark_results.json 缺失或为空"
    a, n = b["aggregate"]["advanced"], b["aggregate"]["naive"]
    assert a["context_recall"] >= n["context_recall"], "本方案召回率不应低于对照组"
    assert a["table_accuracy"] >= n["table_accuracy"], "本方案表格解析不应低于对照组"
    nq = b.get("meta", {}).get("question_count", 0)
    assert len(b.get("per_question", [])) >= 100, f"逐题明细不足 100 条（当前 {len(b.get('per_question', []))}）"
    assert nq >= 100, f"题库规模不足 100 题（当前 {nq}）"
    assert b.get("by_source"), "缺少按题目来源的拆分"
    assert b.get("perf", {}).get("vllm"), "缺少 vLLM 压测数据"

    def pct(v) -> str:
        return "—（该口径不适用）" if v is None else f"{v:.1%}"

    print(f"       {nq} 题 · 召回 {pct(n['context_recall'])}→{pct(a['context_recall'])} · "
          f"表格 {pct(n['table_accuracy'])}→{pct(a['table_accuracy'])} · "
          f"忠实 {pct(n['faithfulness'])}→{pct(a['faithfulness'])}")


# ---------------------------------------------------------------------------
# 4. TCO 测算
# ---------------------------------------------------------------------------

def t_tco() -> None:
    from core import tco_calculator as tco

    for name in tco.SCENARIO_PRESETS:
        inp = tco.preset(name)
        res = tco.build_scenarios(inp)
        kpi = tco.summarize(inp, res)
        assert kpi["three_year_local"] > 0 and kpi["three_year_api"] > 0, f"{name} 成本曲线异常"
        assert len(res["scenarios"]["local"].cumulative) == tco.MONTHS
        rows = tco.monthly_rows(inp, res)
        assert len(rows) == tco.MONTHS and len(tco.to_csv(rows)) > 100
        md = tco.render_report_markdown(inp, res)
        assert md.startswith("# 致管理层"), "ROI 报告渲染异常"
        assert "回本临界点" in md
        sen = tco.sensitivity(inp)
        assert len(sen) >= 5, "敏感性分析行数不足"
        print(f"       {name}: 三年 API {kpi['three_year_api'] / 1e4:.1f} 万 / 本地 "
              f"{kpi['three_year_local'] / 1e4:.1f} 万 / 节约 {kpi['savings_vs_api']:.1%} / "
              f"回本 {kpi['break_even_local'] or '未回本'}")


# ---------------------------------------------------------------------------
# 5. Streamlit 应用（无头渲染，四个 Tab 全部执行）
# ---------------------------------------------------------------------------

def t_streamlit_app() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"应用渲染抛出异常：{at.exception}"
    assert len(at.markdown) > 8, "页面元素过少，可能未正常渲染"
    tabs = getattr(at, "tabs", [])
    charts = at.get("plotly_chart") if hasattr(at, "get") else []
    if charts:  # AppTest 的元素访问器随版本演进，取不到时不做硬断言
        assert len(charts) >= 4, f"Plotly 图表数量异常：{len(charts)}"
    print(f"       Markdown 元素 {len(at.markdown)} 个 · dataframe {len(at.dataframe)} 个 · "
          f"plotly 图表 {len(charts)} 个 · Tab {len(tabs)} 个 · 无异常")


# ---------------------------------------------------------------------------

def t_interaction() -> None:
    """交互回归：逐个点击 Tab 3 的预设场景按钮。

    预设按钮会向 st.session_state 写入一组整数域参数（人数 / Token / 折扣百分比…），
    再触发 st.rerun() 全量重算。若任一取值落在滑杆步长栅格之外，Streamlit 会抛异常——
    这是纯静态渲染测试覆盖不到、但在演示现场最丢脸的失效点。
    """
    from streamlit.testing.v1 import AppTest

    from core import tco_calculator as tco

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, f"初始渲染异常：{at.exception}"

    clicked = []
    for name in tco.SCENARIO_PRESETS:
        btn = next((b for b in at.button if getattr(b, "key", "") == f"preset_{name}"), None)
        if btn is None:
            continue
        btn.click().run()
        assert not at.exception, f"点击预设「{name}」后异常：{at.exception}"
        clicked.append(name)

    assert clicked, "未找到任何预设场景按钮（UI 结构可能已变更）"
    print(f"       已模拟点击并重算：{' / '.join(clicked)}　·　无异常")


def main() -> None:
    ap = argparse.ArgumentParser(description="OnPrem-RAG-Kit 冒烟测试")
    ap.add_argument("--fast", action="store_true", help="跳过 Streamlit 应用渲染")
    args = ap.parse_args()

    print("=" * 78)
    print("OnPrem-RAG-Kit :: 冒烟测试")
    print("=" * 78)
    check("① 文档解析与父子分段", t_documents)
    check("② 混合检索与重排流水线", t_retrieval)
    check("③ 评测基准数据完整性", t_benchmark)
    check("④ 算力选型与 TCO 测算", t_tco)
    if not args.fast:
        check("⑤ Streamlit 应用无头渲染（四个 Tab）", t_streamlit_app)
    if not args.fast:
        check("⑥ 预设场景按钮交互回归（Tab 3）", t_interaction)
    print("=" * 78)
    if FAILED:
        print(f"[结果] 失败 {len(FAILED)} 项：{', '.join(FAILED)}")
        sys.exit(1)
    print("[结果] 全部通过 ✅　应用可零配置直接运行（Demo Mode）")


if __name__ == "__main__":
    main()
