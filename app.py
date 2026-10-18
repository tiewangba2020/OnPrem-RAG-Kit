# -*- coding: utf-8 -*-
"""app.py —— OnPrem-RAG-Kit 交互式看板（Streamlit）

四大核心模块
------------
Tab 1  架构对比与交互式问答沙盒：同一问题，Naive RAG（固定窗口+纯向量）与服务级
       Hybrid RAG（父子分段 + BM25/向量混合 + RRF + 重排 + 整表回溯）同屏 PK
Tab 2  121 题规模化基准评测看板：召回率 / 答案准确性 / 表格解析准确率
       （题库 = 20 道人工刁钻题 + 10 道跨文档题 + 91 道由文档表格行机械派生的题）
Tab 3  企业私有化算力选型与 TCO / ROI 测算器（含 36 个月成本曲线与回本临界点）
Tab 4  vLLM vs Ollama 并发压测基准（TTFT / 吞吐量 / 端到端延迟）

设计原则
--------
**零配置可用**：没有任何 API Key、没有 GPU 时，检索、评测、成本测算全部真实运行，
只有"生成式作答"降级为抽取式作答器，并在界面上用徽标明确标注，绝不冒充 LLM 输出。
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.doc_processor import (corpus_stats, load_documents, relevant_table_rows,   # noqa: E402
                                table_row_status)
from core.evaluator import load_benchmark, normalize, run_full_benchmark, save_benchmark  # noqa: E402
from core.hybrid_retriever import (DEFAULT_TOP_K_CONTEXTS, build_pipelines, hard_tokens,  # noqa: E402
                                   load_llm_config)
from core import tco_calculator as tco                               # noqa: E402

DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "sample_docs"
# Tab1 沙盒演示用：20 道刁钻题，带预置对照答案（用于展示"答非所问"）
QUESTIONS_PATH = DATA_DIR / "eval_20_questions.json"
# Tab2 基准评测用：121 题规模化题库
BENCH_QUESTIONS_PATH = DATA_DIR / "eval_bench_questions.json"

st.set_page_config(
    page_title="OnPrem-RAG-Kit · 私有化 RAG 落地套件 · 检索优化与算力 TCO 评估",
    page_icon="🛰", layout="wide", initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# 主题：浅色企业风 —— 冷白作「场」，科技蓝作「信号」
# （图表 5 个序列色取自参考色板的浅色档位，已在 #F7F9FC 与 #FFFFFF 两个表面跑过
#   CVD 色盲安全校验：亮度带 / 色度下限 / 色盲分离度 / 常视分离度 全项通过；
#   青、黄、品红在浅底低于 3:1，按规则以直接数值标签 + 图例 + 明细表作为补救）
# ---------------------------------------------------------------------------

PAL = {
    "bg": "#F7F9FC", "surface": "#FFFFFF", "surface2": "#EEF4FC",
    "ink": "#0E1B2E", "ink2": "#41536B", "muted": "#5F7186",
    "s1": "#2a78d6",   # 主色：本方案 / 本地私有化（浅色档位）
    "s2": "#eb6834",   # 对照组：Naive RAG / 商用 API（浅色档位）
    "s3": "#1baf7a",   # 混合云（浅色档位）
    "s4": "#eda100", "s5": "#e87ba4",
    "good": "#0ca30c", "warn": "#B07A00", "serious": "#ec835a", "bad": "#d03b3b",
}
FONT = '"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,-apple-system,sans-serif'

CSS = """
<style>
/* ===========================================================================
   设计令牌 —— 冷白作「场」，科技蓝作「信号」
   信号色不填充任何区块，只出现在本方案读数 / 当前分段 / 引用角标上；
   语义色只用于表意：橙色=对照失败，蓝色=本方案，绿色=已核实。
   浅色底上的小字必须用加深档（绿 #006300、琥珀 #8A5A00、橙 #C24E20），
   状态原色 #0ca30c 在浅底只有 3.27:1，仅够做圆点与填充。
   =========================================================================== */
:root{
  --paper:#F7F9FC;   /* 页面平面 */
  --hull:#FFFFFF;    /* 分组表面（白色浮在冷白底上） */
  --deck:#EEF4FC;    /* 可交互 / 悬停表面 */
  --rule:rgba(16,42,80,.10);
  --rule-2:rgba(16,42,80,.20);
  --ink:#0E1B2E; --ink-2:#41536B; --ink-3:#5F7186;
  --sig:#2A78D6;     /* 科技蓝：本方案 */
  --sig-text:#1C62B4;
  --warn:#EB6834;    /* 对照橙：Naive RAG */
  --warn-text:#C24E20;
  --ok-text:#006300; --amber-text:#8A5A00;
  --r:10px;
}
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stHeaderActionElements"],
[data-testid="stAppDeployButton"]{display:none !important;}

.stApp{background:var(--paper);}
.block-container{padding:1.1rem 2.4rem 4.5rem 2.4rem; max-width:1560px;}
html, body, [class*="css"]{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",system-ui,sans-serif;}
/* 数字全部等宽对齐，让指标能按列扫读；替代原先的等宽字体（生成式页面的典型痕迹） */
*{font-variant-numeric:tabular-nums;}
:focus-visible{outline:2px solid var(--sig); outline-offset:2px; border-radius:4px;}
@media (prefers-reduced-motion: reduce){*{transition:none !important; animation:none !important;}}

/* ---------- 顶部导轨：品牌与实时状态同行 ---------- */
.rail{display:flex; align-items:flex-end; gap:20px; flex-wrap:wrap;
      padding:2px 2px 14px 2px; border-bottom:1px solid var(--rule-2); margin-bottom:24px;}
.rail-id{font-size:23px; font-weight:750; letter-spacing:-.4px; color:var(--ink); line-height:1.2;}
.rail-id span{display:block; font-size:13px; font-weight:500; color:var(--ink-2); margin-top:4px;}
.rail-meta{margin-left:auto; display:flex; align-items:center; gap:16px; flex-wrap:wrap;
      font-size:12px; color:var(--ink-3); padding-bottom:3px;}

/* ---------- 分段控件（原 Tab 标签）---------- */
.stTabs [data-baseweb="tab-list"]{gap:2px; border-bottom:1px solid var(--rule-2); padding:0; margin-bottom:20px;}
.stTabs [data-baseweb="tab"]{height:38px; padding:0 17px; border-radius:8px 8px 0 0; background:transparent;
     color:var(--ink-3); font-weight:560; font-size:13.5px; border-bottom:2px solid transparent;}
.stTabs [data-baseweb="tab"]:hover{color:var(--ink-2); background:rgba(42,120,214,.06);}
.stTabs [aria-selected="true"]{background:transparent; color:var(--ink) !important; border-bottom:2px solid var(--sig);}

/* ---------- 判决带：每个 Tab 只有一个主角数字 ---------- */
.verdict{padding:2px 2px 18px 2px; border-bottom:1px solid var(--rule); margin-bottom:26px;}
.verdict-label{font-size:12.5px; color:var(--ink-3);}
.verdict-line{display:flex; align-items:baseline; gap:15px; flex-wrap:wrap; margin-top:5px;}
.verdict-from{font-size:22px; font-weight:600; color:var(--ink-3);}
.verdict-arrow{font-size:16px; color:var(--ink-3);}
.verdict-to{font-size:40px; line-height:1.04; font-weight:750; color:var(--ink); letter-spacing:-1.2px;}
.verdict-delta{font-size:13.5px; font-weight:700; color:var(--ok-text); margin-left:4px;}
.verdict-bar{position:relative; height:3px; background:var(--rule); border-radius:2px;
             margin:15px 0 13px 0; max-width:700px;}
.verdict-bar .fill{position:absolute; left:0; top:0; bottom:0; background:var(--sig); border-radius:2px;}
.verdict-bar .tick{position:absolute; top:-4px; width:1px; height:11px; background:var(--ink-3);}
.verdict-sub{display:flex; gap:28px; flex-wrap:wrap; font-size:13px; color:var(--ink-2);}
.verdict-sub b{color:var(--ink); font-weight:700;}

/* ---------- 分节：发丝线 + 留白，不用描边盒 ---------- */
.sec{border-top:1px solid var(--rule-2); padding:18px 2px 8px 2px; margin:6px 0 4px 0;}
.sec-t{font-size:14px; font-weight:680; color:var(--ink); letter-spacing:-.1px;}
.sec-s{font-size:12.5px; color:var(--ink-3); margin-top:5px; max-width:78ch; line-height:1.65;}
.panel{background:var(--hull); border:1px solid var(--rule); border-radius:var(--r);
       padding:15px 18px; margin:0 0 14px 0;}
.panel-head{font-size:13.5px; font-weight:680; color:var(--ink); margin:0 0 7px 0;}
.muted{color:var(--ink-3); font-size:12px; line-height:1.6;}

/* ---------- 读数条：KPI 去掉盒子，改成带分隔线的仪表读数 ---------- */
.kpi-row{display:flex; flex-wrap:wrap; border-top:1px solid var(--rule); margin:6px 0 12px 0;}
.kpi{flex:1 1 148px; min-width:128px; padding:11px 16px 11px 0;}
.kpi + .kpi{border-left:1px solid var(--rule); padding-left:16px;}
.kpi-label{font-size:12px; color:var(--ink-3);}
.kpi-value{font-size:22px; font-weight:700; line-height:1.3; color:var(--ink); margin-top:2px;}
.kpi-value.blue{color:var(--sig-text);} .kpi-value.green{color:var(--ok-text);}
.kpi-value.amber{color:var(--amber-text);} .kpi-value.red{color:var(--warn-text);}
.kpi-sub{font-size:11.5px; color:var(--ink-3); margin-top:1px; line-height:1.45;}
@media (max-width:1100px){
  .kpi-row{border-top:0;} .kpi{flex:1 1 44%; border-top:1px solid var(--rule); padding:10px 0;}
  .kpi + .kpi{border-left:0; padding-left:0;}
}

/* ---------- 双栏对比：只在这里用盒子，且用 3px 语义左规线代替整圈彩框 ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:0 var(--r) var(--r) 0;
     border:1px solid var(--rule); border-left:3px solid var(--rule-2); background:var(--hull);}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.vs-fail){border-left-color:var(--warn);}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.vs-pass){border-left-color:var(--sig);}
.vs-fail, .vs-pass{display:block; height:0; margin:0; padding:0; overflow:hidden; font-size:0;}
.answer-hd{font-size:12px; font-weight:700; letter-spacing:.02em; padding:0 0 9px 0; margin:0 0 9px 0;
           border-bottom:1px solid var(--rule);}
.answer-hd.bad{color:var(--warn-text);} .answer-hd.good{color:var(--sig-text);}

/* ---------- 引文与证据 ---------- */
.cite{background:transparent; color:var(--sig-text); font-weight:700; font-size:10px;
      padding:0 2px; vertical-align:super;}
.src{border-left:2px solid var(--rule-2); padding:2px 0 2px 12px; margin:0 0 8px 0;
     font-size:12.5px; color:var(--ink-2); line-height:1.65;}
.src b{color:var(--sig-text); font-weight:700;}
.chip{display:inline-block; padding:1px 8px; border-radius:5px; font-size:11.5px; margin:2px 5px 2px 0;}
.chip.ok{background:#E4F5EC; color:var(--ok-text);}
.chip.no{background:#FDECEA; color:#B3261E; text-decoration:line-through;
         text-decoration-thickness:1px;}
.impact{border-left:3px solid #EDA100; background:#FEF8EC; padding:9px 14px; margin:10px 0 14px 0;
        border-radius:0 8px 8px 0; color:#6B4A00; font-size:12.5px; line-height:1.68; max-width:92ch;}

/* ---------- 控件 ---------- */
.badge{display:inline-block; padding:2px 9px; border-radius:6px; font-size:11px;
      font-weight:600; border:1px solid transparent; white-space:nowrap; letter-spacing:.02em;}
.b-blue{background:#E8F1FC; color:var(--sig-text);}
.b-green{background:#E4F5EC; color:var(--ok-text);}
.b-red{background:#FDEDE7; color:var(--warn-text);}
.b-amber{background:#FDF3E0; color:var(--amber-text);}
.b-grey{background:#EEF2F7; color:var(--ink-2);}
.stButton>button{border-radius:8px; border:1px solid #C9D6E6; background:var(--hull);
     color:var(--ink-2); font-weight:600; font-size:12.5px; padding:5px 12px;}
.stButton>button:hover{border-color:var(--sig); color:var(--ink); background:var(--deck);}
.stButton>button:focus:not(:active){border-color:var(--sig); color:var(--ink);}
.stDownloadButton>button{border-radius:8px; border:1px solid #9ED3B6; background:var(--hull);
     color:#0A6B3D; font-weight:600; font-size:12.5px;}
.stDownloadButton>button:hover{background:#EAF7F0; border-color:#0A7A3F;}
div[data-testid="stExpander"] details{border:1px solid var(--rule); border-radius:var(--r);
     background:var(--hull);}
div[data-testid="stExpander"] summary{font-size:12.5px; color:var(--ink-2);}
[data-baseweb="select"]>div, [data-baseweb="input"]{background:var(--hull) !important;
     border-color:#D8E2EE !important; border-radius:8px !important;}
[data-testid="stDataFrame"]{border:1px solid #E3EAF3; border-radius:var(--r); overflow:hidden;}
[data-testid="stCaptionContainer"] p{font-size:12px !important; color:var(--ink-3) !important;
     max-width:82ch; line-height:1.6;}
code{color:var(--sig-text); background:#EEF4FC; padding:1px 5px; border-radius:4px; font-size:.92em;}
hr{border:0; border-top:1px solid var(--rule); margin:28px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 通用渲染工具
# ---------------------------------------------------------------------------

def style_fig(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=PAL["ink2"]),
        margin=dict(l=6, r=14, t=46 if legend else 16, b=6),
        showlegend=legend, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11.5),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="rgba(42,120,214,.45)",
                        font=dict(family=FONT, size=12, color=PAL["ink"])),
    )
    fig.update_xaxes(gridcolor="rgba(16,42,80,.07)", zerolinecolor="rgba(16,42,80,.12)",
                     linecolor="rgba(16,42,80,.18)")
    fig.update_yaxes(gridcolor="rgba(16,42,80,.07)", zerolinecolor="rgba(16,42,80,.12)",
                     linecolor="rgba(16,42,80,.18)")
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def badge(text: str, tone: str = "blue") -> str:
    return f'<span class="badge b-{tone}">{text}</span>'


def kpi(label: str, value: str, sub: str = "", tone: str = "blue") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value {tone}">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')


def kpi_row(items) -> None:
    st.markdown('<div class="kpi-row">' + "".join(kpi(*i) for i in items) + "</div>",
                unsafe_allow_html=True)


CITE_RE = re.compile(r"【(\d+)】")


def with_cites(text: str) -> str:
    return CITE_RE.sub(r'<sup class="cite">\1</sup>', text or "")


def rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover
        st.experimental_rerun()


def broken_table_rows(text: str) -> int:
    """统计文本中被窗口切断的表格残行数量（Naive 固定窗口切分的典型损伤）。

    注意：这里只用于「逐片段打标」，页面上的汇总 KPI 统一走
    `core.doc_processor.table_row_status`，避免同一现象出现两套口径两个数。
    """
    bad = 0
    for ln in text.splitlines():
        s = ln.strip()
        if "|" in s and not re.fullmatch(r"\|.*\|", s):
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# 数据与引擎装载（自动补齐样例数据，保证 clone 即用）
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    if QUESTIONS_PATH.exists() and DOCS_DIR.exists():
        return
    with st.spinner("首次运行：正在生成行业样例文档与评测基准数据…"):
        spec = importlib.util.spec_from_file_location(
            "_gen", ROOT / "scripts" / "generate_sample_data.py")
        mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
        spec.loader.exec_module(mod)                          # type: ignore[union-attr]
        mod.generate_documents()
        mod.generate_questions()
        mod.generate_benchmark(try_measured=True)


@st.cache_resource(show_spinner=False)
def get_engine():
    docs = load_documents(DOCS_DIR)
    cfg = load_llm_config()
    adv, naive, llm = build_pipelines(docs, config=cfg)
    return docs, adv, naive, cfg


@st.cache_data(show_spinner=False)
def get_questions():
    if not QUESTIONS_PATH.exists():
        return []
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")).get("questions", [])


@st.cache_data(show_spinner=False)
def get_benchmark():
    data = load_benchmark()
    # 兜底文件可能只有压测数据（aggregate 为空 dict），此时必须真的跑一遍评测，
    # 不能把空指标当成"已评测"。
    if not data or not data.get("aggregate"):
        data = run_full_benchmark(verbose=False)
    return data


# ---------------------------------------------------------------------------
# Tab 1 · 同屏 PK 沙盒
# ---------------------------------------------------------------------------

def _fact_chips(facts, context: str) -> str:
    body = normalize(context)
    chips = []
    for f in facts:
        ok = normalize(f) in body
        chips.append(f'<span class="chip {"ok" if ok else "no"}">{f}</span>')
    return "".join(chips) if chips else '<span class="muted">—</span>'


def tab_sandbox(adv, naive, questions, docs=None) -> None:
    st.markdown('<div class="sec"><div class="sec-t">同一个问题，两种架构同屏对撞</div>'
                '<div class="sec-s">左侧是业界最常见的 Naive RAG（固定长度切分 + 纯向量 '
                'Top-2 召回），右侧是本方案的 Hybrid RAG。'
                '两条链路的检索每次都真实执行，不是预置结果。</div></div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.0, 2.2, 1.3])
    with c1:
        doc_sel = st.selectbox("① 文档范围", ["全部文档"] + sorted({q["doc_title"] for q in questions}),
                               key="sb_doc")
    pool = [q for q in questions if doc_sel == "全部文档" or q["doc_title"] == doc_sel]
    with c2:
        label = st.selectbox("② 预置刁钻问题（20 题）",
                             [f"{q['id']}｜{q['question']}" for q in pool], key="sb_q")
    with c3:
        custom = st.text_input("③ 或自定义提问", key="sb_custom",
                               placeholder="例如：E-603 复位角度基准是多少？")

    q = next((x for x in pool if label.startswith(x["id"] + "｜")), pool[0] if pool else {})
    if not q:
        return
    use_custom = bool(custom.strip())
    query = custom.strip() if use_custom else q["question"]

    adv_res = adv.answer(query, top_k_contexts=DEFAULT_TOP_K_CONTEXTS,
                         preset_answer=("" if use_custom else q.get("advanced_answer", "")))
    naive_res = naive.answer(query, preset_answer=("" if use_custom else q.get("naive_answer", "")))

    mode_html = badge("实时检索", "green") if not use_custom else badge("自定义提问 · 实时检索", "green")
    st.markdown(
        f'<div class="panel"><div style="font-size:15px;font-weight:700;color:var(--ink);line-height:1.7">'
        f'{query}</div><div style="margin-top:8px">{badge(q.get("doc_title", "跨文档"), "grey")}'
        f'{badge(q.get("dimension", "自定义"), "blue")}{mode_html}</div></div>',
        unsafe_allow_html=True)

    if q.get("business_value"):
        naive_min = q.get("naive_time_min")
        adv_min = q.get("advanced_time_min")
        if naive_min and adv_min:
            saved = max(0, int(naive_min) - int(adv_min))
            eff_line = (f'效率视角：传统人工翻手册 / 查条款约 {naive_min} 分钟 → '
                        f'本方案 {adv_min} 分钟内闭环，单次节省约 {saved} 分钟（题库标注基线）。')
        else:
            eff_line = "效率视角：该题未标注人工基线耗时。"
        st.markdown(f'<div class="impact">💼 业务价值：{q["business_value"]}<br>'
                    f'<span class="muted">{eff_line}</span></div>',
                    unsafe_allow_html=True)

    left, right = st.columns(2)

    # ---------------- 左：Naive RAG ----------------
    with left:
        st.markdown(badge("固定 320 字窗口切分", "red") + " " + badge("纯向量召回 Top-2", "red")
                    + " " + badge("无重排 · 无结构感知", "red"), unsafe_allow_html=True)
        # 表格完整性统一口径：与评测引擎共用 table_row_status，不再出现"页面 0 行 / README 24 行"
        rel_rows = relevant_table_rows(docs, hard_tokens(query)) if docs else []
        naive_body = "\n".join(c.text for c in naive_res.contexts)
        adv_body = "\n".join(c.text for c in adv_res.contexts)
        if rel_rows:
            n_st = table_row_status(rel_rows, naive_body)
            tbl_val = f"{n_st['full']} / {len(rel_rows)}"
            tbl_sub = f"残缺 {n_st['partial']} · 缺失 {n_st['missing']}（与问题相关的表格行）"
            tbl_tone = "red" if n_st["full"] < len(rel_rows) else "green"
        else:
            tbl_val = str(sum(broken_table_rows(c.text) for c in naive_res.contexts))
            tbl_sub, tbl_tone = "召回片段中被截断的表格残行", "red"
        kpi_row([
            ("检索耗时", f"{naive_res.latency_ms:.0f} ms", "单次检索链路总耗时", "neutral"),
            ("回答置信度", f"{naive_res.confidence:.2f}", "向量相似度归一化", "amber"),
            ("表格行完整", tbl_val, tbl_sub, tbl_tone),
        ])
        with st.container(border=True):
            st.markdown('<div class="vs-fail"></div><div class="answer-hd bad">Naive RAG 输出</div>',
                        unsafe_allow_html=True)
            if use_custom:
                st.caption("自定义提问：Demo 模式下为抽取式作答（仅截取召回原文，无生成）。")
            st.markdown(with_cites(naive_res.answer), unsafe_allow_html=True)
        with st.expander(f"查看召回的 {len(naive_res.contexts)} 个原始片段（固定窗口切分结果）"):
            for c in naive_res.contexts:
                frag = c.text.replace("\n", " ⏎ ")
                flag = " ⚠️ 表格被截断" if broken_table_rows(c.text) else ""
                st.markdown(f'<div class="src"><b>片段 {c.index}</b>《{c.title}》{flag}<br>'
                            f'<span class="muted">{frag[:520]}{"…" if len(frag) > 520 else ""}</span></div>',
                            unsafe_allow_html=True)
        if q.get("recall_facts"):
            st.markdown('<div class="muted" style="margin-top:6px">标准答案关键事实命中情况：</div>'
                        + _fact_chips(q["recall_facts"],
                                      "\n".join(c.text for c in naive_res.contexts)),
                        unsafe_allow_html=True)

    # ---------------- 右：Hybrid RAG ----------------
    with right:
        st.markdown(badge("父子分段 + 整表回溯", "green") + " " + badge("BM25 × 向量混合召回", "green")
                    + " " + badge("重排打分（离线可解释 / 生产可换 BGE）", "green")
                    + " " + badge("引用溯源 + 置信度", "green"),
                    unsafe_allow_html=True)
        hard_hits = sorted({h for c in adv_res.contexts for h in c.hard_hits})
        if rel_rows:
            a_st = table_row_status(rel_rows, adv_body)
            adv_tbl_val = f"{a_st['full']} / {len(rel_rows)}"
            adv_tbl_sub = f"残缺 {a_st['partial']} · 缺失 {a_st['missing']}（与问题相关的表格行）"
            adv_tbl_tone = "green" if a_st["full"] == len(rel_rows) else "amber"
        else:
            adv_tbl_val, adv_tbl_sub, adv_tbl_tone = "—", "本题不涉及表格", "neutral"
        kpi_row([
            ("检索耗时", f"{adv_res.latency_ms:.0f} ms", "召回+融合+重排全链路", "neutral"),
            ("回答置信度", f"{adv_res.confidence:.2f}", "重排分归一化", "neutral"),
            ("精确命中", f"{len(hard_hits)} 项", "故障码 / 数值 / 单位", "green"),
            ("表格行完整", adv_tbl_val, adv_tbl_sub, adv_tbl_tone),
        ])
        with st.container(border=True):
            st.markdown('<div class="vs-pass"></div>'
                        '<div class="answer-hd good">本方案 Hybrid RAG 输出</div>',
                        unsafe_allow_html=True)
            if use_custom:
                st.caption("自定义提问：Demo 模式下为抽取式作答；配置 .env 后自动切换为生成式。")
            st.markdown(with_cites(adv_res.answer), unsafe_allow_html=True)
        if adv_res.contexts:
            st.markdown('<div class="muted">引用来源（点击展开可查看完整父块上下文）：</div>',
                        unsafe_allow_html=True)
            for c in adv_res.contexts:
                if c.kind == "table":
                    kind = "完整表格（已整表回溯）"
                elif c.has_table:
                    kind = "小节全文（内含完整表格）"
                else:
                    kind = "正文段落"
                hits = f' · 命中 {", ".join(c.hard_hits)}' if c.hard_hits else ""
                st.markdown(f'<div class="src"><b>【{c.index}】</b>《{c.title}》 › {c.heading_path}<br>'
                            f'<span class="muted">{kind} · 相关度 {c.score:.3f}{hits}</span></div>',
                            unsafe_allow_html=True)
        with st.expander("查看检索链路明细（混合召回 → RRF 融合 → 重排打分）"):
            st.markdown(f'<span class="muted">阶段耗时：' +
                        " · ".join(f"{k} {v}" for k, v in adv_res.stages.items()) + "</span>",
                        unsafe_allow_html=True)
            rows = []
            for r in adv_res.retrieved[:6]:
                rows.append({
                    "排名": r.rank,
                    "来源文档": r.chunk.doc_title,
                    "块类型": "表格行" if r.chunk.chunk_type == "table_row" else "正文",
                    "章节": (r.chunk.heading_path or "—")[:46],
                    "BM25": round(r.bm25, 3),
                    "向量相似度": round(r.vector, 4),
                    "重排得分": round(r.rerank, 4),
                    "精确命中": ", ".join(r.hard_hits) or "—",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if q.get("recall_facts"):
            st.markdown('<div class="muted" style="margin-top:6px">标准答案关键事实命中情况：</div>'
                        + _fact_chips(q["recall_facts"],
                                      "\n".join(c.text for c in adv_res.contexts)),
                        unsafe_allow_html=True)

    if q.get("gold_answer"):
        with st.expander("📌 标准答案与检索证据（评测口径）"):
            st.markdown(f"**标准答案**\n\n{q['gold_answer']}")
            st.markdown("**逐字证据（用于客观评分，必须原样出现在召回上下文中）**")
            for ev in q.get("gold_evidence", []):
                st.code(ev, language="markdown")
            if q.get("gold_row"):
                st.markdown("**表格完整性校验行**：该整行必须完整出现在上下文中，"
                            "方判定表格解析成功。")
                st.code(q["gold_row"], language="markdown")


# ---------------------------------------------------------------------------
# Tab 2 · 121 题规模化基准评测看板
# ---------------------------------------------------------------------------

METRIC_ORDER = [
    ("context_recall", "上下文召回率", "Context Recall"),
    ("answer_accuracy", "答案准确性", "Answer Accuracy"),
    ("table_accuracy", "表格解析准确率", "Table Accuracy"),
    ("faithfulness", "答案忠实度", "Faithfulness"),
]


def _pct(v, dash: str = "—") -> str:
    """None 表示"该指标在当前口径下不适用"，与 0% 是两件事，不能混显示。"""
    return dash if v is None else f"{v:.1%}"


def tab_benchmark(bench) -> None:
    agg = bench.get("aggregate", {})
    n, a = agg.get("naive", {}), agg.get("advanced", {})
    if not n or not a:
        st.warning("尚未生成评测基准数据，请先运行 `python -m core.evaluator --write`。")
        return

    meta = bench.get("meta", {})
    nq = meta.get("question_count", 0)
    answer_src = meta.get("answer_source", "extractive")
    faith_ok = a.get("faithfulness") is not None
    extractive = answer_src == "extractive"

    st.markdown(
        f'<div class="sec"><div class="sec-t">{nq} 道题的真实基准：Naive RAG 对比本方案 Hybrid RAG</div>'
        f'<div class="sec-s">题库 <code>{meta.get("questions_file", "—")}</code>：'
        f'20 道人工设计的刁钻题 + 10 道跨文档 / 跨条款题 + 91 道由文档表格行机械派生的题'
        f'（标准答案逐字取自文档单元格原值，可逐字复核）。<br>'
        f'数据来源 <code>data/benchmark_results.json</code>。'
        f'<b>召回率与表格解析准确率为实时检索实测</b>；'
        f'答案准确性由两条流水线各自的检索结果经<b>同一个抽取器</b>作答后与原文事实逐字比对得出——'
        f'不是预置答案，也不依赖 LLM。</div></div>',
        unsafe_allow_html=True)

    tt_n, tt_a = n["table_accuracy"], a["table_accuracy"]
    st.markdown(
        f'<div class="verdict">'
        f'<div class="verdict-label">表格解析准确率 · 整行表格完整召回</div>'
        f'<div class="verdict-line"><span class="verdict-from">{tt_n:.1%}</span>'
        f'<span class="verdict-arrow">→</span><span class="verdict-to">{tt_a:.1%}</span>'
        f'<span class="verdict-delta">+{(tt_a - tt_n) * 100:.1f} pp</span></div>'
        f'<div class="verdict-bar"><span class="fill" style="width:{tt_a * 100:.0f}%"></span>'
        f'<span class="tick" style="left:{tt_n * 100:.0f}%"></span></div>'
        f'<div class="verdict-sub">'
        f'<span>上下文召回率 <b>{_pct(n["context_recall"])} → {_pct(a["context_recall"])}</b></span>'
        f'<span>答案准确性 <b>{_pct(n["answer_accuracy"])} → {_pct(a["answer_accuracy"])}</b></span>'
        + (f'<span>答案忠实度 <b>{_pct(n["faithfulness"])} → {_pct(a["faithfulness"])}</b></span>'
           if faith_ok else
           '<span>答案忠实度 <b>—（抽取式作答下恒真，不展示）</b></span>')
        + f'</div></div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns([1.05, 1])
    with cc1:
        shown = [m for m in METRIC_ORDER if n.get(m[0]) is not None and a.get(m[0]) is not None]
        labels = [m[1] for m in shown][::-1]
        nv = [n[m[0]] for m in shown][::-1]
        av = [a[m[0]] for m in shown][::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=labels, x=nv, orientation="h", name="Naive RAG（固定窗口 + 纯向量）",
                             marker=dict(color=PAL["s2"]), text=[f"{v:.0%}" for v in nv],
                             textposition="outside", textfont=dict(color=PAL["ink2"], size=12),
                             hovertemplate="%{y}<br>Naive RAG %{x:.1%}<extra></extra>"))
        fig.add_trace(go.Bar(y=labels, x=av, orientation="h", name="本方案 Hybrid RAG",
                             marker=dict(color=PAL["s1"]), text=[f"{v:.0%}" for v in av],
                             textposition="outside", textfont=dict(color=PAL["ink"], size=12),
                             hovertemplate="%{y}<br>Hybrid RAG %{x:.1%}<extra></extra>"))
        fig.update_layout(bargap=0.32, bargroupgap=0.14)
        fig.update_xaxes(range=[0, 1.16], tickformat=".0%")
        plot(style_fig(fig, height=320))
        st.caption(
            "核心指标对比：长条为本方案。表格解析准确率差距最大（固定窗口切分把表格行拦腰截断）。"
            + ("　忠实度未列出：抽取式作答的每个字都来自上下文，该指标恒为 100%、没有区分度，"
               "接入 LLM 后才会出现在这里。" if extractive else ""))

    with cc2:
        bydim = bench.get("by_dimension", [])
        if bydim:
            df = pd.DataFrame(bydim).sort_values("advanced_context_recall")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df["dimension"], x=df["naive_context_recall"], orientation="h",
                                 name="Naive RAG", marker=dict(color=PAL["s2"]),
                                 hovertemplate="%{y}<br>Naive %{x:.1%}<extra></extra>"))
            fig.add_trace(go.Bar(y=df["dimension"], x=df["advanced_context_recall"], orientation="h",
                                 name="Hybrid RAG（本方案）", marker=dict(color=PAL["s1"]),
                                 hovertemplate="%{y}<br>本方案 %{x:.1%}<extra></extra>"))
            fig.update_xaxes(range=[0, 1.12], tickformat=".0%")
            fig.update_layout(bargap=0.34, bargroupgap=0.14)
            plot(style_fig(fig, height=320))
            st.caption("按考察维度拆分的上下文召回率：精确定位 / 跨条款 / 例外否定类问题差距最明显。")

    bysrc = bench.get("by_source", [])
    if bysrc:
        st.markdown('<div class="panel-head">按题目来源拆分：简单查表题会淹没真实差距</div>',
                    unsafe_allow_html=True)
        st.caption("题库里 91/121 是「按行查表」的单点题，难度远低于人工设计的跨文档题。"
                   "只看总平均会被简单题拉高，看不出架构差异真正体现在哪里——所以拆开报。")
        sdf = pd.DataFrame(bysrc)
        st.dataframe(
            pd.DataFrame({
                "题目来源": sdf["label"],
                "题数": sdf["count"],
                "Naive 召回": sdf["naive_context_recall"],
                "本方案 召回": sdf["advanced_context_recall"],
                "Naive 准确": sdf["naive_answer_accuracy"],
                "本方案 准确": sdf["advanced_answer_accuracy"],
                "Naive 表格完整": sdf["naive_table_accuracy"],
                "本方案 表格完整": sdf["advanced_table_accuracy"],
            }),
            width="stretch", hide_index=True,
            column_config={
                "Naive 召回": st.column_config.NumberColumn("Naive 召回", format="percent"),
                "本方案 召回": st.column_config.NumberColumn("本方案 召回", format="percent"),
                "Naive 准确": st.column_config.NumberColumn("Naive 准确", format="percent"),
                "本方案 准确": st.column_config.NumberColumn("本方案 准确", format="percent"),
                "Naive 表格完整": st.column_config.NumberColumn("Naive 表格完整", format="percent"),
                "本方案 表格完整": st.column_config.NumberColumn("本方案 表格完整", format="percent"),
            })

    budget_rows = bench.get("context_budget", [])
    if budget_rows:
        st.markdown('<div class="panel-head">上下文预算 vs 质量：为什么不直接把预算拉满</div>',
                    unsafe_allow_html=True)
        bdf = pd.DataFrame(budget_rows)
        has_faith = "faithfulness" in bdf.columns and bdf["faithfulness"].notna().any()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bdf["budget"], y=bdf["context_recall"], mode="lines+markers",
                                 name="上下文召回率", line=dict(color=PAL["s1"], width=2.4),
                                 marker=dict(size=8),
                                 hovertemplate="预算 %{x} 块<br>召回 %{y:.1%}<extra></extra>"))
        if has_faith:
            fig.add_trace(go.Scatter(x=bdf["budget"], y=bdf["faithfulness"], mode="lines+markers",
                                     name="答案忠实度（仅带预置答案的题）",
                                     line=dict(color=PAL["s3"], width=2.4, dash="dot"),
                                     marker=dict(size=8),
                                     hovertemplate="预算 %{x} 块<br>忠实度 %{y:.1%}<extra></extra>"))
        for _, r in bdf.iterrows():
            base = r["context_recall"]
            if has_faith and pd.notna(r["faithfulness"]):
                base = min(base, r["faithfulness"])
            fig.add_annotation(x=r["budget"], y=base - 0.03,
                               text=f"{r['avg_context_tokens']:.0f} tok", showarrow=False,
                               font=dict(size=10, color=PAL["muted"]))
        fig.update_xaxes(title_text="每次请求取用的上下文块数", tickvals=list(bdf["budget"]))
        # y 轴范围要同时容纳两条曲线：只按召回率算会把忠实度线裁到视野外（实测踩过）
        lo = float(bdf["context_recall"].min())
        if has_faith:
            lo = min(lo, float(bdf["faithfulness"].min()))
        fig.update_yaxes(range=[max(0.0, lo - 0.05), 1.015], tickformat=".0%")
        plot(style_fig(fig, height=300))
        st.caption(
            "每点下方标注为该预算下平均喂给 LLM 的上下文 token 量。"
            + ("预算 3 就已接近满召回，预算 5 达到 100%；再往上只会让 token 成本线性上涨。"
               if float(bdf["context_recall"].iloc[0]) > 0.97 else
               "预算越大召回越高，但每次请求喂给 LLM 的 token 也同步上涨——这是权衡，不是缺陷。")
            + ("　忠实度曲线只覆盖题库中 20 道带预置答案的题，与主表口径不同，仅作趋势参考。"
               if has_faith else ""))

    rows = bench.get("per_question", [])
    if rows:
        st.markdown('<div class="panel-head">逐题召回率分布（121 题全部纳入，不做抽样）</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame(rows)

        # 121 题画成哑铃图会糊成一团，改用分布直方图：一眼能看出 Naive 摊在
        # 各个区间、而本方案堆在 100% 这一格。
        buckets = ["0%", "1–39%", "40–59%", "60–79%", "80–99%", "100%"]

        def bucket(v: float) -> str:
            if v >= 0.999:
                return "100%"
            if v >= 0.8:
                return "80–99%"
            if v >= 0.6:
                return "60–79%"
            if v >= 0.4:
                return "40–59%"
            return "1–39%" if v > 0 else "0%"

        n_cnt = df["naive"].apply(lambda x: bucket(x["context_recall"])).value_counts()
        a_cnt = df["advanced"].apply(lambda x: bucket(x["context_recall"])).value_counts()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=buckets, y=[int(n_cnt.get(b, 0)) for b in buckets],
                             name="Naive RAG", marker=dict(color=PAL["s2"]),
                             text=[int(n_cnt.get(b, 0)) for b in buckets], textposition="outside",
                             textfont=dict(color=PAL["ink2"], size=12),
                             hovertemplate="召回 %{x}<br>Naive %{y} 题<extra></extra>"))
        fig.add_trace(go.Bar(x=buckets, y=[int(a_cnt.get(b, 0)) for b in buckets],
                             name="Hybrid RAG（本方案）", marker=dict(color=PAL["s1"]),
                             text=[int(a_cnt.get(b, 0)) for b in buckets], textposition="outside",
                             textfont=dict(color=PAL["ink"], size=12),
                             hovertemplate="召回 %{x}<br>本方案 %{y} 题<extra></extra>"))
        fig.update_layout(bargap=0.34, bargroupgap=0.12)
        fig.update_xaxes(title_text="单题上下文召回率区间")
        fig.update_yaxes(title_text="题目数")
        plot(style_fig(fig, height=340))
        st.caption(f"共 {len(df)} 题。本方案有 "
                   f"{int((df['advanced'].apply(lambda x: x['context_recall'] >= 0.999)).sum())} 题达到满召回；"
                   f"Naive 有 "
                   f"{int((df['naive'].apply(lambda x: x['context_recall'] >= 0.999)).sum())} 题。"
                   "整行表格完整率（表格题）分别是 "
                   f"{df['naive'].apply(lambda x: x.get('table_accuracy') or 0).mean():.1%} 与 "
                   f"{df['advanced'].apply(lambda x: x.get('table_accuracy') or 0).mean():.1%}。")

        st.markdown('<div class="panel-head">逐题评测明细（可筛选）</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns([1, 1.4, 1])
        with f1:
            doc_f = st.multiselect("文档", sorted(df["doc_title"].unique()), key="bm_doc")
        with f2:
            dim_f = st.multiselect("考察维度", sorted(df["dimension"].unique()), key="bm_dim")
        with f3:
            verdict = st.radio("筛选", ["全部", "仅看本方案胜出", "仅看表格题"], horizontal=True, key="bm_v")

        view = df.copy()
        if doc_f:
            view = view[view["doc_title"].isin(doc_f)]
        if dim_f:
            view = view[view["dimension"].isin(dim_f)]
        if verdict == "仅看本方案胜出":
            view = view[view["winner"] == "advanced"]
        elif verdict == "仅看表格题":
            view = view[view["answer_type"] == "table"]

        table = pd.DataFrame({
            "题号": view["id"],
            "考察维度": view["dimension"],
            "问题": view["question"].str.slice(0, 46) + "…",
            "Naive 召回": view["naive"].apply(lambda x: x["context_recall"]),
            "本方案 召回": view["advanced"].apply(lambda x: x["context_recall"]),
            "Naive 准确": view["naive"].apply(lambda x: x["answer_accuracy"]),
            "本方案 准确": view["advanced"].apply(lambda x: x["answer_accuracy"]),
            "表格完整(N)": view["naive"].apply(lambda x: x.get("table_accuracy")),
            "表格完整(A)": view["advanced"].apply(lambda x: x.get("table_accuracy")),
        })
        cfg = {
            "Naive 召回": st.column_config.ProgressColumn("Naive 召回", format="%.0f%%", min_value=0, max_value=1),
            "本方案 召回": st.column_config.ProgressColumn("本方案 召回", format="%.0f%%", min_value=0, max_value=1),
            "Naive 准确": st.column_config.NumberColumn("Naive 准确", format="percent"),
            "本方案 准确": st.column_config.NumberColumn("本方案 准确", format="percent"),
            "表格完整(N)": st.column_config.NumberColumn("表格完整(N)", format="percent"),
            "表格完整(A)": st.column_config.NumberColumn("表格完整(A)", format="percent"),
            "问题": st.column_config.TextColumn("问题", width="large"),
        }
        if faith_ok:                                   # 抽取式模式下该列为空，不如不显示
            table["忠实度(N)"] = view["naive"].apply(lambda x: x["faithfulness"])
            table["忠实度(A)"] = view["advanced"].apply(lambda x: x["faithfulness"])
            cfg["忠实度(N)"] = st.column_config.NumberColumn("忠实度(N)", format="percent")
            cfg["忠实度(A)"] = st.column_config.NumberColumn("忠实度(A)", format="percent")
        st.dataframe(table, width="stretch", hide_index=True, height=430, column_config=cfg)

        with st.expander("🔍 单题下钻：失败样例 vs 本方案回答 vs 标准答案"):
            pick = st.selectbox("选择题号", [r["id"] for r in rows], key="bm_pick")
            row = next((r for r in rows if r["id"] == pick), None)
            if row:
                st.markdown(f"**问题**：{row['question']}")
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown('<div class="answer-hd bad">Naive RAG（典型失败输出）</div>',
                                unsafe_allow_html=True)
                    st.markdown(row.get("naive_answer", ""))
                with d2:
                    st.markdown('<div class="answer-hd good">本方案 Hybrid RAG</div>',
                                unsafe_allow_html=True)
                    st.markdown(with_cites(row.get("advanced_answer", "")), unsafe_allow_html=True)
                st.markdown("**标准答案**")
                st.markdown(row.get("gold_answer", "—"))
                diag = row.get("diagnostics", {})
                kpi_row([
                    ("Naive 漏召回事实", str(len(diag.get("naive_missing_recall", []))),
                     "、".join(diag.get("naive_missing_recall", []))[:52] or "无", "red"),
                    ("本方案 漏召回事实", str(len(diag.get("advanced_missing_recall", []))),
                     "、".join(diag.get("advanced_missing_recall", []))[:52] or "无", "green"),
                    ("Naive 答案漏答", str(len(diag.get("naive_missing_answer_facts", []))),
                     "、".join(diag.get("naive_missing_answer_facts", []))[:52] or "无", "red"),
                    ("本方案 答案漏答", str(len(diag.get("advanced_missing_answer_facts", []))),
                     "、".join(diag.get("advanced_missing_answer_facts", []))[:52] or "无", "green"),
                ])

    c1, c2 = st.columns([1, 1])
    with c1:
        llm_ready = not load_llm_config().is_demo
        label = ("🔁 重新运行实测评测（真实检索 + LLM 实时生成后评分）" if llm_ready
                 else f"🔁 重新运行实测评测（真实检索 + 抽取式作答，{meta.get('question_count', 0)} 题）")
        if st.button(label, width="stretch"):
            with st.spinner(f"正在对 {meta.get('question_count', 0)} 道题逐题执行检索与评分…"):
                # 配了 LLM 就用真实生成回答评分；没配则用抽取式作答（真实抽取，非预置答案）
                result = run_full_benchmark(
                    verbose=False, answer_mode="generated" if llm_ready else "extractive")
                save_benchmark(result)
            st.cache_data.clear()
            st.success("评测完成，已更新 data/benchmark_results.json")
            rerun()
    with c2:
        st.markdown(f'<div class="muted">基准生成时间：{meta.get("generated_at", "—")}　|　'
                    f'题目 {meta.get("question_count", "—")} 道　|　模式：{meta.get("mode_label", "—")}<br>'
                    f'向量后端：{meta.get("vector_backend", "—")}　|　嵌入：{meta.get("embedder", "—")}　|　'
                    f'重排：{meta.get("reranker", "—")}</div>', unsafe_allow_html=True)

    with st.expander("📐 评测口径与诚实披露（为什么这些数字可信）"):
        nq = meta.get("question_count", 0)
        q18 = next((r for r in bench.get("per_question", []) if r.get("id") == "Q18"), None)
        q18_recall = q18["advanced"]["context_recall"] if q18 else None
        hard = next((s for s in bench.get("by_source", []) if s["source"] == "hand-crafted"), None)
        deriv = next((s for s in bench.get("by_source", []) if s["source"] == "table-row-derived"), None)
        deriv_a = f"{deriv['advanced_answer_accuracy']:.1%}" if deriv else "接近满分"
        hard_a = f"{hard['advanced_answer_accuracy']:.1%}" if hard else "—"
        hard_n = f"{hard['naive_answer_accuracy']:.1%}" if hard else ""
        st.markdown(
            f"**1. 题库怎么来的：{nq} 道题，标准答案不是我手写的。**\n"
            f"　　· 20 道人工设计的刁钻题（跨文档、例外否定、数值计算）；\n"
            "　　· 10 道跨文档 / 跨条款题，证据逐字摘自原文（如两篇制度互相引用的第 1.3 条）；\n"
            "　　· 91 道**由文档表格行机械派生**的题：题目文本由「表头 + 行键」拼成，"
            "标准答案逐字取自单元格原值，`gold_row` 是该行原文整行。"
            "出题脚本 `scripts/generate_eval_set.py` 可复跑，任何第三方都能拿原文逐字复核。\n\n"
            "**2. 召回率 / 表格解析准确率 = 实时实测。** 每次打开页面、每次点「重新运行实测评测」，"
            f"都是真的把 {nq} 道题喂给两条流水线重新检索一遍，再逐字比对原文证据。没有硬编码分数。\n\n"
            "**3. 答案准确性 = 抽取式作答的真实结果，不是预置答案。**\n"
            "　　两条流水线用**同一个**确定性抽取器：各自从自己召回的上下文里挑出最相关的原文行"
            "拼成回答，再与标准事实逐字比对。检索不同 → 上下文不同 → 抽出来的答案不同，"
            "所以这个指标是跑出来的，不是写死的。\n"
            "　　**它是下界，不是上限**：抽取器一次只能摘 6 行、也不会跨行推理，"
            "需要汇总多行才能回答的题它会失分。接入 LLM 后同一套打分器即改为对生成回答评分"
            "（`python -m core.evaluator --answer-mode generated`）。\n\n"
            "**4. 为什么没有「答案忠实度」？**\n"
            "　　抽取式回答的每一个字都来自上下文，按定义不可能出现无依据断言——"
            "两条流水线都会是 100%。把一个恒真指标当结论展示，等于用满分给弱系统背书。"
            "所以该口径下忠实度直接留空（JSON 里是 `null`），只有接入 LLM 后它才重新出现。"
            "这不是删指标，是不拿假指标充数。\n\n"
            "**5. 简单题会淹没真实差距，所以按来源拆开报。**\n"
            f"　　91/121 是单点查表题，本方案在这类题上达 {deriv_a}，"
            f"但真正体现架构差异的是人工题：本方案 {hard_a}"
            + (f"，Naive {hard_n}" if hard_n else "")
            + "。只看总平均会得出「差距没那么大」的错误印象。\n\n"
            "**6. 本方案的召回缺口在哪？** "
            + ("当前全量召回率已达 100%。" if (a.get("context_recall") or 0) >= 0.999 else
               f"当前为 {a.get('context_recall', 0):.1%}。")
            + (f" 历史缺口曾集中在 Q18（跨文档汇编题，当前 {q18_recall:.0%}）："
               "当时的归因是「上下文窗口不够」，但把上下文数从 4 加到 32 也没改善——"
               "真因是**重排候选池**只有 8 条、被单篇文档垄断。"
               "本版已把候选池改为随语料规模自适应（`max(12, 3√n)`）并加入文档多样性配额。"
               if q18_recall is not None else "")
        )


# ---------------------------------------------------------------------------
# Tab 3 · TCO / ROI 测算器
# ---------------------------------------------------------------------------

REDUNDANCY_OPTS = {
    "无冗余（单机，成本最低）": "none",
    "冷备（备机不含 GPU，故障切换约 4 小时）": "cold",
    "双机热备（可用性 99.9%，硬件成本翻倍）": "hot",
}
REV_REDUNDANCY = {v: k for k, v in REDUNDANCY_OPTS.items()}


def _ensure_tco_state() -> None:
    p = tco.preset("平衡型（推荐）")
    defaults = {
        "tco_employees": int(p.employees), "tco_qpd": float(p.queries_per_user_day),
        "tco_tokens": int(p.tokens_per_query), "tco_doc_gb": int(p.doc_gb),
        "tco_input_ratio": int(p.input_ratio * 100), "tco_secrecy": p.secrecy_level,
        "tco_api_model": p.api_model, "tco_workdays": int(p.workdays_per_month),
        "tco_peak": int(p.peak_factor * 10), "tco_growth": int(p.usage_growth * 100),
        "tco_decline": int(p.api_price_decline * 100),
        "tco_redundancy": REV_REDUNDANCY.get(p.redundancy, "冷备（备机不含 GPU，故障切换约 4 小时）"),
        "tco_ops_fte": int(p.ops_fte * 10), "tco_hw_disc": int(p.hardware_discount * 100),
        "tco_overflow": int(p.overflow_ratio * 100), "tco_benefit_disc": int(p.benefit_discount * 100),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _apply_preset(name: str) -> None:
    p = tco.preset(name)
    ss = st.session_state
    ss["tco_employees"] = int(p.employees)
    ss["tco_qpd"] = float(p.queries_per_user_day)
    ss["tco_tokens"] = int(p.tokens_per_query)
    ss["tco_doc_gb"] = int(p.doc_gb)
    ss["tco_input_ratio"] = int(round(p.input_ratio * 100))
    ss["tco_secrecy"] = p.secrecy_level
    ss["tco_api_model"] = p.api_model
    ss["tco_workdays"] = int(p.workdays_per_month)
    ss["tco_peak"] = int(round(p.peak_factor * 10))
    ss["tco_growth"] = int(round(p.usage_growth * 100))
    ss["tco_decline"] = int(round(p.api_price_decline * 100))
    ss["tco_redundancy"] = REV_REDUNDANCY.get(p.redundancy, list(REDUNDANCY_OPTS)[1])
    ss["tco_ops_fte"] = int(round(p.ops_fte * 10))
    ss["tco_hw_disc"] = int(round(p.hardware_discount * 100))
    ss["tco_overflow"] = int(round(p.overflow_ratio * 100))
    ss["tco_benefit_disc"] = int(round(p.benefit_discount * 100))


def _collect_tco_inputs() -> tco.TCOInputs:
    ss = st.session_state
    raw = {
        "employees": ss["tco_employees"], "queries_per_user_day": ss["tco_qpd"],
        "tokens_per_query": ss["tco_tokens"], "doc_gb": float(ss["tco_doc_gb"]),
        "input_ratio": ss["tco_input_ratio"] / 100.0, "secrecy_level": ss["tco_secrecy"],
        "api_model": ss["tco_api_model"], "workdays_per_month": ss["tco_workdays"],
        "peak_factor": ss["tco_peak"] / 10.0, "usage_growth": ss["tco_growth"] / 100.0,
        "api_price_decline": ss["tco_decline"] / 100.0,
        "redundancy": REDUNDANCY_OPTS.get(ss["tco_redundancy"], "cold"),
        "ops_fte": ss["tco_ops_fte"] / 10.0, "hardware_discount": ss["tco_hw_disc"] / 100.0,
        "overflow_ratio": ss["tco_overflow"] / 100.0, "benefit_discount": ss["tco_benefit_disc"] / 100.0,
    }
    return tco.TCOInputs.from_dict({**tco.TCOInputs().to_dict(), **raw})


def _tco_curve_fig(inp, res, kpi) -> go.Figure:
    sc = res["scenarios"]
    months = list(range(1, tco.MONTHS + 1))
    fig = go.Figure()
    if sc["api"].available:
        fig.add_trace(go.Scatter(x=months, y=[v / 1e4 for v in sc["api"].cumulative], name="① 商用闭源 API（按量付费）",
                                 mode="lines", line=dict(color=PAL["s2"], width=2.4),
                                 hovertemplate="第 %{x} 个月<br>累计 %{y:.1f} 万元<extra></extra>"))
    if sc["hybrid"].available:
        fig.add_trace(go.Scatter(x=months, y=[v / 1e4 for v in sc["hybrid"].cumulative],
                                 name="③ 混合云（本地基线 + 云端弹性）", mode="lines",
                                 line=dict(color=PAL["s3"], width=2.4),
                                 hovertemplate="第 %{x} 个月<br>累计 %{y:.1f} 万元<extra></extra>"))
    fig.add_trace(go.Scatter(x=months, y=[v / 1e4 for v in sc["local"].cumulative],
                             name="② 本地私有化服务器（自建）", mode="lines",
                             line=dict(color=PAL["s1"], width=2.8),
                             hovertemplate="第 %{x} 个月<br>累计 %{y:.1f} 万元<extra></extra>"))
    be = kpi.get("break_even_local")
    if be and sc["api"].available:
        i = max(0, int(be) - 1)
        y = sc["local"].cumulative[i] / 1e4
        fig.add_vline(x=be, line=dict(color=PAL["warn"], width=1.6, dash="dash"))
        fig.add_trace(go.Scatter(x=[be], y=[y], mode="markers+text", showlegend=False,
                                 marker=dict(symbol="star", size=15, color=PAL["warn"],
                                             line=dict(color="#7a4d00", width=1)),
                                 text=[f"回本临界点<br>第 {be} 个月"], textposition="top right",
                                 textfont=dict(color=PAL["warn"], size=12),
                                 hovertemplate="回本临界点：第 %{x} 个月<extra></extra>"))
    fig.update_xaxes(title=dict(text="月份", font=dict(size=12)), dtick=3)
    fig.update_yaxes(title=dict(text="累计成本（万元）", font=dict(size=12)))
    return style_fig(fig, height=400)


def tab_tco() -> None:
    _ensure_tco_state()
    st.markdown('<div class="sec"><div class="sec-t">企业私有化算力选型与三年 TCO / ROI 测算器</div>'
                '<div class="sec-s">把「员工规模 / 调用量 / Token 长度 / 数据密级」映射为'
                '「硬件选型 + 部署架构 + 36 个月现金流 + 回本临界点」，'
                '所有参数可调，结论实时重算。</div></div>', unsafe_allow_html=True)

    pcols = st.columns([1, 1, 1, 2.2])
    for i, name in enumerate(tco.SCENARIO_PRESETS):
        with pcols[i]:
            if st.button(name, key=f"preset_{name}", width="stretch"):
                _apply_preset(name)
                rerun()
    with pcols[3]:
        st.markdown('<div class="muted" style="padding-top:8px">预设一键载入：轻量型用于验证'
                    '「规模不足时不该私有化」的反向结论；平衡型与规模化对应主流立项场景。</div>',
                    unsafe_allow_html=True)

    left, right = st.columns([1.0, 2.35])

    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-head">① 业务负载</div>', unsafe_allow_html=True)
            st.slider("覆盖员工规模（人）", 50, 5000, step=50, key="tco_employees")
            st.slider("人均日提问次数", 0.5, 15.0, step=0.5, key="tco_qpd")
            st.slider("单次问答平均 Token（含检索上下文）", 500, 8000, step=250, key="tco_tokens")
            st.slider("知识库文档规模（GB）", 10, 2000, step=10, key="tco_doc_gb")
            st.slider("输入 Token 占比（%）", 50, 90, step=1, key="tco_input_ratio")
        with st.container(border=True):
            st.markdown('<div class="panel-head">② 合规与云端基线</div>', unsafe_allow_html=True)
            st.selectbox("数据保密等级", tco.SECRECY_LEVELS, key="tco_secrecy")
            st.selectbox("云端对照模型", list(tco.API_MODELS),
                         format_func=lambda k: tco.API_MODELS[k]["name"], key="tco_api_model")
        with st.expander("③ 高级参数（工程 / 财务假设）"):
            st.slider("月工作日（天）", 15, 30, step=1, key="tco_workdays")
            st.slider("峰值系数（峰值 / 均值）", 10, 50, step=1, key="tco_peak",
                      help="×0.1 后生效，例如 25 表示峰值是均值的 2.5 倍")
            st.slider("业务量年增长（%）", 0, 50, step=1, key="tco_growth")
            st.slider("云端 API 单价年降幅（%）", 0, 30, step=1, key="tco_decline")
            st.selectbox("冗余策略", list(REDUNDANCY_OPTS), key="tco_redundancy")
            st.slider("运维人力投入（×0.1 FTE）", 0, 20, step=1, key="tco_ops_fte")
            st.slider("硬件采购折扣（%）", 70, 110, step=1, key="tco_hw_disc")
            st.slider("混合云云端溢出流量占比（%）", 5, 60, step=5, key="tco_overflow")
            st.slider("效率收益折现系数（%）", 10, 100, step=10, key="tco_benefit_disc",
                      help="效率节约属于影子价值，默认只按 50% 计入，避免高估 ROI")

    inp = _collect_tco_inputs()
    res = tco.build_scenarios(inp)
    kpi = tco.summarize(inp, res)
    wl, plan = res["workload"], res["plan"]
    eff = kpi["efficiency"]

    be = kpi["break_even_local"]
    savings = kpi["savings_vs_api"]
    saving_txt = f"{savings:.1%}" if savings >= 0 else f"高出 {abs(savings):.1%}"
    saving_tone = "green" if savings >= 0 else "amber"
    be_txt = f"第 {be} 个月" if be else "36 个月内未回本"

    with right:
        kpi_row([
            ("回本临界点", be_txt, "本地私有化累计成本低于云端 API 的月份", "green" if be else "amber"),
            ("三年综合成本节约", saving_txt, f"对比 {tco.API_MODELS[inp.api_model]['name']}", saving_tone),
            ("本地一次性投入", f"{kpi['local_capex'] / 1e4:.1f} 万", "硬件 + 存储 + 实施合规", "neutral"),
            ("本地月度运营", f"{kpi['local_monthly_opex'] / 1e4:.1f} 万", "电费/托管/运维/迭代/维保", "neutral"),
            ("单次问答成本", f"¥{kpi['tco_per_query_local']:.4f}", f"云端 ¥{kpi['tco_per_query_api']:.4f}", "neutral"),
        ])

        if kpi["compliance_blocked"]:
            st.warning("当前为「绝密」级：数据禁止出域，云端 API 与混合云方案不适用，"
                       "仅保留本地私有化方案。回本结论不适用，应直接按合规要求立项。")
        elif savings < 0:
            st.info(f"当前规模与模型档位下，**自建硬件在三年内并不比云端 API 便宜**（{saving_txt}）。"
                    f"这是测算器的常态结论之一——若属此类场景，立项依据应转为"
                    f"「数据不出域合规 / 峰值并发自主可控 / 模型微调与自主迭代」，"
                    f"而不是省钱。可拖动左侧滑块找到本项目的成本反转点。")

        st.markdown('<div class="sec"><div class="sec-t">硬件选型与部署架构建议书</div></div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f'<div style="font-size:14px;line-height:2">'
                f'{badge("推荐配置", "blue")} <b>{plan.gpu_name} × {plan.gpu_count}</b>　'
                f'（{plan.servers} 台服务器 · 显存合计 {plan.vram_total_gb:.0f} GB · '
                f'实测吞吐能力 {plan.tps_capacity:,.0f} tokens/s）<br>'
                f'{badge("推理模型", "blue")} <b>{plan.model_name}</b>　'
                f'{badge(plan.quant, "grey")}{badge("vLLM PagedAttention + Continuous Batching", "grey")}<br>'
                f'{badge("检索架构", "green")} BM25 精确召回 × 向量语义召回 → RRF 融合 → '
                f'重排 Rerank（离线领域打分器，生产可换 BGE-Reranker）→ 父子分段整表回溯<br>'
                f'{badge("服务端压力", "amber")} 平均 {wl.avg_qps:.3f} QPS · 峰值 {wl.peak_qps:.3f} QPS · '
                f'峰值输出吞吐需求 {wl.required_tps:,.0f} tokens/s（含 {inp.headroom:.1f}× 余量）'
                f'</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted" style="margin-top:8px">选型理由：{plan.rationale}</div>',
                        unsafe_allow_html=True)
            if plan.alternatives:
                alt = pd.DataFrame(plan.alternatives)
                alt.columns = ["备选 GPU", "卡数", "硬件成本(元)", "吞吐能力(tokens/s)", "定位"]
                st.dataframe(alt, width="stretch", hide_index=True)
            gpu_alt = tco.cloud_gpu_alternative(inp, plan)
            st.markdown(
                f'<div class="src"><b>参照对比 · 云上 GPU 实例租赁</b><br>'
                f'{gpu_alt["instance"]}：月租约 {gpu_alt["monthly"]:,.0f} 元，三年约 '
                f'{gpu_alt["three_year"] / 1e4:.1f} 万元；自建硬件投资约 {kpi["local_capex"] / 1e4:.1f} 万元，'
                f'较云租赁{"节省" if gpu_alt["saving_vs_cloud_gpu"] >= 0 else "高出"} '
                f'{abs(gpu_alt["saving_vs_cloud_gpu"]):.1%}（未计运营成本）。<br>'
                f'<span class="muted">{gpu_alt["note"]}</span></div>', unsafe_allow_html=True)

        plot(_tco_curve_fig(inp, res, kpi))
        st.caption("36 个月累计成本对比。虚线为回本临界点：此后本地私有化的累计成本将持续低于按量付费的云端 API。")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="panel-head">效率收益（影子价值，非现金）</div>', unsafe_allow_html=True)
            # 展示绝对净收益而不是"收益率"：人力单价假设会把 ROI 推到四位数，
            # 那种数字只会让评审失去判断力，不构成决策依据。
            kpi_row([
                ("每月释放工时", f"{eff['monthly_hours_saved']:,.0f} 人·小时",
                 f"混合节省口径 {eff['blended_save_min']:.1f} 分钟/次"
                 + ("（已按人均每日检索工时封顶）" if eff["capped_by_capacity"] else ""), "green"),
                ("折现后月价值", f"{eff['monthly_net'] / 1e4:.1f} 万元",
                 f"已按 {inp.benefit_discount:.0%} 价值实现率折算", "green"),
                ("三年净收益", f"{kpi['three_year_net_benefit'] / 1e4:,.0f} 万元",
                 f"含 {eff['adoption_ramp_months']} 个月采纳爬坡 · 口径高度敏感", "amber"),
            ])
            st.caption("效率收益属影子价值，**不计入上方基础设施 TCO 对比**。该数字完全由「人均节省时长 × "
                       "价值实现率」两项假设决定，与架构选型无关；如需作为立项依据，请先做小范围 A/B 实测替换默认值。")
        with c2:
            st.markdown('<div class="panel-head">敏感性分析</div>', unsafe_allow_html=True)
            srows = tco.sensitivity(inp)
            sdf = pd.DataFrame({
                "云端模型": [r["model_name"] for r in srows],
                "API 月成本(元)": [r["api_monthly"] for r in srows],
                "三年节约率": [r["saving_ratio"] for r in srows],
                "回本临界点": [f"第 {r['break_even_month']} 个月" if r["break_even_month"] else "未回本"
                               for r in srows],
            })
            st.dataframe(sdf, width="stretch", hide_index=True, height=230,
                         column_config={"三年节约率": st.column_config.NumberColumn(
                             "三年节约率", format="percent")})
            st.caption("换用不同档位的云端模型，私有化的回本周期差异可达数倍——这正是立项评审时最该盯住的变量。")

        with st.expander("📋 36 个月现金流明细表（可导出 CSV）"):
            rows = tco.monthly_rows(inp, res)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=300)

        st.markdown('<div class="panel-head">一键导出立项材料</div>', unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "⬇️ 下载《致管理层：AI 私有化立项 ROI 评估报告》(Markdown)",
                data=tco.render_report_markdown(inp, res).encode("utf-8"),
                file_name="致管理层_AI私有化立项ROI评估报告.md", mime="text/markdown",
                width="stretch")
        with e2:
            st.download_button(
                "⬇️ 下载 36 个月现金流明细 (CSV，Excel 可直接打开)",
                data=tco.to_csv(tco.monthly_rows(inp, res)).encode("utf-8-sig"),
                file_name="TCO_36个月现金流明细.csv", mime="text/csv", width="stretch")

        with st.expander("⚖️ 模型边界与假设披露"):
            st.markdown(
                "* 本地私有化的成本**不止硬件**：实施费（知识库工程、系统集成、安全加固、培训）、"
                "机房托管、电费（PUE 1.45）、运维人力、模型迭代与硬件维保全部计入。"
                "只算硬件会得出过于乐观的结论。\n"
                f"* 云端 API 单价按年下降（当前假设 {inp.api_price_decline:.0%}/年），"
                f"业务量按年增长（{inp.usage_growth:.0%}/年）——两个因素对自建并不天然有利。\n"
                "* 效率收益属于影子价值，默认只按折现系数计入，且**不混入**基础设施成本对比。\n"
                "* 所有价格为公开市场参考价，建议以厂商正式报价替换后提交立项。\n"
                f"* 当前密级「{inp.secrecy_level}」的合规建议：{kpi['secrecy_note']}"
            )


# ---------------------------------------------------------------------------
# Tab 4 · 并发压测基准
# ---------------------------------------------------------------------------

def tab_perf(bench) -> None:
    perf = bench.get("perf") or {}
    v, o = perf.get("vllm"), perf.get("ollama")
    if not v or not o:
        st.warning("尚未找到压测数据（data/benchmark_results.json → perf）。"
                   "请先运行 `python scripts/generate_sample_data.py`。")
        return

    st.markdown('<div class="sec"><div class="sec-t">vLLM 与 Ollama 在并发压力下的推理表现</div>'
                f'<div class="sec-s">参考环境：{perf.get("reference_hardware") or perf.get("hardware", "—")}<br>'
                f'模型：{perf.get("model", "—")}<br>负载：{perf.get("workload", "—")}<br>'
                f'{perf.get("vllm_version", "")}　对比　{perf.get("ollama_version", "")}</div></div>',
                unsafe_allow_html=True)
    # 免责声明不能依赖数据文件里某个字段是否完整——字段缺失时默认按"非本机实测"展示，
    # 只有显式标注为实测数据才收起横幅。
    source = perf.get("data_source") or "行业公开基准参考值（非本机实测）"
    if "非本机实测" in source:
        st.markdown(
            f'<div class="impact">数据来源声明：本组曲线为 <b>{source}</b>，'
            f'依据 vLLM / Ollama 社区公开压测报告与同型号硬件的公开实测区间整理，用于演示与容量规划参考。'
            f'<span class="muted">正式立项前请在目标硬件上按相同负载重新压测后替换 '
            f'<code>data/benchmark_results.json</code> 中的 <code>perf</code> 字段。</span></div>',
            unsafe_allow_html=True)

    dfv, dfo = pd.DataFrame(v), pd.DataFrame(o)
    c50 = dfv[dfv["concurrency"] == 50].iloc[0]
    o50 = dfo[dfo["concurrency"] == 50].iloc[0]
    kpi_row([
        ("并发 50 吞吐量", f"{c50['throughput_tps'] / o50['throughput_tps']:.1f}×",
         f"vLLM {c50['throughput_tps']:,.0f} vs Ollama {o50['throughput_tps']:,.0f} tokens/s", "green"),
        ("并发 50 首字延迟", f"1 / {o50['ttft_ms'] / c50['ttft_ms']:.1f}",
         f"vLLM {c50['ttft_ms'] / 1000:.2f}s vs Ollama {o50['ttft_ms'] / 1000:.1f}s", "neutral"),
        ("Ollama 吞吐平台期", f"{o50['throughput_tps']:,.0f} tokens/s",
         "5 并发后基本不再增长（llama.cpp 无 Continuous Batching）", "amber"),
        ("vLLM 吞吐线性度", f"{c50['throughput_tps'] / dfv.iloc[0]['throughput_tps']:.0f}×",
         "相比单并发，20 并发 → 50 并发仍在上升", "neutral"),
    ])

    x = [str(c) for c in dfv["concurrency"]]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=dfv["ttft_ms"], name="vLLM · TTFT 均值", mode="lines+markers",
                                 line=dict(color=PAL["s1"], width=2.4), marker=dict(size=9),
                                 hovertemplate="并发 %{x}<br>TTFT %{y:.0f} ms<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=dfv["ttft_p95_ms"], name="vLLM · TTFT P95", mode="lines",
                                 line=dict(color=PAL["s1"], width=1.6, dash="dot"),
                                 hovertemplate="并发 %{x}<br>P95 %{y:.0f} ms<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=dfo["ttft_ms"], name="Ollama · TTFT 均值", mode="lines+markers",
                                 line=dict(color=PAL["s2"], width=2.4), marker=dict(size=9),
                                 hovertemplate="并发 %{x}<br>TTFT %{y:.0f} ms<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=dfo["ttft_p95_ms"], name="Ollama · TTFT P95", mode="lines",
                                 line=dict(color=PAL["s2"], width=1.6, dash="dot"),
                                 hovertemplate="并发 %{x}<br>P95 %{y:.0f} ms<extra></extra>"))
        fig.update_xaxes(title=dict(text="并发请求数", font=dict(size=12)))
        fig.update_yaxes(title=dict(text="首字延迟 TTFT（ms，对数轴）", font=dict(size=12)),
                         type="log", tickvals=[100, 300, 1000, 3000, 10000, 30000])
        plot(style_fig(fig, height=380))
        st.caption("首字延迟：Ollama 在 5 并发后因请求排队出现数量级劣化；vLLM 依靠 Continuous Batching 把 TTFT 压在 1 秒内。")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=dfv["throughput_tps"], name="vLLM（PagedAttention + 连续批处理）",
                                 mode="lines+markers", line=dict(color=PAL["s1"], width=2.6),
                                 marker=dict(size=10), fill="tozeroy",
                                 fillcolor="rgba(42,120,214,.10)",
                                 hovertemplate="并发 %{x}<br>吞吐 %{y:,.0f} tokens/s<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=dfo["throughput_tps"], name="Ollama（llama.cpp 后端）",
                                 mode="lines+markers", line=dict(color=PAL["s2"], width=2.6),
                                 marker=dict(size=10), fill="tozeroy",
                                 fillcolor="rgba(235,104,52,.10)",
                                 hovertemplate="并发 %{x}<br>吞吐 %{y:,.0f} tokens/s<extra></extra>"))
        fig.update_xaxes(title=dict(text="并发请求数", font=dict(size=12)))
        fig.update_yaxes(title=dict(text="系统总吞吐量（tokens/s）", font=dict(size=12)))
        plot(style_fig(fig, height=380))
        st.caption("吞吐量：vLLM 随并发近似线性扩展，Ollama 在 5 并发后进入约 165 tokens/s 的平台期。")

    c3, c4 = st.columns([1.3, 1])
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[v2 / 1000 for v2 in dfv["e2e_latency_ms"]], name="vLLM 端到端",
                                 mode="lines+markers", line=dict(color=PAL["s1"], width=2.4),
                                 marker=dict(size=9),
                                 hovertemplate="并发 %{x}<br>端到端 %{y:.1f} s<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=[v2 / 1000 for v2 in dfo["e2e_latency_ms"]], name="Ollama 端到端",
                                 mode="lines+markers", line=dict(color=PAL["s2"], width=2.4),
                                 marker=dict(size=9),
                                 hovertemplate="并发 %{x}<br>端到端 %{y:.1f} s<extra></extra>"))
        fig.update_xaxes(title=dict(text="并发请求数", font=dict(size=12)))
        fig.update_yaxes(title=dict(text="单请求端到端延迟（秒，对数轴）", font=dict(size=12)), type="log")
        plot(style_fig(fig, height=350))
    with c4:
        st.markdown('<div class="panel-head">原始压测数据</div>', unsafe_allow_html=True)
        tbl = pd.DataFrame({
            "并发": dfv["concurrency"],
            "vLLM TTFT(ms)": dfv["ttft_ms"], "Ollama TTFT(ms)": dfo["ttft_ms"],
            "vLLM 吞吐": dfv["throughput_tps"].round(0), "Ollama 吞吐": dfo["throughput_tps"].round(0),
            "Ollama 成功率": dfo["success_rate"],
        })
        st.dataframe(tbl, width="stretch", hide_index=True, height=300)

    st.markdown(f'<div class="impact">结论：{perf.get("conclusion", "")}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">工程建议：<b>本地开发与单用户验证用 Ollama 足够</b>（部署简单、'
        '无需显存规划）；但一旦进入多部门并发（工单系统、OA 嵌入、多部门同时使用），'
        '必须切换到 vLLM —— 否则并发 10 以上就会出现数十秒的排队首字延迟，'
        '用户体验直接崩坏，且 GPU 利用率长期低于 15%。</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    _bootstrap()
    docs, adv, naive, cfg = get_engine()
    questions = get_questions()
    bench = get_benchmark()
    stats = corpus_stats(docs)

    demo = cfg.is_demo
    n_bench = (bench.get("meta", {}) or {}).get("question_count") or 121
    mode_badge = badge("Demo Mode · 高保真演示", "amber") if demo else badge(f"已接入 {cfg.label()}", "green")
    st.markdown(
        f'<div class="rail">'
        f'<div class="rail-id">OnPrem-RAG-Kit'
        f'<span>私有化 RAG 落地套件 · 检索优化与算力 TCO 评估</span></div>'
        f'<div class="rail-meta">'
        f'<span>语料 {stats["doc_count"]} 篇 / {stats["table_count"]} 张表格 / {n_bench} 题基准</span>'
        f'<span>向量后端 {adv.retriever.vector_index.backend.upper()} + BM25 混合检索</span>'
        f'{mode_badge}'
        f'</div></div>', unsafe_allow_html=True)

    if demo:
        st.markdown(
            '<div class="muted" style="margin:0 0 14px 2px">Demo Mode：检索、评测与 TCO 测算'
            '全部为真实运行结果；仅「生成式作答」降级为抽取式（未配置 LLM）。'
            '填写 <code>.env</code> 即可接入 vLLM / Ollama / 云端 API。</div>',
            unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs([
        "① 架构对比与问答沙盒",
        f"② {n_bench} 题基准评测",
        "③ 算力选型与 TCO 测算",
        "④ 并发压测基准",
    ])
    with t1:
        tab_sandbox(adv, naive, questions, docs)
    with t2:
        tab_benchmark(bench)
    with t3:
        tab_tco()
    with t4:
        tab_perf(bench)

    st.markdown("---")
    st.markdown(
        '<div class="muted">技术栈：Python 3.10+ / Streamlit / Plotly / BM25 与向量混合检索 / '
        '父子分段 / vLLM 与 Ollama</div>'
        '<div class="muted" style="margin-top:4px">全部数字均可复现：'
        '<code>python scripts/smoke_test.py</code>　或　<code>python -m core.evaluator --write</code></div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
