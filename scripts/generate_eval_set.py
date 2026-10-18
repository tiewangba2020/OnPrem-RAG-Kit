# -*- coding: utf-8 -*-
"""scripts/generate_eval_set.py —— 从样例文档的表格行机械派生规模化评测集

为什么这些题算"真实"
--------------------
1. **标准答案不是人手写的**：答案字段逐字取自文档表格的单元格原值。题目文本由
   「表头 + 行键」拼成，出题人没有把答案写进题目的空间。
2. **每道题都带 `gold_row`**：文档里那一行的原文整行。表格解析准确率就是检查这
   一整行有没有被完整召回，任何第三方都能拿原文逐字复核。
3. **只对有区分度的值出题**：某个值若在语料里反复出现（"否"、"是"、"30 分钟"），
   拿它单独做召回判定会失真——错误片段也能"命中"。这类列自动降权或跳过，
   跳过规则写在 `pick_target_column()` 里，可复核。
4. **跨文档题有原文依据**：两篇制度互相引用的条款（如"招标程序执行 A 办法、
   审批权限执行 B 制度"）直接摘自原文第 1.3 条。

用法::

    python scripts/generate_eval_set.py            # 生成 data/eval_bench_questions.json
    python scripts/generate_eval_set.py --show     # 只打印统计，不写文件
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "sample_docs"
BASE_QUESTIONS = DATA_DIR / "eval_20_questions.json"
OUT_PATH = DATA_DIR / "eval_bench_questions.json"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

# 文档清单：(doc_id, 文件名, 简称, 完整标题)
DOCS: List[Tuple[str, str, str, str]] = [
    ("cnc_sop", "1_精密数控机床故障排查SOP手册.md", "CNC 故障排查 SOP",
     "精密数控机床故障排查SOP手册"),
    ("construction_spec", "2_建筑工程招投标与施工质量验收管理办法.md", "工程招投标办法",
     "建筑工程招投标与施工质量验收管理办法（规范）"),
    ("procurement_policy", "3_企业大额采购审批与合规管理制度.md", "大额采购制度",
     "企业大额采购审批与合规管理制度"),
]

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")

# 太泛、不足以单独作为召回判定的值
_TRIVIAL_VALUES = {
    "是", "否", "无", "—", "-", "／", "/", "不适用", "全数检查", "每日巡查",
    "内部", "通用", "其他", "备注",
}


def split_cells(row: str) -> List[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def parse_tables(path: Path) -> List[Dict[str, Any]]:
    """解析 Markdown，返回 [{heading, header, rows:[cells], raw_rows:[str]}]。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    stack: List[Tuple[int, str]] = []
    tables: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        m = HEADING_RE.match(lines[i])
        if m:
            lvl = len(m.group(1))
            stack = [(l, t) for l, t in stack if l < lvl]
            stack.append((lvl, m.group(2).strip()))
            i += 1
            continue
        if TABLE_ROW_RE.match(lines[i]):
            raw: List[str] = []
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                raw.append(lines[i].strip())
                i += 1
            body = [r for r in raw if not TABLE_SEP_RE.match(r)]
            if len(body) >= 2:
                tables.append({
                    "heading": " › ".join(t for _, t in stack),
                    "header": split_cells(body[0]),
                    "rows": [split_cells(r) for r in body[1:]],
                    "raw_rows": body[1:],
                })
            continue
        i += 1
    return tables


# ---------------------------------------------------------------------------
# 列语义：把列名映射成自然的中文问法
# ---------------------------------------------------------------------------

_QUESTION_TEMPLATES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"处理时限|响应时限|时限"), "的处理时限是多久"),
    (re.compile(r"公示时长|公示要求|公示"), "的公示时长 / 公示要求是什么"),
    (re.compile(r"建议更换周期|更换周期|周期"), "的建议更换周期是多久"),
    (re.compile(r"缺陷责任期"), "的缺陷责任期是多久"),
    (re.compile(r"保修"), "的保修期是多久"),
    (re.compile(r"上限"), "的上限是多少"),
    (re.compile(r"比例"), "的比例是多少"),
    (re.compile(r"免罚天数"), "的免罚天数是多少"),
    (re.compile(r"库存下限"), "的库存下限是多少"),
    (re.compile(r"合格标准|判定标准|目标值"), "的合格标准 / 判定标准是什么"),
    (re.compile(r"触发条件"), "的触发条件是什么"),
    (re.compile(r"风险等级"), "的风险等级是什么"),
    (re.compile(r"审批链|审批层级|审批"), "的审批链 / 审批层级是什么"),
    (re.compile(r"投标家数|家数|最少询价"), "的最少询价 / 投标家数是多少"),
    (re.compile(r"定标规则|评标方法"), "的定标 / 评标规则是什么"),
    (re.compile(r"检测方法"), "的检测方法是什么"),
    (re.compile(r"抽样频次"), "的抽样频次是什么"),
    (re.compile(r"归档时限"), "的归档时限是多久"),
    (re.compile(r"退还时限|退还节点"), "的退还时限 / 退还节点是什么"),
    (re.compile(r"处罚标准"), "的处罚标准是什么"),
    (re.compile(r"禁入期限"), "的禁入期限是多久"),
    (re.compile(r"累计付款比例|付款"), "的付款比例 / 触发条件是什么"),
    (re.compile(r"适用情形|情形"), "的适用情形是什么"),
    (re.compile(r"规格型号|型号"), "的规格型号是什么"),
    (re.compile(r"适用机型"), "的适用机型是什么"),
    (re.compile(r"供应商"), "的供应商是谁"),
    (re.compile(r"中文释义|全称|释义"), "的全称 / 中文释义是什么"),
    (re.compile(r"使用工具"), "的排查使用工具是什么"),
    (re.compile(r"处置动作"), "的处置动作是什么"),
    (re.compile(r"责任人"), "的责任人是谁"),
    (re.compile(r"记录表单"), "的记录表单是什么"),
    (re.compile(r"采购方式|招标方式"), "的采购 / 招标方式是什么"),
    (re.compile(r"归属模块"), "的归属模块是什么"),
]


def question_phrase(column: str) -> str:
    for pat, phrase in _QUESTION_TEMPLATES:
        if pat.search(column):
            return phrase
    return f"的「{column}」是什么"


def is_ambiguous(header: Sequence[str], target_col: str) -> bool:
    """同一张表里若有多列命中同一个问法模板，则该问法有歧义。

    典型例子：保证金表同时有「投标保证金比例」与「履约保证金比例」，
    问"比例是多少"无法确定问的是哪一个——这种题必须显式点名列名，
    否则标准答案就是出题人单方面指定的，不成立。
    """
    phrase = question_phrase(target_col)
    same = [c for c in header if question_phrase(c) == phrase]
    return len(same) > 1


def is_trivial(value: str) -> bool:
    v = value.strip()
    if len(v) < 2:
        return True
    if v in _TRIVIAL_VALUES:
        return True
    return False


def numeric_signal(value: str) -> int:
    """值里出现数字 / 单位 / 比例符号，说明它是有信息量的判定值。"""
    return len(re.findall(r"\d", value))


# ---------------------------------------------------------------------------
# 出题
# ---------------------------------------------------------------------------

def pick_key_and_target(header: Sequence[str], row: Sequence[str],
                        freq: Counter, variant: int = 0) -> Optional[Tuple[str, str, str, str]]:
    """返回 (key, key_col, target, target_col)，选不出有区分度的组合则返回 None。

    选择规则（可复核）：
    1. 第 1 列作行键（工程表格的通行约定），跳过过短或过泛的键；
    2. 候选列 = 「带数字且全语料出现次数少」的单元格——这类值才具备区分度；
    3. 按 `variant` 在候选列里轮转，避免同一张表的每一行都在问同一列
       （否则 8 行故障代码会生成 8 道一模一样的"触发条件"题）。
    """
    if len(row) < 2 or len(header) < 2:
        return None
    key, key_col = row[0].strip(), header[0].strip()
    if is_trivial(key) and len(key) < 3:
        return None

    cands: List[Tuple[int, str, str]] = []           # (score, value, column)
    for col, val in zip(header[1:], row[1:]):
        v = val.strip()
        if is_trivial(v) or col == key_col:
            continue
        occ = freq.get(v, 0)
        if occ > 6:                                   # 太常见，做召回判定会失真
            continue
        score = numeric_signal(v) * 2 - occ * 3
        if len(v) >= 6:
            score += 1
        cands.append((score, v, col))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    # 只在得分最高的三列之间轮转，既保证信息量又保证覆盖度
    pool = cands[:3]
    _, target, target_col = pool[variant % len(pool)]
    return key, key_col, target, target_col


def _space_before_cjk(key: str, phrase: str) -> str:
    """'E-402' + '的触发条件' → 'E-402 的触发条件'，避免中英混排挤在一起。"""
    if key and key[-1].isascii() and key[-1].isalnum() and phrase.startswith("的"):
        return " " + phrase
    return phrase


def build_generated_questions() -> List[Dict[str, Any]]:
    # 先统计全语料单元格值频次，用于筛掉"到处都有"的值
    freq: Counter = Counter()
    all_tables: List[Tuple[Tuple[str, str, str, str], Dict[str, Any]]] = []
    for meta in DOCS:
        path = DOCS_DIR / meta[1]
        if not path.exists():
            raise SystemExit(f"[error] 缺少文档 {path}")
        for t in parse_tables(path):
            all_tables.append((meta, t))
            for r in t["rows"]:
                for c in r[1:]:
                    if c.strip():
                        freq[c.strip()] += 1

    out: List[Dict[str, Any]] = []
    n = 0
    for (doc_id, _fname, short, full_title), t in all_tables:
        header, rows, raws = t["header"], t["rows"], t["raw_rows"]
        heading = t["heading"]
        # 章节末段更可读，例如「3 典型故障排查标准作业流程 › 3.1 E-402 主轴驱动过载」
        tail = heading.split(" › ")[-1] if heading else ""
        for ri, (row, raw) in enumerate(zip(rows, raws)):
            picked = pick_key_and_target(header, row, freq, variant=ri)
            if not picked:
                continue
            key, _key_col, target, target_col = picked
            if is_ambiguous(header, target_col):
                # 多列同义 → 显式点名列名
                phrase = _space_before_cjk(key, f"的「{target_col}」是什么")
            else:
                phrase = _space_before_cjk(key, question_phrase(target_col))
            where = f"《{short}》" + (f"「{tail}」" if tail else "")
            question = f"{where}中，{key}{phrase}？"
            if len(question) > 130:                     # 过长的区间键会把题目撑爆
                continue
            n += 1
            out.append({
                "id": f"G{n:03d}",
                "question": question,
                "doc_id": doc_id,
                "doc_title": full_title,
                "dimension": "表格数值检索" if numeric_signal(target) else "术语定义",
                "difficulty": "medium",
                "answer_type": "table",
                "recall_facts": [key, target] if key != target else [target],
                "answer_facts": [target],
                "allow_computed": [],
                "gold_row": raw,
                "gold_evidence": [raw],
                "gold_answer": f"{where}中，{key} · {target_col}：{target}。",
                "source": "table-row-derived",
            })
    return out


# ---------------------------------------------------------------------------
# 跨文档 / 跨条款题（人工设计，但证据全部逐字取自原文）
# ---------------------------------------------------------------------------

def build_crossdoc_questions() -> List[Dict[str, Any]]:
    spec = DOCS[1]
    proc = DOCS[2]
    cnc = DOCS[0]
    return [
        {
            "id": "X001",
            "question": "工程类采购的「招标程序」与「审批权限」分别执行哪一份制度？两套制度是什么关系？",
            "doc_id": spec[0], "doc_title": spec[3],
            "dimension": "跨文档推理", "difficulty": "hard", "answer_type": "prose",
            "recall_facts": ["招标程序", "审批权限", "互为补充"],
            "answer_facts": ["招标程序", "审批权限", "互为补充"],
            "allow_computed": [],
            "gold_row": "",
            "gold_evidence": [
                "1.3 工程类采购的**招标程序**执行本办法，**审批权限**执行集团《企业大额采购审批与合规管理制度》。",
                "1.3 工程类采购的招标程序执行《建筑工程招投标与施工质量验收管理办法》，**审批权限执行本制度**。两制度并行适用、互为补充。",
            ],
            "gold_answer": "招标程序执行《建筑工程招投标与施工质量验收管理办法》，审批权限执行《企业大额采购审批与合规管理制度》；两制度并行适用、互为补充。",
            "source": "cross-doc",
        },
        {
            "id": "X002",
            "question": "单项采购含税金额 600 万元、且属于工程类采购时，招标方式与审批链分别是什么？",
            "doc_id": proc[0], "doc_title": proc[3],
            "dimension": "跨文档推理", "difficulty": "hard", "answer_type": "table",
            "recall_facts": ["300 万元（含）至 1000 万元", "公开招标", "采购委员会审议", "总经理"],
            "answer_facts": ["公开招标", "财务总监", "分管副总裁", "总经理", "采购委员会审议"],
            "allow_computed": [],
            "gold_row": "| 300 万元（含）至 1000 万元 | 公开招标 | 5 家 | 综合评分最高者中标 | 公开平台公示 7 个工作日 |",
            "gold_evidence": [
                "| 300 万元（含）至 1000 万元 | 公开招标 | 5 家 | 综合评分最高者中标 | 公开平台公示 7 个工作日 |",
                "| 300 万元（含）至 1000 万元 | 需求部门负责人 | 部门负责人 → 财务总监 → 分管副总裁 → 总经理 | 是 | 否 |",
                "| 500 万元（含）以上 | 公开招标 | 3 家（有效投标） | 综合评估法 | 7 个工作日 |",
            ],
            "gold_answer": "600 万元落入采购制度「300 万元（含）至 1000 万元」区间：采购方式为公开招标，最少 5 家投标，公示 7 个工作日；审批链为部门负责人 → 财务总监 → 分管副总裁 → 总经理，需采购委员会审议、无需董事会审批。",
            "source": "cross-doc",
        },
        {
            "id": "X003",
            "question": "同一故障在机床上连续复现 3 次以上，SOP 要求在多长时间内提交什么报告？由谁主导？",
            "doc_id": cnc[0], "doc_title": cnc[3],
            "dimension": "跨条款推理", "difficulty": "medium", "answer_type": "table",
            "recall_facts": ["同一故障连续复现 3 次", "24 小时内提交根因分析报告", "设备工程师"],
            "answer_facts": ["24 小时内", "根因分析报告", "设备工程师"],
            "allow_computed": [],
            "gold_row": "| 同一故障连续复现 3 次 | 24 小时内提交根因分析报告 | 设备工程师 | 生产技术中心 |",
            "gold_evidence": [
                "| 同一故障连续复现 3 次 | 24 小时内提交根因分析报告 | 设备工程师 | 生产技术中心 |",
                "3. 同一故障连续复现 3 次以上，须升级为设备工程师主导的根因分析（RCA）。",
            ],
            "gold_answer": "同一故障连续复现 3 次以上，须升级为设备工程师主导的根因分析（RCA）；应急路径表要求 24 小时内提交根因分析报告，升级至生产技术中心。",
            "source": "cross-section",
        },
        {
            "id": "X004",
            "question": "E-815 安全门联锁断开时，是否允许旁路运行？处理时限多久？复位失败时检查哪两个部件？",
            "doc_id": cnc[0], "doc_title": cnc[3],
            "dimension": "例外与否定", "difficulty": "hard", "answer_type": "table",
            "recall_facts": ["E-815", "极高", "立即停机", "门锁开关 K15", "安全继电器 KA3"],
            "answer_facts": ["立即停机", "否", "K15", "KA3"],
            "allow_computed": [],
            "gold_row": "| E-815 | 安全门联锁断开 | 加工过程中安全门限位开关断开 | 极高 | 立即停机 | 否 | 安全联锁 |",
            "gold_evidence": [
                "| E-815 | 安全门联锁断开 | 加工过程中安全门限位开关断开 | 极高 | 立即停机 | 否 | 安全联锁 |",
                "4. 若连续 3 次复位仍失败，检查门锁开关 K15 与安全继电器 KA3，更换后再执行复位。",
            ],
            "gold_answer": "E-815 风险等级「极高」，处理时限为立即停机，不允许旁路运行；连续 3 次复位仍失败时检查门锁开关 K15 与安全继电器 KA3。",
            "source": "cross-section",
        },
        {
            "id": "X005",
            "question": "全表中哪些故障代码允许旁路运行？旁路单次时长上限与授权要求是什么？",
            "doc_id": cnc[0], "doc_title": cnc[3],
            "dimension": "例外与否定", "difficulty": "hard", "answer_type": "table",
            "recall_facts": ["仅 E-517 与 E-708 允许旁路运行", "30 分钟", "设备工程师书面授权"],
            "answer_facts": ["E-517", "E-708", "30 分钟", "设备工程师"],
            "allow_computed": [],
            "gold_row": "| E-517 | 气动单元低压报警 | 气源压力低于 0.45 MPa | 中 | 4 小时 | 是 | 气动系统 |",
            "gold_evidence": [
                "| E-517 | 气动单元低压报警 | 气源压力低于 0.45 MPa | 中 | 4 小时 | 是 | 气动系统 |",
                "| E-708 | 冷却液流量不足 | 冷却液流量低于设定值的 60% | 低 | 8 小时 | 是 | 冷却排屑 |",
                "经核查，全表**仅 E-517 与 E-708 允许旁路运行**，且旁路单次时长不得超过 30 分钟、须由设备工程师书面授权；其余故障代码一律禁止旁路。",
            ],
            "gold_answer": "全表仅 E-517 与 E-708 允许旁路运行，旁路单次时长不得超过 30 分钟，须由设备工程师书面授权；其余故障代码一律禁止旁路。",
            "source": "cross-section",
        },
        {
            "id": "X006",
            "question": "伺服电机编码器校准属于哪个点检周期？年度大修侧重哪些项目？",
            "doc_id": cnc[0], "doc_title": cnc[3],
            "dimension": "边界条件", "difficulty": "medium", "answer_type": "table",
            "recall_facts": ["季度", "伺服电机编码器校准", "整机几何精度检测", "主轴动平衡"],
            "answer_facts": ["季度", "几何精度", "主轴动平衡"],
            "allow_computed": [],
            "gold_row": "| 季度 | 伺服电机编码器校准、导轨直线度 | 定位精度不低于 0.01 mm/300 mm | 激光干涉仪 | 设备工程师 | F-04 季度校准报告 |",
            "gold_evidence": [
                "| 季度 | 伺服电机编码器校准、导轨直线度 | 定位精度不低于 0.01 mm/300 mm | 激光干涉仪 | 设备工程师 | F-04 季度校准报告 |",
                "| 年度 | 整机几何精度检测、主轴动平衡 | 主轴振动速度不高于 2.8 mm/s（ISO 10816 二级） | 动平衡仪 | 设备工程部 | F-06 年度大修报告 |",
                "**注意**：伺服电机编码器校准属于**季度**项目，不属于年度大修项目。",
            ],
            "gold_answer": "伺服电机编码器校准属于季度项目，不属于年度大修；年度大修侧重整机几何精度检测与主轴动平衡（主轴振动速度不高于 2.8 mm/s）。",
            "source": "cross-section",
        },
        {
            "id": "X007",
            "question": "承包人逾期提交不可抗力书面通知会有什么后果？通知与证明文件的时限分别是多久？",
            "doc_id": spec[0], "doc_title": spec[3],
            "dimension": "例外与否定", "difficulty": "hard", "answer_type": "prose",
            "recall_facts": ["48 小时", "14 日", "免罚天数按 0 计"],
            "answer_facts": ["48 小时", "14 日", "免罚天数按 0 计"],
            "allow_computed": [],
            "gold_row": "",
            "gold_evidence": [
                "8.2 承包人应在不可抗力事件发生后 **48 小时**内向监理单位提交书面通知，并在 **14 日**内提交有效证明文件（政府公告、气象证明、现场影像等）。",
                "8.3 逾期提交书面通知或证明文件的，**免罚天数按 0 计**，且不得再主张适用本办法第 4 章的免罚期条款。",
            ],
            "gold_answer": "承包人应在不可抗力事件发生后 48 小时内提交书面通知、14 日内提交有效证明文件；逾期提交的，免罚天数按 0 计，且不得再主张适用第 4 章免罚期条款。",
            "source": "cross-section",
        },
        {
            "id": "X008",
            "question": "投标保证金的比例与上限如何同时适用？以哪一个为准？",
            "doc_id": spec[0], "doc_title": spec[3],
            "dimension": "边界条件", "difficulty": "medium", "answer_type": "prose",
            "recall_facts": ["同时适用", "以二者中的较低者为准"],
            "answer_facts": ["同时适用", "较低者"],
            "allow_computed": [],
            "gold_row": "",
            "gold_evidence": [
                "> 投标保证金比例与上限**同时适用**，以二者中的较低者为准。",
                "| 500 万元（含）至 2000 万元 | 1.0% | 20 万元 | 5% | 100 万元 | 中标通知书发出后 7 个工作日 |",
            ],
            "gold_answer": "投标保证金比例与上限同时适用，以二者中的较低者为准。",
            "source": "cross-section",
        },
        {
            "id": "X009",
            "question": "防水与屋面工程的缺陷责任期与最低保修期分别是多久？与主体结构相比有何差异？",
            "doc_id": spec[0], "doc_title": spec[3],
            "dimension": "表格数值检索", "difficulty": "medium", "answer_type": "table",
            "recall_facts": ["60 个月", "24 个月", "防水与屋面工程"],
            "answer_facts": ["60 个月", "24 个月"],
            "allow_computed": [],
            "gold_row": "| 防水与屋面工程 | 60 个月 | 60 个月 | 5% | 合同金额的 5% | 5 年保修期满后 28 日内 |",
            "gold_evidence": [
                "| 防水与屋面工程 | 60 个月 | 60 个月 | 5% | 合同金额的 5% | 5 年保修期满后 28 日内 |",
                "| 房屋建筑主体结构 | 24 个月 | 设计使用年限 | 3% | 合同金额的 3% | 缺陷责任期满且无质量问题后 14 日内 |",
                "> 缺陷责任期自**实际竣工日期**起算，不以竣工验收备案日期起算。防水与屋面工程的缺陷责任期为 60 个月，显著长于主体结构的 24 个月。",
            ],
            "gold_answer": "防水与屋面工程的缺陷责任期为 60 个月、最低保修期 60 个月；主体结构缺陷责任期为 24 个月、最低保修期为设计使用年限。缺陷责任期自实际竣工日期起算。",
            "source": "cross-section",
        },
        {
            "id": "X010",
            "question": "质量保证金比例与上限如何约束？房屋建筑主体结构的退还节点是什么？",
            "doc_id": spec[0], "doc_title": spec[3],
            "dimension": "表格数值检索", "difficulty": "medium", "answer_type": "table",
            "recall_facts": ["3%", "合同金额的 3%", "缺陷责任期满且无质量问题后 14 日内"],
            "answer_facts": ["3%", "14 日内"],
            "allow_computed": [],
            "gold_row": "| 房屋建筑主体结构 | 24 个月 | 设计使用年限 | 3% | 合同金额的 3% | 缺陷责任期满且无质量问题后 14 日内 |",
            "gold_evidence": [
                "| 房屋建筑主体结构 | 24 个月 | 设计使用年限 | 3% | 合同金额的 3% | 缺陷责任期满且无质量问题后 14 日内 |",
            ],
            "gold_answer": "房屋建筑主体结构的质量保证金比例为 3%、上限为合同金额的 3%，退还节点为缺陷责任期满且无质量问题后 14 日内。",
            "source": "cross-section",
        },
    ]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_all() -> List[Dict[str, Any]]:
    """20 道人工题 + 10 道跨文档题 + N 道表格行派生题。"""
    base: List[Dict[str, Any]] = []
    if BASE_QUESTIONS.exists():
        raw = json.loads(BASE_QUESTIONS.read_text(encoding="utf-8"))
        base = raw.get("questions", []) if isinstance(raw, dict) else raw
        for q in base:
            q.setdefault("source", "hand-crafted")
    return base + build_crossdoc_questions() + build_generated_questions()


def write_questions(path: Path = OUT_PATH) -> Tuple[Path, int]:
    """把题库落盘，返回 (路径, 题数)。供 generate_sample_data.py 首次启动时调用。"""
    questions = build_all()
    n_gen = sum(1 for q in questions if q.get("source") == "table-row-derived")
    n_cross = sum(1 for q in questions if str(q.get("source", "")).startswith("cross"))
    n_base = len(questions) - n_gen - n_cross
    payload = {
        "meta": {
            "version": "2.0",
            "total": len(questions),
            "description": (
                f"{len(questions)} 道规模化评测题：{n_base} 道人工设计的刁钻题 + "
                f"{n_cross} 道跨文档/跨条款题 + {n_gen} 道由文档表格行机械派生的题。"
                "派生题的标准答案逐字取自文档单元格原值，gold_row 为该行原文整行，可逐字复核。"
            ),
            "documents": [{"doc_id": d[0], "file": d[1], "title": d[3]} for d in DOCS],
            "generator": "scripts/generate_eval_set.py",
        },
        "questions": questions,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, len(questions)


def main() -> None:
    ap = argparse.ArgumentParser(description="从样例文档派生规模化评测集")
    ap.add_argument("--show", action="store_true", help="只打印统计，不写文件")
    args = ap.parse_args()

    questions = build_all()
    dims = Counter(q["dimension"] for q in questions)
    srcs = Counter(q.get("source", "?") for q in questions)
    docs = Counter(q["doc_id"] for q in questions)

    print(f"[gen] 合计 {len(questions)} 道")
    print(f"[gen] 来源分布: {dict(srcs)}")
    print(f"[gen] 维度分布: {dict(dims)}")
    print(f"[gen] 文档分布: {dict(docs)}")

    if args.show:
        for q in questions[-6:]:
            print(f"\n  {q['id']} [{q['dimension']}] {q['question']}")
            print(f"     答案: {q['gold_answer'][:90]}")
        return

    path, n = write_questions()
    print(f"[gen] 已写入 {path}  ({path.stat().st_size / 1024:.0f} KB, {n} 题)")


if __name__ == "__main__":
    main()
