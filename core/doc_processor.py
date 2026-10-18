# -*- coding: utf-8 -*-
"""core.doc_processor —— 结构化文档解析与父子分段（Parent-Child Chunking）

设计目标
--------
1. **表格完整性优先**：Markdown 表格在解析阶段被识别为原子块，切分时绝不跨越表格边界，
   每一行都会带上表头成为"自描述行块"，父块则保存整张表，检索命中行块后可回溯整表。
2. **父子分段**：子块（child）用于召回，粒度细、可精确定位；父块（parent）用于喂给 LLM，
   上下文完整。命中子块 → 返回父块，兼顾精度与完整性。
3. **结构化标题路径**：每个块都携带 heading_path（如 "3. 典型故障排查 › 3.1 E-402 主轴驱动过载"），
   既用于构造可读引用，也作为检索时的上下文增强字段（contextual header）。

对外接口
--------
    docs = load_documents("data/sample_docs")
    stats = corpus_stats(docs)
    naive = naive_fixed_chunks(doc)          # 对照组：固定长度暴力切分
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

CHILD_MIN_CHARS = 110          # 子块合并下限：过短的段落向后合并
CHILD_MAX_CHARS = 680          # 子块切分上限：超长段落按句切分
PARENT_MAX_CHARS = 3000        # 父块上限（喂给 LLM 的上下文片段）
NAIVE_CHUNK_CHARS = 320        # 对照组固定窗口长度（模拟 CharacterTextSplitter）

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
SENT_SPLIT_RE = re.compile(r"(?<=[。；！？!?])")

DOC_INDUSTRY = {
    "cnc_sop": "装备制造",
    "construction_spec": "建筑工程",
    "procurement_policy": "企业合规",
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """检索单元（子块）。"""
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str                       # 用于展示的正文
    heading_path: str = ""
    chunk_type: str = "text"        # text | table_row | naive
    parent_id: str = ""
    table_title: str = ""
    row_index: int = -1
    char_len: int = 0

    def index_text(self) -> str:
        """用于建索引的增强文本：标题路径 + 表名 + 正文。"""
        prefix = " ".join(x for x in (self.heading_path, self.table_title) if x)
        return f"{prefix}\n{self.text}" if prefix else self.text

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["char_len"] = len(self.text)
        return d


@dataclass
class ParentBlock:
    """上下文单元（父块）：完整小节正文，或一张完整的表。"""
    parent_id: str
    doc_id: str
    doc_title: str
    heading_path: str
    text: str
    kind: str = "section"           # section | table
    table_title: str = ""
    child_ids: List[str] = field(default_factory=list)
    section_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Document:
    doc_id: str
    title: str
    industry: str
    path: str
    raw_text: str
    parents: Dict[str, ParentBlock] = field(default_factory=dict)
    children: List[Chunk] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)   # 父块超限导致的内容截断/丢弃

    @property
    def n_tables(self) -> int:
        return sum(1 for p in self.parents.values() if p.kind == "table")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id, "title": self.title, "industry": self.industry,
            "path": self.path, "char_len": len(self.raw_text),
            "parents": [p.to_dict() for p in self.parents.values()],
            "children": [c.to_dict() for c in self.children],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _sid(*parts: str) -> str:
    import hashlib
    return hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()[:10]


def _norm_space(text: str) -> str:
    return re.sub(r"[ \t\u3000]+", " ", text).strip()


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中文约 1 字 ≈ 0.65 token，英文数字按 4 字符 1 token。"""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return int(cjk * 0.7 + other / 3.6) + 1


# ---------------------------------------------------------------------------
# Markdown 结构化解析
# ---------------------------------------------------------------------------

def _is_table_row(line: str) -> bool:
    return bool(TABLE_ROW_RE.match(line))


def _is_table_sep(line: str) -> bool:
    return bool(TABLE_SEP_RE.match(line)) and "-" in line


def split_markdown_blocks(text: str) -> List[Dict[str, Any]]:
    """把 Markdown 切成结构化块：heading / para / table。

    表格作为**原子块**返回，rows 保留原始行，不参与任何字符切分。
    """
    lines = text.splitlines()
    blocks: List[Dict[str, Any]] = []
    buf: List[str] = []

    def flush_para() -> None:
        nonlocal buf
        if buf:
            body = "\n".join(buf).strip()
            if body:
                blocks.append({"type": "para", "text": body})
            buf = []

    i = 0
    while i < len(lines):
        line = lines[i]
        m = HEADING_RE.match(line)
        if m:
            flush_para()
            blocks.append({"type": "heading", "level": len(m.group(1)), "text": m.group(2).strip()})
            i += 1
            continue
        if _is_table_row(line):
            flush_para()
            rows: List[str] = []
            while i < len(lines) and _is_table_row(lines[i]):
                rows.append(_norm_space(lines[i]))
                i += 1
            if len(rows) >= 2:
                # 表格标题取块前最近的一条非表格行（由调用方按顺序回填）
                blocks.append({"type": "table", "rows": rows})
            else:
                blocks.append({"type": "para", "text": "\n".join(rows)})
            continue
        if line.strip() == "":
            flush_para()
            i += 1
            continue
        buf.append(line.rstrip())
        i += 1
    flush_para()
    return blocks


def _split_long_paragraph(text: str, max_chars: int = CHILD_MAX_CHARS) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    cur = ""
    for sent in SENT_SPLIT_RE.split(text):
        if not sent:
            continue
        if len(cur) + len(sent) > max_chars and cur:
            parts.append(cur)
            cur = ""
        cur += sent
    if cur:
        parts.append(cur)
    # 极端长句兜底硬切
    out: List[str] = []
    for p in parts:
        while len(p) > max_chars:
            out.append(p[:max_chars])
            p = p[max_chars:]
        if p:
            out.append(p)
    return out


def _table_row_caption(rows: List[str], fallback: str) -> str:
    """从表头行提炼一个可读的表名，例如 '步骤 | 排查项目 | 判定标准 / 目标值 …'。"""
    header = rows[0] if rows else ""
    cells = [c.strip() for c in header.strip("|").split("|") if c.strip()]
    if not cells:
        return fallback
    return " / ".join(cells[:6])


# ---------------------------------------------------------------------------
# 主解析：生成父子块
# ---------------------------------------------------------------------------

def parse_markdown(doc_id: str, title: str, text: str, path: str = "",
                   industry: str = "") -> Document:
    doc = Document(doc_id=doc_id, title=title, industry=industry or DOC_INDUSTRY.get(doc_id, "通用"),
                   path=path, raw_text=text)
    blocks = split_markdown_blocks(text)

    heading_stack: List[Tuple[int, str]] = []
    section_id = ""
    section_title_path = ""
    section_kind_seen = False

    def new_section() -> str:
        nonlocal section_kind_seen
        sid = f"{doc_id}-s{len([p for p in doc.parents.values() if p.kind == 'section']) + 1:02d}"
        head = heading_path()
        doc.parents[sid] = ParentBlock(
            parent_id=sid, doc_id=doc_id, doc_title=title,
            # 父块正文带上标题路径：既让 LLM 能引用章节号，也让"E-815"这类实体可被溯源
            heading_path=head, text=(f"## {head}" if head else ""), kind="section", section_id=sid,
        )
        section_kind_seen = False
        return sid

    def heading_path() -> str:
        return " › ".join(t for _, t in heading_stack if t)

    def ensure_section() -> str:
        nonlocal section_id
        if not section_id:
            section_id = new_section()
        return section_id

    pending_text: List[str] = []

    def flush_text_children() -> None:
        """把累积的段落文本合并/切分为子块并挂到当前小节的父块上。"""
        nonlocal pending_text
        if not pending_text:
            return
        sid = ensure_section()
        merged: List[str] = []
        cur = ""
        for p in pending_text:
            p = p.strip()
            if not p:
                continue
            if len(cur) + len(p) <= CHILD_MAX_CHARS:
                cur = (cur + "\n" + p).strip()
            else:
                if len(cur) >= CHILD_MIN_CHARS:
                    merged.append(cur)
                    cur = p
                else:  # 当前内容过短，与上一段合并后一起切分
                    merged.extend(_split_long_paragraph((cur + "\n" + p).strip()))
                    cur = ""
        if cur:
            merged.append(cur)

        for piece in merged:
            for sub in _split_long_paragraph(piece):
                cid = f"{doc_id}-c{len(doc.children) + 1:04d}"
                ch = Chunk(chunk_id=cid, doc_id=doc_id, doc_title=title, text=sub.strip(),
                           heading_path=heading_path(), chunk_type="text", parent_id=sid,
                           char_len=len(sub))
                doc.children.append(ch)
                doc.parents[sid].child_ids.append(cid)
                parent = doc.parents[sid]
                if len(parent.text) + len(sub) + 2 <= PARENT_MAX_CHARS:
                    parent.text = (parent.text + "\n\n" + sub).strip()
                else:
                    # 早期版本在这里静默丢弃正文：检索命中子块、回溯父块时内容却不见了。
                    # 现在显式记录，交给 corpus_stats / CLI 暴露出来。
                    doc.warnings.append(
                        f"{sid} 父块已达 {PARENT_MAX_CHARS} 字上限，子块 {cid} 未并入父块正文")
        pending_text = []

    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if blk["type"] == "heading":
            lvl = blk["level"]
            flush_text_children()
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, blk["text"]))
            if lvl <= 3:
                section_id = new_section()
            i += 1
            continue

        if blk["type"] == "table":
            flush_text_children()
            sid = ensure_section()
            rows: List[str] = blk["rows"]
            caption = _table_row_caption(rows, fallback=heading_path() or title)
            tid = f"{doc_id}-t{len([p for p in doc.parents.values() if p.kind == 'table']) + 1:02d}"
            table_md = f"## {heading_path()}\n【表格】{caption}\n" + "\n".join(rows) if heading_path() \
                else f"【表格】{caption}\n" + "\n".join(rows)
            if len(table_md) > PARENT_MAX_CHARS:
                # 表格原子块的承诺是"绝不切断"，一旦超限必须显式告警，不能悄悄截断
                doc.warnings.append(
                    f"{tid} 整表 {len(table_md)} 字超过父块上限 {PARENT_MAX_CHARS}，"
                    f"表格将被截断（建议调大 PARENT_MAX_CHARS 或对该表做分片）")
            doc.parents[tid] = ParentBlock(
                parent_id=tid, doc_id=doc_id, doc_title=title, heading_path=heading_path(),
                text=table_md[:PARENT_MAX_CHARS], kind="table", table_title=caption, section_id=sid,
            )
            # 整表也作为一段正文并入小节父块，保证"整节上下文"可用
            if len(doc.parents[sid].text) + len(table_md) + 2 <= PARENT_MAX_CHARS:
                doc.parents[sid].text = (doc.parents[sid].text + "\n\n" + table_md).strip()

            header_idx = 1 if len(rows) > 1 and _is_table_sep(rows[1]) else 0
            header = rows[header_idx] if rows else ""
            body_rows = rows[header_idx + 1:] if header else rows
            for r_i, row in enumerate(body_rows):
                cid = f"{doc_id}-c{len(doc.children) + 1:04d}"
                ch = Chunk(chunk_id=cid, doc_id=doc_id, doc_title=title,
                           text=f"{header}\n{row}" if header else row,
                           heading_path=heading_path(), chunk_type="table_row", parent_id=tid,
                           table_title=caption, row_index=r_i, char_len=len(row))
                doc.children.append(ch)
                doc.parents[tid].child_ids.append(cid)
            i += 1
            continue

        pending_text.append(blk["text"])
        i += 1

    flush_text_children()

    # 清理空父块
    doc.parents = {k: v for k, v in doc.parents.items() if v.text.strip()}
    valid = set(doc.parents)
    doc.children = [c for c in doc.children if c.parent_id in valid]
    return doc


# ---------------------------------------------------------------------------
# 对照组：固定长度暴力切分（Naive RAG 的典型做法）
# ---------------------------------------------------------------------------

def naive_fixed_chunks(doc: Document, size: int = NAIVE_CHUNK_CHARS,
                       overlap: int = 0) -> List[Chunk]:
    """不做任何结构识别的字符窗口切分——表格行会被拦腰截断。

    这正是 Naive RAG 在工程文档上丢分的第一大原因。
    """
    text = doc.raw_text
    chunks: List[Chunk] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if not piece.strip():
            continue
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}-n{len(chunks) + 1:04d}",
            doc_id=doc.doc_id, doc_title=doc.title, text=piece,
            heading_path="", chunk_type="naive", parent_id="", char_len=len(piece),
        ))
    return chunks


# ---------------------------------------------------------------------------
# 文档装载
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> Optional[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return None
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except Exception:
        return None


def _doc_id_from_name(stem: str) -> Tuple[str, str]:
    """从文件名推断 doc_id 与标题，例如 '1_精密数控机床故障排查SOP手册'。"""
    name = re.sub(r"^\d+[_\-.]*", "", stem).strip()
    table = [
        ("cnc", "cnc_sop"), ("机床", "cnc_sop"),
        ("建筑", "construction_spec"), ("招投标", "construction_spec"),
        ("采购", "procurement_policy"), ("合规", "procurement_policy"),
    ]
    for key, did in table:
        if key in name:
            return did, name
    return _sid(stem)[:8], name


def load_documents(docs_dir: str | Path) -> List[Document]:
    """装载目录下的 .md / .txt / .pdf 文档并完成父子分段。"""
    docs_dir = Path(docs_dir)
    out: List[Document] = []
    if not docs_dir.exists():
        return out
    files = sorted([p for p in docs_dir.iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"}])
    for path in files:
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
            if text is None:
                continue
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        doc_id, title = _doc_id_from_name(path.stem)
        doc = parse_markdown(doc_id=doc_id, title=title, text=text,
                             path=str(path), industry=DOC_INDUSTRY.get(doc_id, "通用"))
        out.append(doc)
    return out


def load_questions(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("questions", data if isinstance(data, list) else [])


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def corpus_stats(docs: Sequence[Document], naive_size: int = NAIVE_CHUNK_CHARS) -> Dict[str, Any]:
    children = [c for d in docs for c in d.children]
    table_rows = [c for c in children if c.chunk_type == "table_row"]
    text_children = [c for c in children if c.chunk_type == "text"]
    naive = [c for d in docs for c in naive_fixed_chunks(d, size=naive_size)]
    return {
        "doc_count": len(docs),
        "char_total": sum(len(d.raw_text) for d in docs),
        "token_estimate": sum(estimate_tokens(d.raw_text) for d in docs),
        "parent_count": sum(len(d.parents) for d in docs),
        "table_count": sum(d.n_tables for d in docs),
        "child_count": len(children),
        "child_text_count": len(text_children),
        "child_table_row_count": len(table_rows),
        "avg_child_chars": round(sum(c.char_len for c in children) / max(1, len(children)), 1),
        "naive_chunk_count": len(naive),
        "naive_avg_chars": round(sum(c.char_len for c in naive) / max(1, len(naive)), 1),
        "naive_broken_table_rows": count_broken_table_rows(docs, naive),
        "truncation_warnings": sum(len(d.warnings) for d in docs),
    }


def count_broken_table_rows(docs: Sequence[Document], naive_chunks: Sequence[Chunk]) -> int:
    """统计有多少张表行在固定窗口切分下被截断——这是 Naive RAG 的硬伤量化指标。"""
    broken = 0
    for d in docs:
        joined = "\n".join(c.text for c in naive_chunks if c.doc_id == d.doc_id)
        for parent in d.parents.values():
            if parent.kind != "table":
                continue
            for line in parent.text.splitlines():
                line = _norm_space(line)
                if line.startswith("|") and line not in joined:
                    broken += 1
    return broken


def table_data_rows(docs: Sequence[Document]) -> List[str]:
    """取出语料中所有表格的数据行（去掉表头与分隔行）。"""
    rows: List[str] = []
    for d in docs:
        for parent in d.parents.values():
            if parent.kind != "table":
                continue
            lines = [_norm_space(x) for x in parent.text.splitlines() if x.strip().startswith("|")]
            for i, line in enumerate(lines):
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if _is_table_sep(line) or _is_table_sep(nxt):
                    continue          # 分隔行 / 表头行
                rows.append(line)
    return rows


def table_row_status(rows: Sequence[str], text: str) -> Dict[str, int]:
    """统计给定文本对一批表格行的覆盖情况，返回 {full, partial, missing}。

    这是 UI 与评测共用的**唯一**判定口径，避免出现"同一个现象两个数"：
    * full    整行原样出现（可被完整引用）
    * partial 只出现了被窗口切断的残行（前缀或后缀）
    * missing 完全没出现
    """
    body = _norm_space(text)
    lines = [_norm_space(x) for x in text.splitlines() if x.strip()]
    full = partial = missing = 0
    for row in rows:
        if not row:
            continue
        if row in body:
            full += 1
            continue
        cut = any(ln.startswith("|") and len(ln) >= 10 and (row.startswith(ln) or row.endswith(ln))
                  for ln in lines)
        if cut:
            partial += 1
        else:
            missing += 1
    return {"full": full, "partial": partial, "missing": missing}


def relevant_table_rows(docs: Sequence[Document], hard_tokens: Sequence[str]) -> List[str]:
    """挑出与查询硬 token 相关的表格行（故障码 / 数值 / 单位命中），用于逐题表格完整性评估。"""
    keys = {t for t in hard_tokens if t}
    if not keys:
        return []
    out: List[str] = []
    for row in table_data_rows(docs):
        low = row.lower()
        if any(k in low.replace(" ", "") or k in low for k in keys):
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    root = Path(__file__).resolve().parent.parent
    docs = load_documents(root / "data" / "sample_docs")
    stats = corpus_stats(docs)
    print("=" * 70)
    print("文档解析统计（Parent-Child Chunking）")
    print("=" * 70)
    for d in docs:
        print(f"  [{d.doc_id:<20}] {d.title[:34]:<36} 父块 {len(d.parents):>3}  子块 {len(d.children):>4}  表格 {d.n_tables}")
    print("-" * 70)
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    print("=" * 70)
    print(f"结论：子块 {stats['child_count']} 个（平均 {stats['avg_child_chars']} 字）"
          f"，对照组固定窗口 {stats['naive_chunk_count']} 个；"
          f"固定窗口下有 {stats['naive_broken_table_rows']} 行表格被截断。")
    if stats["truncation_warnings"]:
        print(f"⚠️  父块上限告警 {stats['truncation_warnings']} 条（内容被截断或未并入父块）：")
        for d in docs:
            for w in d.warnings[:5]:
                print(f"   - {d.doc_id}: {w}")


if __name__ == "__main__":
    main()
