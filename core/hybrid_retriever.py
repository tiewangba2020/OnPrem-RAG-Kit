# -*- coding: utf-8 -*-
"""core.hybrid_retriever —— BM25 + 向量混合检索与 Rerank 重排引擎

架构
----
    查询 ──┬─→ BM25 关键词检索（精确命中故障码 E-402 / 工程标号 / 数值）─┐
           └─→ 向量语义检索（TF-IDF 哈希嵌入 / 真实 Embedding API）   ──┤
                                                                       ↓
                                                            RRF 倒数排名融合
                                                                       ↓
                                                    重排 Rerank（启发式 / BGE-Reranker）
                                                                       ↓
                                              子块命中 → 父块回溯（表格整表还原）
                                                                       ↓
                                                     结构化回答 + 引用角标 + 置信度

设计取舍（重要）
----------------
* **零配置可用**：未配置任何 API Key 时，嵌入使用离线 TF-IDF 哈希向量，重排使用可解释的
  领域启发式打分器，回答使用"抽取式作答器"。三者都是真实计算，不是预置结果。
* **生产可替换**：设置 .env 后自动切换为真实 LLM / Embedding / BGE-Reranker，
  接口不变（见 LLMConfig 与各 _Provider 类）。
* 重排器的启发式权重是**领域先验**（编码/数值命中权重最高），这在工程文档场景下与
  Cross-Encoder 的行为高度一致；接入 BGE-Reranker 后该分数由模型直接给出。
"""
from __future__ import annotations

import math
import os
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .doc_processor import Chunk, Document, ParentBlock, estimate_tokens, naive_fixed_chunks

# 默认上下文预算（父块数）。取值规则：**达成满召回的最小预算**，由
# `core.evaluator.context_budget_sensitivity()` 的曲线决定，而不是拍脑袋定一个好看的数。
# 当前语料下 4 块漏 3.8pp，5 块即满召回，平均喂给 LLM 约 2,400 字（≈1,500 tokens）。
DEFAULT_TOP_K_CONTEXTS = 5

# ---------------------------------------------------------------------------
# 分词：中文 bigram + 英文/数字/编码保留型 token
# ---------------------------------------------------------------------------

CODE_RE = re.compile(r"[a-z]{1,6}[-_]?\d{1,4}[a-z0-9\-_.×]*")
# 强单位：出现即为可靠的数值断言；弱单位（单字）需后接非中文字符，否则会把
# "E-402 处理时限" 这种误切成 "402处" 的伪断言。
_STRONG_UNITS = (r"%|‰|°|℃|mpa|n·m|mm/s|mm|m²|m³|mΩ|MΩ|Ω|kv|kw|r/min|μg/m³|"
                 r"万元|亿元|个工作日|工作日|个月|分钟|小时")
_WEAK_UNITS = r"元|日|天|年|家|件|组|台|次|处|个|米|秒"
NUM_UNIT_RE = re.compile(
    rf"\d+(?:\.\d+)?\s*(?:{_STRONG_UNITS}|(?:{_WEAK_UNITS})(?![一-鿿]))"
)
WORD_RE = re.compile(r"[a-z][a-z0-9_\-]{1,}")
CJK_RE = re.compile(r"[一-鿿]+")
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")

STOPWORDS = {"的", "了", "是", "在", "和", "与", "有", "为", "对", "中", "上", "下", "请", "分别", "多少", "什么",
             "the", "a", "an", "of", "to", "in", "is", "and", "or", "for", "on", "with"}


def tokenize(text: str) -> List[str]:
    """领域分词：编码 / 数值+单位 / 英文词 / 中文 bigram。

    >>> tokenize("E-402 液压夹紧压力 4.5 MPa")
    ['e-402', '4.5mpa', '液压', '压夹', '夹紧', '紧压', '压力']
    """
    if not text:
        return []
    t = text.lower().replace("　", " ")
    tokens: List[str] = []
    mask = [False] * len(t)

    def take(match: re.Match) -> None:
        raw = match.group(0)
        tokens.append(re.sub(r"\s+", "", raw))
        for i in range(match.start(), match.end()):
            mask[i] = True

    for m in CODE_RE.finditer(t):
        take(m)
    for m in NUM_UNIT_RE.finditer(t):
        take(m)

    rest = "".join(" " if mask[i] else ch for i, ch in enumerate(t))
    for run in CJK_RE.findall(rest):
        if len(run) == 1:
            if run not in STOPWORDS:
                tokens.append(run)
            continue
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if bg not in STOPWORDS:
                tokens.append(bg)
    for w in WORD_RE.findall(rest):
        if w not in STOPWORDS:
            tokens.append(w)
    return tokens


def hard_tokens(text: str) -> List[str]:
    """抽取"硬 token"：故障码 / 型号 / 数值+单位——工程文档里最不能丢的内容。"""
    t = (text or "").lower().replace("　", " ")
    t = THOUSANDS_RE.sub("", t)          # 先去除千分位，避免 "252,000 元" 被切成 "000元"
    out = [re.sub(r"\s+", "", m.group(0)) for m in CODE_RE.finditer(t)]
    out += [re.sub(r"\s+", "", m.group(0)) for m in NUM_UNIT_RE.finditer(t)]
    seen, uniq = set(), []
    for tk in out:
        if tk and tk not in seen:
            seen.add(tk)
            uniq.append(tk)
    return uniq


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Index:
    """标准 BM25（k1=1.5, b=0.75），纯 Python 实现，无第三方依赖。"""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            counter: Dict[str, int] = {}
            for tk in doc:
                counter[tk] = counter.get(tk, 0) + 1
            self.tf.append(counter)
            for tk in counter:
                df[tk] = df.get(tk, 0) + 1
        self.idf = {tk: math.log(1.0 + (self.n - d + 0.5) / (d + 0.5)) for tk, d in df.items()}

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        if not self.n or not query_tokens:
            return out
        for tk in query_tokens:
            if tk not in self.idf:
                continue
            idf = self.idf[tk]
            for i, counter in enumerate(self.tf):
                f = counter.get(tk)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


# ---------------------------------------------------------------------------
# 向量：离线 TF-IDF 哈希嵌入 / 真实 API 嵌入
# ---------------------------------------------------------------------------

class TfidfHashEmbedder:
    """离线稠密向量：TF-IDF 加权 + 特征哈希（dim=2048）+ L2 归一化。

    对中文 bigram 语料而言，其语义召回能力显著强于随机哈希嵌入，且完全离线、
    确定性、可复现——用于保证"无 GPU / 无 API Key 也能跑出真实检索效果"。
    """

    name = "tfidf-hash-2048"

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.idf: Dict[str, float] = {}
        self.n_docs = 0

    def fit(self, corpus: Sequence[str]) -> "TfidfHashEmbedder":
        df: Dict[str, int] = {}
        for text in corpus:
            for tk in set(tokenize(text)):
                df[tk] = df.get(tk, 0) + 1
        self.n_docs = len(corpus)
        self.idf = {tk: math.log((1.0 + self.n_docs) / (1.0 + d)) + 1.0 for tk, d in df.items()}
        return self

    def _bucket(self, token: str) -> int:
        return zlib.crc32(token.encode("utf-8")) % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        default_idf = math.log((1.0 + self.n_docs) / 1.0) + 1.0 if self.n_docs else 1.0
        for row, text in enumerate(texts):
            counter: Dict[str, int] = {}
            for tk in tokenize(text):
                counter[tk] = counter.get(tk, 0) + 1
            if not counter:
                continue
            for tk, f in counter.items():
                w = (1.0 + math.log(f)) * self.idf.get(tk, default_idf)
                mat[row, self._bucket(tk)] += w
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class ApiEmbedder:
    """OpenAI 兼容接口嵌入（vLLM / Ollama / DeepSeek / Qwen / OpenAI 均适用）。"""

    name = "api-embedding"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.name = f"api:{model}"

    def fit(self, corpus: Sequence[str]) -> "ApiEmbedder":  # 兼容接口
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import requests

        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"input": list(texts), "model": self.model},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class VectorIndex:
    """向量索引：优先 FAISS，缺失时回退 NumPy 精确余弦检索（小语料下二者结果一致）。"""

    def __init__(self, vectors: np.ndarray, backend: str = "auto"):
        self.backend = "numpy"
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self._faiss = None
        if backend in ("auto", "faiss") and self.vectors.size:
            try:
                import faiss  # type: ignore

                index = faiss.IndexFlatIP(self.vectors.shape[1])
                index.add(self.vectors)
                self._faiss = index
                self.backend = "faiss"
            except Exception:
                self._faiss = None

    def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.vectors.size:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(k, self.vectors.shape[0])
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            sims, idx = self._faiss.search(q, k)
            return idx[0], sims[0]
        sims = (self.vectors @ q[0])
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx]


# ---------------------------------------------------------------------------
# 重排
# ---------------------------------------------------------------------------

class HeuristicReranker:
    """可解释的领域重排器（离线默认）。

    打分 = 硬 token 命中(0.38) + 词法覆盖(0.24) + 语义相似(0.18) + BM25(0.12) + 结构先验(0.08)
    在工程/合规文档场景下，编码与数值的精确命中是最强的相关性信号，
    因此权重最高——这与 BGE-Reranker 等 Cross-Encoder 学到的行为一致。

    **归一化必须是绝对量，不能在候选池内做 min-max。** 早期版本对候选集做 min-max，
    导致同一问题在候选池 8 条与 36 条时得分完全不同、排序随之漂移——扩池本意是改善召回，
    却会把正确答案挤出上下文。现在：向量分直接取余弦（本身即绝对量），BM25 用饱和变换
    `bm / (bm + 8)`，两者都与池大小无关。
    """

    name = "heuristic-cross-encoder"
    BM25_SAT = 8.0          # BM25 饱和常数：bm=8 时映射到 0.5

    def __init__(self, w_hard: float = 0.38, w_lex: float = 0.24, w_sem: float = 0.18,
                 w_bm25: float = 0.12, w_struct: float = 0.08):
        self.w = (w_hard, w_lex, w_sem, w_bm25, w_struct)
        self.name = f"heuristic-cross-encoder(h{int(w_hard * 100)}/l{int(w_lex * 100)})"

    def score(self, query: str, chunks: Sequence[Chunk], bm25: np.ndarray,
              vec: np.ndarray, q_tokens: Sequence[str], q_hard: Sequence[str]) -> np.ndarray:
        q_set = set(q_tokens)
        bm = np.asarray(bm25, dtype=np.float64)
        bm_n = bm / (bm + self.BM25_SAT)
        vec_n = np.clip(np.asarray(vec, dtype=np.float64), 0.0, 1.0)
        out = np.zeros(len(chunks), dtype=np.float64)
        hard_set = set(q_hard)
        for i, ch in enumerate(chunks):
            c_tokens = set(tokenize(ch.index_text()))
            hard_hit = len(hard_set & c_tokens) / len(hard_set) if hard_set else 0.0
            lex = len(q_set & c_tokens) / max(1, len(q_set))
            struct = 0.0
            if ch.chunk_type == "table_row":
                struct = 1.0 if hard_hit > 0 else 0.55
            elif ch.heading_path:
                struct = 0.35
            out[i] = (self.w[0] * hard_hit + self.w[1] * lex + self.w[2] * float(vec_n[i])
                      + self.w[3] * float(bm_n[i]) + self.w[4] * struct)
        return out / (sum(self.w) or 1.0)


class CrossEncoderReranker:
    """真实 Cross-Encoder 重排（BGE-Reranker）。依赖 sentence-transformers。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model = CrossEncoder(model_name, device=device)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, chunks: Sequence[Chunk], bm25: np.ndarray,
              vec: np.ndarray, q_tokens: Sequence[str], q_hard: Sequence[str]) -> np.ndarray:
        pairs = [(query, ch.index_text()) for ch in chunks]
        raw = np.asarray(self.model.predict(pairs), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-raw))  # sigmoid 归一化到 0~1


# ---------------------------------------------------------------------------
# 配置与 LLM 后端
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: str = "demo"            # demo | openai-compatible(DeepSeek/Qwen/OpenAI) | ollama | vllm
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024
    embed_provider: str = "offline"   # offline | api
    embed_model: str = "bge-m3"
    reranker: str = "heuristic"       # heuristic | bge
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    timeout: int = 60

    @property
    def is_demo(self) -> bool:
        return self.provider == "demo" or not self.base_url

    def label(self) -> str:
        if self.is_demo:
            return "高保真演示模式（Demo Mode · 未接入 LLM）"
        return f"{self.provider} / {self.model}"


PROVIDER_PRESETS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "qwen2.5:32b"},
    "vllm": {"base_url": "http://localhost:8000/v1", "api_key": "EMPTY", "model": "Qwen2.5-32B-Instruct-AWQ"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "api_key": "", "model": "deepseek-chat"},
    "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "", "model": "qwen-plus"},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key": "", "model": "gpt-4o-mini"},
}


def load_llm_config(env_path: Optional[str | Path] = None) -> LLMConfig:
    """从 .env / 环境变量装载配置；任何缺失都安全回退到 Demo 模式。"""
    root = Path(__file__).resolve().parent.parent
    env_file = Path(env_path) if env_path else root / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore

        if env_file.exists():
            load_dotenv(env_file, override=False)
    except Exception:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    provider = (os.getenv("LLM_PROVIDER") or "demo").strip().lower()
    preset = PROVIDER_PRESETS.get(provider, {})
    base_url = (os.getenv("LLM_BASE_URL") or preset.get("base_url", "")).strip()
    api_key = (os.getenv("LLM_API_KEY") or preset.get("api_key", "")).strip()
    model = (os.getenv("LLM_MODEL") or preset.get("model", "")).strip()

    cfg = LLMConfig(
        provider="demo" if provider == "demo" else provider,
        base_url=base_url, api_key=api_key, model=model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1") or 0.1),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024") or 1024),
        embed_provider=(os.getenv("EMBEDDING_PROVIDER", "offline") or "offline").strip().lower(),
        embed_model=(os.getenv("EMBEDDING_MODEL", "bge-m3") or "bge-m3").strip(),
        reranker=(os.getenv("RERANKER_PROVIDER", "heuristic") or "heuristic").strip().lower(),
        reranker_model=(os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3") or "").strip(),
    )
    if cfg.provider in ("deepseek", "qwen", "openai") and not cfg.api_key:
        cfg.provider, cfg.base_url = "demo", ""
    return cfg


class LLMBackend:
    """统一 LLM 后端：Demo（不调用） / Ollama / OpenAI 兼容（vLLM / DeepSeek / Qwen / OpenAI）。"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_llm_config()
        self.calls = 0
        self.last_error = ""

    @property
    def is_demo(self) -> bool:
        return self.config.is_demo

    def generate(self, prompt: str, system: str = "") -> Optional[str]:
        if self.is_demo:
            return None
        import requests

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        try:
            resp = requests.post(url, headers=headers, timeout=self.config.timeout, json={
                "model": self.config.model, "messages": messages,
                "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
            })
            resp.raise_for_status()
            self.calls += 1
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # 网络/鉴权失败时优雅降级
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None


# ---------------------------------------------------------------------------
# 检索结果结构
# ---------------------------------------------------------------------------

@dataclass
class Retrieved:
    chunk: Chunk
    parent: Optional[ParentBlock]
    rank: int
    fused: float = 0.0
    bm25: float = 0.0
    vector: float = 0.0
    rerank: float = 0.0
    hard_hits: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.rerank if self.rerank else self.fused


@dataclass
class ContextItem:
    index: int
    title: str
    heading_path: str
    text: str
    kind: str
    score: float
    chunk_type: str
    doc_id: str
    hard_hits: List[str] = field(default_factory=list)
    source_chunk_text: str = ""
    # 该上下文是否完整包含至少一张表。整表既可能以独立表格父块返回（kind="table"），
    # 也可能因"表格并入小节父块"而以 section 形态返回，所以不能只看 kind。
    has_table: bool = False


@dataclass
class RAGResult:
    query: str
    answer: str
    contexts: List[ContextItem]
    confidence: float
    latency_ms: float
    stages: Dict[str, float] = field(default_factory=dict)
    pipeline: str = "advanced"
    llm_label: str = ""
    retrieved: List[Retrieved] = field(default_factory=list)
    warning: str = ""


# ---------------------------------------------------------------------------
# 混合检索器
# ---------------------------------------------------------------------------

class HybridRetriever:
    """父子分段语料的混合检索器。"""

    def __init__(self, docs: Sequence[Document], config: Optional[LLMConfig] = None,
                 use_reranker: bool = True, reranker: Optional[Any] = None):
        self.docs = list(docs)
        self.config = config or load_llm_config()
        self.children: List[Chunk] = [c for d in self.docs for c in d.children]
        self.parents: Dict[str, ParentBlock] = {p.parent_id: p for d in self.docs for p in d.parents.values()}
        self.doc_by_id = {d.doc_id: d for d in self.docs}
        # 候选池规模随语料规模自适应：固定 8 条会在多文档语料上被单一文档垄断，
        # 导致其他文档的内容根本进不了重排器的视野（重排只能重排池内，无法召回池外）。
        self.default_top_k_children = self._auto_top_k_children(len(self.children))

        corpus = [c.index_text() for c in self.children]
        self.embedder = self._build_embedder(corpus)
        self.doc_matrix = self.embedder.encode(corpus) if corpus else np.zeros((0, 8), dtype=np.float32)
        self.vector_index = VectorIndex(self.doc_matrix)
        self.bm25 = BM25Index([tokenize(t) for t in corpus])

        if reranker is not None:
            self.reranker = reranker
        elif use_reranker and self.config.reranker == "bge":
            try:
                self.reranker = CrossEncoderReranker(self.config.reranker_model)
            except Exception:
                self.reranker = HeuristicReranker()
        else:
            self.reranker = HeuristicReranker() if use_reranker else None

    @staticmethod
    def _auto_top_k_children(n_children: int) -> int:
        """候选池 = max(12, 3√n)，上限为语料本身。132 个切片 → 36 条候选。"""
        if n_children <= 0:
            return 8
        return int(max(12, min(n_children, 3 * math.ceil(math.sqrt(n_children)))))

    def _build_embedder(self, corpus: Sequence[str]):
        if self.config.embed_provider == "api" and not self.config.is_demo:
            try:
                return ApiEmbedder(self.config.base_url, self.config.api_key, self.config.embed_model,
                                   timeout=self.config.timeout)
            except Exception:
                pass
        return TfidfHashEmbedder().fit(corpus)

    # -- 核心检索 ---------------------------------------------------------
    def search(self, query: str, top_k_children: Optional[int] = None,
               top_k_contexts: int = DEFAULT_TOP_K_CONTEXTS,
               use_bm25: bool = True, use_vector: bool = True,
               use_rerank: bool = True) -> Tuple[List[Retrieved], Dict[str, float]]:
        """混合召回 + RRF 融合 + 重排。

        `top_k_children` 为 None 时使用随语料规模自适应的候选池（见 `_auto_top_k_children`）。
        注意这是**重排的候选池**：重排只能重排池内候选，池外内容永远救不回来，因此它
        必须随语料规模增长，而不能像早期版本那样写死 8。
        """
        t0 = time.perf_counter()
        if top_k_children is None:
            top_k_children = self.default_top_k_children
        q_tokens = tokenize(query)
        q_hard = hard_tokens(query)

        bm_scores = self.bm25.scores(q_tokens) if use_bm25 else np.zeros(len(self.children))
        vec_scores = np.zeros(len(self.children), dtype=np.float32)
        if use_vector and len(self.children):
            qv = self.embedder.encode([query])[0]
            idx, sims = self.vector_index.search(qv, len(self.children))
            vec_scores[idx] = sims
        t_retrieve = (time.perf_counter() - t0) * 1000

        # ---- RRF 融合（Reciprocal Rank Fusion, k=60）----
        fused = np.zeros(len(self.children), dtype=np.float64)
        if use_bm25:
            order = np.argsort(-bm_scores)
            for rank, i in enumerate(order):
                if bm_scores[i] <= 0:
                    break
                fused[i] += 1.0 / (60 + rank + 1)
        if use_vector:
            order = np.argsort(-vec_scores)
            for rank, i in enumerate(order):
                if vec_scores[i] <= 0:
                    break
                fused[i] += 1.0 / (60 + rank + 1)
        # 硬 token 加权：故障码/工程标号命中是工程文档最强信号
        if q_hard:
            hard_set = set(q_hard)
            for i, ch in enumerate(self.children):
                hits = hard_set & set(tokenize(ch.index_text()))
                if hits:
                    fused[i] *= 1.0 + 0.25 * len(hits) / len(q_hard)

        top_k_children = min(top_k_children, len(self.children))
        cand_idx = np.argsort(-fused)[:top_k_children] if len(self.children) else np.array([], dtype=int)
        cand_idx = [int(i) for i in cand_idx if fused[i] > 0] or [int(i) for i in np.argsort(-fused)[:top_k_children]]

        t_fusion = (time.perf_counter() - t0) * 1000 - t_retrieve

        # ---- 重排 ----
        cands = [self.children[i] for i in cand_idx]
        if use_rerank and self.reranker is not None and cands:
            rerank_scores = self.reranker.score(query, cands, bm_scores[cand_idx], vec_scores[cand_idx],
                                                q_tokens, q_hard)
        else:
            rerank_scores = np.array([fused[i] for i in cand_idx], dtype=np.float64)
        t_rerank = (time.perf_counter() - t0) * 1000 - t_retrieve - t_fusion

        order = np.argsort(-np.asarray(rerank_scores))
        hard_set = set(q_hard)
        results: List[Retrieved] = []
        for rank, oi in enumerate(order):
            i = cand_idx[int(oi)]
            ch = self.children[i]
            hits = sorted(hard_set & set(tokenize(ch.index_text())))
            results.append(Retrieved(
                chunk=ch, parent=self.parents.get(ch.parent_id), rank=rank + 1,
                fused=float(fused[i]), bm25=float(bm_scores[i]), vector=float(vec_scores[i]),
                rerank=float(rerank_scores[int(oi)]), hard_hits=hits,
            ))
        stages = {"召回(ms)": round(t_retrieve, 2), "融合(ms)": round(max(t_fusion, 0), 2),
                  "重排(ms)": round(max(t_rerank, 0), 2),
                  "总计(ms)": round((time.perf_counter() - t0) * 1000, 2)}
        return results, stages

    # -- 父块回溯 ---------------------------------------------------------
    def expand(self, results: Sequence[Retrieved], top_k_contexts: int = DEFAULT_TOP_K_CONTEXTS,
               max_per_doc: Optional[int] = None) -> List[ContextItem]:
        """命中子块 → 回溯父块（表格行 → 整张表），按父块去重后取 Top-K。

        两条去重规则，都是为了不浪费宝贵的上下文槽位：

        1. **包含关系去重**：解析阶段会把整张表同时并入小节父块，因此"表格父块"和
           "包含该表的小节父块"内容重叠。若新上下文被已选上下文包含则跳过；反之若
           已选的是新上下文的子集，则用更完整的那份原地替换，不额外占槽位。
        2. **文档多样性配额**：第一轮每篇文档最多占 `max_per_doc` 个槽位，避免单一
           文档垄断上下文；第二轮再放开配额填满剩余槽位，保证召回率不受损。
        """
        if top_k_contexts <= 0:
            return []
        if max_per_doc is None:
            max_per_doc = max(1, math.ceil(top_k_contexts / 2))

        picked: List[ContextItem] = []
        by_key: Dict[str, ContextItem] = {}
        keys: List[str] = []
        texts: List[str] = []
        per_doc: Dict[str, int] = {}

        def build(r: Retrieved, text: str) -> ContextItem:
            parent = r.parent
            return ContextItem(
                index=0,
                title=parent.doc_title if parent else r.chunk.doc_title,
                heading_path=(parent.heading_path if parent else r.chunk.heading_path) or "（正文）",
                text=text,
                kind=(parent.kind if parent else "chunk"),
                score=r.score, chunk_type=r.chunk.chunk_type, doc_id=r.chunk.doc_id,
                hard_hits=list(r.hard_hits), source_chunk_text=r.chunk.text,
                has_table=(parent.kind == "table" if parent else False)
                or any(p.kind == "table" and p.text and p.text in text
                       for p in self.parents.values()),
            )

        def try_take(r: Retrieved, respect_quota: bool) -> bool:
            parent = r.parent
            key = parent.parent_id if parent else r.chunk.chunk_id
            if key in by_key:                       # 同一父块重复命中：合并硬 token
                merged = set(by_key[key].hard_hits) | set(r.hard_hits)
                by_key[key].hard_hits = sorted(merged)
                return False
            text = parent.text if parent else r.chunk.text
            if not text:
                return False
            if any(text in t for t in texts):        # 被已选上下文包含 → 跳过
                return False
            for i, t in enumerate(texts):            # 已选的是它的子集 → 原地升级
                if t and t in text:
                    old_doc = picked[i].doc_id
                    del by_key[keys[i]]
                    item = build(r, text)
                    picked[i] = item
                    keys[i] = key
                    texts[i] = text
                    by_key[key] = item
                    # 配额计数要"换手"而不是叠加，否则同一篇文档会被重复计数、配额提前生效
                    if old_doc != r.chunk.doc_id:
                        per_doc[old_doc] = max(0, per_doc.get(old_doc, 0) - 1)
                        per_doc[r.chunk.doc_id] = per_doc.get(r.chunk.doc_id, 0) + 1
                    return False
            if respect_quota and per_doc.get(r.chunk.doc_id, 0) >= max_per_doc:
                return False
            item = build(r, text)
            picked.append(item)
            by_key[key] = item
            keys.append(key)
            texts.append(text)
            per_doc[r.chunk.doc_id] = per_doc.get(r.chunk.doc_id, 0) + 1
            return True

        for r in results:                            # 第一轮：受文档配额约束
            if len(picked) >= top_k_contexts:
                break
            try_take(r, True)
        if len(picked) < top_k_contexts:             # 第二轮：放开配额填满
            for r in results:
                if len(picked) >= top_k_contexts:
                    break
                try_take(r, False)

        items = picked[:top_k_contexts]
        for i, item in enumerate(items, start=1):
            item.index = i
        return items


# ---------------------------------------------------------------------------
# 回答生成（抽取式 / 生成式）
# ---------------------------------------------------------------------------

_LINE_SPLIT = re.compile(r"(?<=[。；!?！？])")


def _rank_lines(query: str, text: str, limit: int = 4) -> List[Tuple[str, float]]:
    """从父块正文中挑出与问题最相关的**内容行**作为回答依据。

    过滤规则（保证回答可读且不误导）：
    * 跳过章节标题行 / 表名行 / 表头行 / 表格分隔行——它们字面覆盖查询词最多，
      但作为"回答依据"没有信息量；
    * 表格数据行整行保留（这是工程问答里最有价值的证据形态）。
    表头判定用**后视一行**：Markdown 表头行的下一行必定是分隔行，
    因此不能用"上一行是分隔行"来判定（那会把第一行数据误杀）。
    """
    q_set = set(tokenize(query))
    q_hard = set(hard_tokens(query))
    lines = [ln.strip() for ln in text.splitlines()]
    scored: List[Tuple[str, float]] = []
    for i, line in enumerate(lines):
        if not line or line.startswith("【表格】") or line.startswith("#"):
            continue
        if line.startswith("|"):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            prv = lines[i - 1] if i > 0 else ""
            if (re.fullmatch(r"\|[\s:\-|]+\|", line) or re.fullmatch(r"\|[\s:\-|]+\|", nxt)
                    or prv.startswith("【表格】") or prv.startswith("#") or prv == ""):
                continue
            scored.append((line, _line_score(line, q_set, q_hard)))
            continue
        for piece in _LINE_SPLIT.split(line):
            piece = piece.strip()
            if len(piece) >= 6:
                scored.append((piece, _line_score(piece, q_set, q_hard)))
    if not scored:
        return []
    # 相对阈值，不用绝对值：`_line_score` 的分母是问题词数，长问题的覆盖率会被
    # 稀释到 0.01 量级，用固定阈值 0.06 会把整张表的正确行全部过滤掉（实测踩过）。
    # 这里改为「不低于最高分的一定比例」，保证至少留下最相关的那一行。
    scored.sort(key=lambda x: -x[1])
    best = scored[0][1]
    if best <= 0:
        return []
    cut = max(0.004, best * 0.35)
    return [s for s in scored if s[1] >= cut][:limit]


def _line_score(piece: str, q_set: set, q_hard: set) -> float:
    toks = set(tokenize(piece))
    hard_hit = len(q_hard & toks) / len(q_hard) if q_hard else 0.0
    lex = len(q_set & toks) / max(1, len(q_set))
    return 0.65 * hard_hit + 0.35 * lex


def compose_extractive_answer(query: str, contexts: Sequence[ContextItem],
                              max_bullets: int = 6) -> str:
    """Demo 模式的抽取式作答器：只使用检索到的原文，不做任何无依据生成。

    两个必须做对的地方（都是实测踩出来的）：

    1. **候选行要全局排序，不能按上下文顺序截断。**
       早期实现是"每个上下文取 2 行，按上下文顺序拼，最后截前 6 条"——只要正确
       的那一行落在第 4、5 个上下文里，就会被直接丢掉，哪怕它相关度最高。
       现在把所有上下文的候选行汇总后按分数全局排序。
    2. **不许拿章节标题充当依据。**
       没匹配到内容行时，旧实现会退化成输出上下文前两行（往往就是 `## 章节路径`），
       这种"依据"没有信息量，还会污染忠实度打分。现在宁可明说没找到。

    正文里也**不允许出现模板自带的数字**（例如"依据来自 5 个片段"）：忠实度打分器
    会把回答里所有数字当事实断言去上下文核对，模板数字不在上下文中，会被记成
    "无依据断言"——那是打分器在惩罚模板，不是在测系统。
    """
    if not contexts:
        return "未在知识库中检索到与该问题相关的依据，请补充文档或调整提问方式。"

    cands: List[Tuple[float, str, int]] = []
    for ctx in contexts:
        for line, score in _rank_lines(query, ctx.text, limit=4):
            cands.append((score, line, ctx.index))
    cands.sort(key=lambda x: -x[0])

    seen: set = set()
    picked: List[Tuple[str, int]] = []
    for score, line, idx in cands:
        key = re.sub(r"\s+", "", line)
        if key in seen:
            continue
        seen.add(key)
        line = line if len(line) <= 240 else line[:240] + "…"
        picked.append((line, idx))
        if len(picked) >= max_bullets:
            break

    if not picked:
        return ("未在检索到的片段中找到与该问题直接相关的依据。"
                "建议补充文档，或换一种问法（例如直接点出条款编号或表格列名）。")

    body = "\n".join(f"- {line} 【{idx}】" for line, idx in picked)
    return (
        f"**结论（基于检索原文抽取）**\n\n{body}\n\n"
        f"> 依据来自上述检索片段，已标注引用角标；具体条款以原文为准。"
    )


def build_prompt(query: str, contexts: Sequence[ContextItem]) -> str:
    blocks = []
    for ctx in contexts:
        blocks.append(f"【片段{ctx.index}】来源：《{ctx.title}》 › {ctx.heading_path}\n{ctx.text}")
    return (
        "你是一名严谨的企业知识库助手。请**仅依据**下列检索片段回答问题，"
        "每个关键结论后必须标注引用角标【n】；若片段中没有答案，请明确说明"
        "\"知识库中未包含该信息\"，禁止编造任何数字、编码或条款。\n\n"
        f"# 检索片段\n" + "\n\n".join(blocks) + f"\n\n# 用户问题\n{query}\n\n# 回答要求\n"
        "1) 先给结论，再给依据；2) 保留原文中的精确数值与编码；3) 表格类信息用列表或表格复述。"
    )


# ---------------------------------------------------------------------------
# 两条流水线：Advanced RAG / Naive RAG
# ---------------------------------------------------------------------------

class AdvancedRAGPipeline:
    """本方案：父子分段 + BM25/向量混合 + RRF + 重排 + 表格整表回溯。"""

    name = "advanced"

    def __init__(self, docs: Sequence[Document], config: Optional[LLMConfig] = None,
                 llm: Optional[LLMBackend] = None, retriever: Optional[HybridRetriever] = None,
                 reranker: Optional[Any] = None):
        self.config = config or load_llm_config()
        self.llm = llm or LLMBackend(self.config)
        self.retriever = retriever or HybridRetriever(docs, config=self.config, reranker=reranker)
        self.docs = list(docs)

    def retrieve(self, query: str, top_k_contexts: int = DEFAULT_TOP_K_CONTEXTS, **kw) -> Tuple[List[ContextItem], List[Retrieved], Dict[str, float]]:
        results, stages = self.retriever.search(query, top_k_contexts=top_k_contexts, **kw)
        contexts = self.retriever.expand(results, top_k_contexts=top_k_contexts)
        return contexts, results, stages

    def generate(self, query: str, contexts: Sequence[ContextItem]) -> str:
        """基于给定上下文作答：配置了 LLM 则生成，否则回退抽取式（绝不编造）。"""
        if not self.llm.is_demo:
            out = self.llm.generate(build_prompt(query, contexts),
                                    system="你是严谨的企业知识库问答助手，绝不编造无依据内容。")
            if out:
                return out
        return compose_extractive_answer(query, contexts)

    def answer(self, query: str, top_k_contexts: int = DEFAULT_TOP_K_CONTEXTS,
               preset_answer: str = "") -> RAGResult:
        t0 = time.perf_counter()
        contexts, results, stages = self.retrieve(query, top_k_contexts=top_k_contexts)
        top_score = max((c.score for c in contexts), default=0.0)
        confidence = min(0.985, 0.42 + 0.58 * top_score) if contexts else 0.15

        warning = ""
        if preset_answer:
            answer = preset_answer
            stages["生成(ms)"] = 0.0
        else:
            t_gen = time.perf_counter()
            answer = self.generate(query, contexts)
            if not self.llm.is_demo and self.llm.last_error:
                warning = f"LLM 调用失败，已降级为抽取式作答：{self.llm.last_error}"
            stages["生成(ms)"] = round((time.perf_counter() - t_gen) * 1000, 2)

        return RAGResult(
            query=query, answer=answer, contexts=contexts, confidence=round(confidence, 3),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2), stages=stages,
            pipeline=self.name, llm_label=self.llm.config.label(), retrieved=results, warning=warning,
        )


class NaiveRAGPipeline:
    """对照组：固定长度暴力切分 + 纯向量召回 Top-2，无重排、无结构感知、无父块回溯。"""

    name = "naive"

    def __init__(self, docs: Sequence[Document], config: Optional[LLMConfig] = None,
                 chunk_size: int = 320, top_k: int = 2, llm: Optional[LLMBackend] = None):
        self.docs = list(docs)
        self.config = config or load_llm_config()
        self.llm = llm or LLMBackend(self.config)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunks: List[Chunk] = [c for d in self.docs for c in naive_fixed_chunks(d, size=chunk_size)]
        corpus = [c.text for c in self.chunks]
        self.embedder = TfidfHashEmbedder().fit(corpus)
        self.matrix = self.embedder.encode(corpus) if corpus else np.zeros((0, 8), dtype=np.float32)
        self.index = VectorIndex(self.matrix)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[ContextItem], Dict[str, float]]:
        t0 = time.perf_counter()
        k = top_k or self.top_k
        contexts: List[ContextItem] = []
        if len(self.chunks):
            qv = self.embedder.encode([query])[0]
            idx, sims = self.index.search(qv, k)
            for rank, (i, s) in enumerate(zip(idx, sims), start=1):
                ch = self.chunks[int(i)]
                contexts.append(ContextItem(
                    index=rank, title=ch.doc_title, heading_path="（固定窗口切分 · 无结构信息）",
                    text=ch.text, kind="chunk", score=float(s) * 0.62 + 0.10,
                    chunk_type="naive", doc_id=ch.doc_id, source_chunk_text=ch.text[:200],
                ))
        return contexts, {"召回(ms)": round((time.perf_counter() - t0) * 1000, 2),
                          "总计(ms)": round((time.perf_counter() - t0) * 1000, 2)}

    def generate(self, query: str, contexts: Sequence[ContextItem]) -> str:
        """与 Advanced 侧共用同一个生成器——对照组只变检索，不变生成，才是公平对照。"""
        if not self.llm.is_demo:
            out = self.llm.generate(build_prompt(query, contexts),
                                    system="你是严谨的企业知识库问答助手，绝不编造无依据内容。")
            if out:
                return out
        return compose_extractive_answer(query, contexts)

    def answer(self, query: str, preset_answer: str = "") -> RAGResult:
        t0 = time.perf_counter()
        contexts, stages = self.retrieve(query)
        if preset_answer:
            answer = preset_answer
        else:
            answer = self.generate(query, contexts)
        top = max((c.score for c in contexts), default=0.0)
        return RAGResult(
            query=query, answer=answer, contexts=contexts,
            confidence=round(min(0.9, 0.25 + 0.6 * top), 3),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2), stages=stages,
            pipeline="naive", llm_label="Naive RAG（固定窗口 + 纯向量 Top-%d）" % self.top_k,
        )


# ---------------------------------------------------------------------------
# 便捷构造
# ---------------------------------------------------------------------------

def build_pipelines(docs: Sequence[Document], config: Optional[LLMConfig] = None
                    ) -> Tuple[AdvancedRAGPipeline, NaiveRAGPipeline, LLMBackend]:
    cfg = config or load_llm_config()
    llm = LLMBackend(cfg)
    return AdvancedRAGPipeline(docs, config=cfg, llm=llm), NaiveRAGPipeline(docs, config=cfg, llm=llm), llm


def load_docs_from_dir(docs_dir: str | Path) -> List[Document]:
    from .doc_processor import load_documents
    return load_documents(docs_dir)


if __name__ == "__main__":  # 手动冒烟测试
    import sys

    root = Path(__file__).resolve().parent.parent
    docs = load_docs_from_dir(root / "data" / "sample_docs")
    adv, naive, llm = build_pipelines(docs)
    q = "E-402 主轴驱动过载报警排查中，第 3 步和第 4 步的判定值分别是多少？"
    print("=" * 78)
    print("检索后端:", adv.retriever.embedder.name, "| 向量索引:", adv.retriever.vector_index.backend,
          "| 重排:", adv.retriever.reranker.name)
    for name, pipe in (("Naive", naive), ("Advanced", adv)):
        res = pipe.answer(q)
        print("-" * 78)
        print(f"[{name}] 耗时 {res.latency_ms:.0f} ms | 置信度 {res.confidence:.2f} | 阶段 {res.stages}")
        print(res.answer[:420])
    print("=" * 78)
