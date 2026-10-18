# -*- coding: utf-8 -*-
"""OnPrem-RAG-Kit 核心引擎包。

模块:
    doc_processor    结构化文档解析与父子分段（Parent-Child Chunking）
    hybrid_retriever BM25 + 向量混合检索与 Rerank 重排引擎
    evaluator        自动化评测（Naive RAG vs Advanced RAG）
    tco_calculator   企业私有化部署算力选型与三年 TCO/ROI 财务测算
"""

__version__ = "1.0.0"
__all__ = ["doc_processor", "hybrid_retriever", "evaluator", "tco_calculator"]
