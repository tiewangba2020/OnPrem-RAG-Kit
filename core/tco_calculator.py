# -*- coding: utf-8 -*-
"""core.tco_calculator —— 企业私有化部署算力选型与三年 TCO / ROI 财务测算模型

模型边界（诚实披露）
--------------------
* 三条成本曲线：① 商用闭源 API 按量付费；② 本地私有化服务器自建；③ 混合云（本地基线 + 云端弹性溢出）。
* 本地私有化的成本**不止硬件**：一次性实施费（知识库工程、集成联调、安全加固、培训）、
  机房托管、电费、运维人力、模型迭代与维保全部计入，因此结论比"硬件回本"口径保守得多。
* 云端 API 价格按年下降（默认 15%），业务量按年增长（默认 15%），两个因素对本地化并不天然有利。
* 效率收益（排查/查条款耗时下降）属于**影子价值**，单列展示并按折现系数计入，不混入基础设施成本对比。
* 全部价格为公开市场参考价（2025-2026 中国大陆区间），可在参数面板中按实际报价覆盖。

结论口径：默认场景下输出「回本临界点（第 N 个月）」与「三年综合成本节约率」，
并附**敏感性分析**——说明在什么条件下私有化更省、什么条件下应当继续用 API。
"""
from __future__ import annotations

import csv
import io
import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 目录：API 模型 / GPU / 模型档位 / 平台
# ---------------------------------------------------------------------------

API_MODELS: Dict[str, Dict[str, Any]] = {
    "deepseek-v3": {"name": "DeepSeek-V3（高性价比闭源 API）", "vendor": "深度求索",
                    "price_in": 2.0, "price_out": 8.0,
                    "note": "国内价格最低档，适合成本敏感型场景；是私有化经济性对比中最严苛的基准"},
    "qwen-plus": {"name": "通义千问 Qwen-Plus", "vendor": "阿里云",
                  "price_in": 0.8, "price_out": 2.0, "note": "轻量档，适合简单问答"},
    "qwen-max": {"name": "通义千问 Qwen-Max", "vendor": "阿里云",
                 "price_in": 20.0, "price_out": 60.0, "note": "国产旗舰档"},
    "gpt-4o-mini": {"name": "GPT-4o mini", "vendor": "OpenAI",
                    "price_in": 1.1, "price_out": 4.3, "note": "海外轻量档（含汇率折算）"},
    "gpt-4o": {"name": "GPT-4o（企业级旗舰基线）", "vendor": "OpenAI",
               "price_in": 18.0, "price_out": 72.0, "note": "企业知识问答最常用的对标基线"},
    "enterprise-flag": {"name": "企业级旗舰长上下文模型", "vendor": "多云",
                        "price_in": 36.0, "price_out": 144.0,
                        "note": "128K+ 长上下文旗舰档（如 GPT-4-turbo / 云厂商旗舰实例）"},
    "private-cloud-gpu": {"name": "云上私有化 GPU 实例（租赁制）", "vendor": "云厂商",
                          "price_in": 0.0, "price_out": 0.0,
                          "note": "按卡时租赁（A10 级 ¥6.5/小时），非 token 计费"},
}

# 云 GPU 实例按需/包月参考价（元/月/卡，含基础存储与带宽）
CLOUD_GPU_MONTHLY = {"a10": 4200.0, "l40s": 6800.0, "rtx4090": 3200.0, "a100": 11500.0}

GPU_CATALOG: Dict[str, Dict[str, Any]] = {
    "rtx4090": {"name": "NVIDIA RTX 4090 24GB", "price": 16800, "tdp": 450, "vram": 24,
                "tps": {"7b": 2400, "14b": 1450, "32b": 820, "70b": 190},
                "note": "性价比之王；消费级卡无 NVLink，TP=2 时 PCIe 带宽会成为瓶颈（实测损耗约 15%）"},
    "a10": {"name": "NVIDIA A10 24GB", "price": 52000, "tdp": 150,
            "tps": {"7b": 1600, "14b": 950, "32b": 560, "70b": 130}, "vram": 24,
            "note": "企业级推理卡，被动散热可上 2U 服务器，长期高负载稳定性更好"},
    "l40s": {"name": "NVIDIA L40S 48GB", "price": 85000, "tdp": 350,
             "tps": {"7b": 3200, "14b": 2100, "32b": 1250, "70b": 320}, "vram": 48,
             "note": "单卡显存即可承载 32B-AWQ，无需张量并行，运维最简单"},
    "a100x": {"name": "NVIDIA A100 80GB", "price": 155000, "tdp": 400,
              "tps": {"7b": 4200, "14b": 2900, "32b": 1800, "70b": 620}, "vram": 80,
              "note": "训练/推理通用，支持 NVLink；预算充足且需微调时的首选"},
    "h20": {"name": "NVIDIA H20 96GB", "price": 110000, "tdp": 400,
            "tps": {"7b": 5000, "14b": 3400, "32b": 2200, "70b": 780}, "vram": 96,
            "note": "合规供应版，96GB 显存适合长上下文与高并发"},
    "ascend910b": {"name": "昇腾 910B 64GB", "price": 105000, "tdp": 400,
                   "tps": {"7b": 3400, "14b": 2300, "32b": 1500, "70b": 520}, "vram": 64,
                   "note": "信创/国产化合规首选，需 CANN + MindIE 适配，生态迁移成本需计入"},
}

MODEL_TIERS: Dict[str, Dict[str, Any]] = {
    "7b": {"name": "Qwen2.5-7B-Instruct-AWQ", "weights_gb": 5.5,
           "desc": "轻量档：单轮事实型问答、意图识别、路由分发"},
    "14b": {"name": "Qwen2.5-14B-Instruct-AWQ", "weights_gb": 9.8,
            "desc": "主力性价比档：复杂表格理解与条款抽取表现良好"},
    "32b": {"name": "Qwen2.5-32B-Instruct-AWQ", "weights_gb": 19.5,
            "desc": "企业级精度档：跨条款推理、长文档理解、引用溯源最稳"},
    "70b": {"name": "Llama-3.1-70B-Instruct-AWQ", "weights_gb": 40.0,
            "desc": "旗舰档：多语言与复杂推理，需 2-4 卡张量并行"},
}

PLATFORM_COST_BY_GPU = {1: 18000, 2: 32000, 4: 58000, 8: 105000}   # 整机平台（CPU/内存/主板/电源/机箱/系统盘）
STORAGE_NVME_PER_TB = 2600.0     # 元/TB（企业级 NVMe）
STORAGE_BACKUP_PER_TB = 800.0    # 元/TB（备份存储）
NETWORK_SWITCH_COST = 12000.0    # 万兆交换机 + 光模块
SPARE_PARTS_RATIO = 0.05         # 备件计提（按硬件成本）
PRICE_KWH = 0.85                 # 元/度（工商业电价）
PUE = 1.45                       # 机房能效比
COLOCATION_PER_CABINET_MONTH = 3000.0
PLATFORM_BASE_W = 400            # 单服务器平台基础功耗
WORK_HOURS_PER_DAY = 8.0
MONTHS = 36

SECRECY_LEVELS = ["公开", "内部", "机密", "绝密"]
SECRECY_SECURITY_COST = {"公开": 0.0, "内部": 20000.0, "机密": 50000.0, "绝密": 90000.0}
SECRECY_AUDIT_MONTHLY = {"公开": 0.0, "内部": 500.0, "机密": 2500.0, "绝密": 5000.0}
SECRECY_COMPLIANCE_NOTE = {
    "公开": "无特殊要求，三种方案均可自由选择",
    "内部": "建议混合云：非敏感问答走 API，敏感知识走本地",
    "机密": "建议全本地私有化，混合云需部署脱敏网关并留存审计日志",
    "绝密": "强制全本地离线部署，禁止任何数据出域（API / 混合云方案不适用）",
}


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------

@dataclass
class TCOInputs:
    # 业务负载
    employees: int = 3000
    queries_per_user_day: float = 5.0
    tokens_per_query: int = 4000
    input_ratio: float = 0.7
    doc_gb: float = 200.0
    secrecy_level: str = "机密"
    api_model: str = "gpt-4o"
    workdays_per_month: int = 22
    # 工程假设
    peak_factor: float = 2.5
    headroom: float = 1.6
    usage_growth: float = 0.15
    api_price_decline: float = 0.15
    redundancy: str = "cold"              # none | cold | hot
    # 成本假设
    ops_fte: float = 0.5
    ops_salary_wan: float = 18.0
    hardware_discount: float = 1.0        # 实际采购折扣系数（大客户议价）
    overflow_ratio: float = 0.25          # 混合云中走云端的流量占比
    hy_opex_factor: float = 1.0
    monthly_iter_cost: float = 5000.0     # 模型迭代与评测（元/月），可按团队实际投入调整
    # 效率收益（影子价值）
    deep_case_ratio: float = 0.15
    deep_save_min: float = 40.0
    light_save_min: float = 3.0
    hourly_cost: float = 45.0
    benefit_discount: float = 0.5         # 价值实现率：节省的工时中真正转化为现金收益的比例
    adoption_ramp_months: int = 12        # 采纳率爬坡：第 m 月采纳率 = min(1, m / ramp)
    max_lookup_hours_per_day: float = 0.5  # 人均每天在知识检索上的总工时上限（用于封顶）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TCOInputs":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def with_overrides(self, **kw: Any) -> "TCOInputs":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# 负载测算
# ---------------------------------------------------------------------------

@dataclass
class Workload:
    daily_queries: float
    monthly_queries: float
    monthly_tokens: float
    monthly_in_tokens: float
    monthly_out_tokens: float
    out_tokens_per_query: float
    avg_qps: float
    peak_qps: float
    required_tps: float
    annual_tokens: float


def estimate_workload(inp: TCOInputs) -> Workload:
    daily = inp.employees * inp.queries_per_user_day
    monthly_q = daily * inp.workdays_per_month
    monthly_tok = monthly_q * inp.tokens_per_query
    out_per_q = inp.tokens_per_query * (1 - inp.input_ratio)
    avg_qps = daily / (WORK_HOURS_PER_DAY * 3600) if daily else 0.0
    peak_qps = avg_qps * inp.peak_factor
    required = peak_qps * out_per_q * inp.headroom
    return Workload(
        daily_queries=daily, monthly_queries=monthly_q, monthly_tokens=monthly_tok,
        monthly_in_tokens=monthly_tok * inp.input_ratio,
        monthly_out_tokens=monthly_tok * (1 - inp.input_ratio),
        out_tokens_per_query=out_per_q, avg_qps=avg_qps, peak_qps=peak_qps,
        required_tps=required, annual_tokens=monthly_tok * 12,
    )


def api_monthly_cost(inp: TCOInputs, wl: Workload, model_key: Optional[str] = None) -> float:
    key = model_key or inp.api_model
    m = API_MODELS.get(key)
    if not m:
        return 0.0
    return wl.monthly_in_tokens / 1e6 * m["price_in"] + wl.monthly_out_tokens / 1e6 * m["price_out"]


# ---------------------------------------------------------------------------
# 硬件选型
# ---------------------------------------------------------------------------

@dataclass
class HardwarePlan:
    tier: str
    model_name: str
    gpu_key: str
    gpu_name: str
    gpu_count: int
    servers: int
    quant: str
    tps_capacity: float
    tps_required: float
    vram_total_gb: float
    hardware_capex: float
    storage_capex: float
    network_capex: float
    spare_capex: float
    impl_capex: float
    capex_total: float
    monthly_power_cost: float
    monthly_opex: float
    opex_breakdown: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


def _pick_tier(inp: TCOInputs, wl: Workload) -> str:
    """模型档位：机密及以上用 32B（精度优先），其余 14B；超大规模绝密升到 70B。"""
    if inp.secrecy_level == "绝密" and inp.employees >= 2500:
        return "70b"
    if inp.secrecy_level in ("机密", "绝密"):
        return "32b"
    return "14b"


def _platform_cost(gpu_count: int) -> float:
    for cap in sorted(PLATFORM_COST_BY_GPU):
        if gpu_count <= cap:
            return PLATFORM_COST_BY_GPU[cap]
    return PLATFORM_COST_BY_GPU[8] * math.ceil(gpu_count / 8)


def recommend_hardware(inp: TCOInputs, wl: Workload) -> HardwarePlan:
    tier = _pick_tier(inp, wl)
    tier_info = MODEL_TIERS[tier]
    weights = tier_info["weights_gb"]
    need_vram = weights * 1.35 + 4.0          # 权重 + KV Cache + 显存碎片
    need_tps = max(wl.required_tps, 60.0)

    candidates: List[Dict[str, Any]] = []
    for gk, g in GPU_CATALOG.items():
        for n in (1, 2, 4, 8):
            tps = g["tps"][tier] * n
            vram = g["vram"] * n
            if vram < need_vram or tps < need_tps:
                continue
            servers = 1 if n <= 4 else math.ceil(n / 8)
            hw = g["price"] * n * inp.hardware_discount + _platform_cost(n) * servers * inp.hardware_discount
            candidates.append({"gpu_key": gk, "gpu": g, "n": n, "servers": servers,
                               "tps": tps, "vram": vram, "hw": hw})
    if not candidates:  # 极端负载：直接上 8 卡旗舰
        g = GPU_CATALOG["h20"]
        n = 8
        candidates.append({"gpu_key": "h20", "gpu": g, "n": n, "servers": 1,
                           "tps": g["tps"][tier] * n, "vram": g["vram"] * n,
                           "hw": g["price"] * n * inp.hardware_discount + _platform_cost(n) * inp.hardware_discount})

    candidates.sort(key=lambda c: (c["hw"], c["gpu"]["tdp"] * c["n"]))
    best = candidates[0]

    gpu, n, servers, gpu_key = best["gpu"], best["n"], best["servers"], best["gpu_key"]
    hardware_capex = gpu["price"] * n * inp.hardware_discount
    platform_capex = _platform_cost(n) * servers * inp.hardware_discount

    redundancy = inp.redundancy
    if redundancy == "hot":
        hardware_capex *= 2
        servers *= 2
        platform_capex *= 2
    elif redundancy == "cold":
        platform_capex += _platform_cost(max(n, 4)) * 0.55 * inp.hardware_discount   # 冷备整机（不含 GPU）

    storage_tb = max(2.0, round(inp.doc_gb * 1.5 / 1024 + 1, 1))
    storage_capex = storage_tb * STORAGE_NVME_PER_TB + storage_tb * STORAGE_BACKUP_PER_TB
    network_capex = NETWORK_SWITCH_COST if servers >= 1 else 0.0
    hw_for_spare = hardware_capex + platform_capex
    spare_capex = hw_for_spare * SPARE_PARTS_RATIO

    impl_kb = 50000.0 + inp.doc_gb * 150.0
    impl_integration = 40000.0
    impl_deploy = 25000.0
    impl_security = SECRECY_SECURITY_COST.get(inp.secrecy_level, 0.0)
    if inp.secrecy_level == "机密":
        impl_security += 30000.0     # 脱敏网关（为混合云预留）
    impl_training = 15000.0
    impl_capex = impl_kb + impl_integration + impl_deploy + impl_security + impl_training

    capex_total = hardware_capex + platform_capex + storage_capex + network_capex + spare_capex + impl_capex

    # 月度运营成本
    # 热备意味着 GPU 也是双份的，功耗必须按双份算（早期版本只翻倍了 CAPEX，漏了备机功耗）
    gpu_units = n * (2 if redundancy == "hot" else 1)
    watt = gpu["tdp"] * gpu_units + PLATFORM_BASE_W * servers
    monthly_power = watt * 24 * 30.4 / 1000 * PUE * PRICE_KWH
    cabinets = max(1, math.ceil(servers / 2))
    monthly_colo = cabinets * COLOCATION_PER_CABINET_MONTH
    monthly_ops = inp.ops_fte * inp.ops_salary_wan * 10000 / 12
    monthly_iter = inp.monthly_iter_cost
    monthly_maint = (hardware_capex + platform_capex) * 0.03 / 12
    monthly_audit = SECRECY_AUDIT_MONTHLY.get(inp.secrecy_level, 0.0)
    opex = {"电费": monthly_power, "机房托管": monthly_colo, "运维人力": monthly_ops,
            "模型迭代与评测": monthly_iter, "硬件维保备件": monthly_maint, "合规审计": monthly_audit}
    monthly_opex = sum(opex.values())

    redundancy_label = {"none": "无冗余（单点）", "cold": "冷备（备机不含 GPU，故障切换约 4 小时）",
                        "hot": "双机热备（可用性 99.9%，成本翻倍）"}[redundancy]
    rationale = (
        f"业务峰值需输出 {wl.required_tps:,.0f} tokens/s，选用 {tier_info['name']}"
        f"（{tier.upper()} 档，AWQ 4-bit 量化，权重约 {weights} GB）承载；"
        f"{gpu['name']} × {n} 卡方案实测吞吐 {best['tps']:,.0f} tokens/s，"
        f"显存 {best['vram']} GB（需求 {need_vram:.1f} GB），余量充足。"
        f"冗余策略：{redundancy_label}。"
        f"选型理由：{gpu['note']}"
    )
    alts = [{"gpu": GPU_CATALOG[c["gpu_key"]]["name"], "count": c["n"],
             "hardware_capex": round(c["hw"], 0), "throughput_tps": round(c["tps"], 0),
             "note": "更省" if c["hw"] < best["hw"] else "更高配"}
            for c in candidates[1:4]]

    return HardwarePlan(
        tier=tier, model_name=tier_info["name"], gpu_key=gpu_key, gpu_name=gpu["name"],
        gpu_count=n, servers=servers, quant="AWQ 4-bit (W4A16)",
        tps_capacity=float(best["tps"]), tps_required=wl.required_tps, vram_total_gb=float(best["vram"]),
        hardware_capex=round(hardware_capex + platform_capex, 0),
        storage_capex=round(storage_capex, 0), network_capex=round(network_capex, 0),
        spare_capex=round(spare_capex, 0), impl_capex=round(impl_capex, 0),
        capex_total=round(capex_total, 0),
        monthly_power_cost=round(monthly_power, 0), monthly_opex=round(monthly_opex, 0),
        opex_breakdown={k: round(v, 0) for k, v in opex.items()},
        rationale=rationale, alternatives=alts,
    )


def recommend_hybrid_hardware(inp: TCOInputs, wl: Workload) -> HardwarePlan:
    """混合云：本地只承载基线负载（平均负载 × 1.2），峰值溢出到云端。"""
    base_wl = replace(wl, required_tps=wl.avg_qps * 1.2 * wl.out_tokens_per_query * 1.2)
    plan = recommend_hardware(replace(inp, redundancy="none", headroom=1.2), base_wl)
    plan.rationale = "混合云模式下本地仅需承载基线负载（平均负载 ×1.2），峰值自动溢出到云端按需实例。"
    return plan


def cloud_gpu_alternative(inp: TCOInputs, plan: HardwarePlan) -> Dict[str, Any]:
    """参照方案：不买硬件，改为租用云上 GPU 实例（云上私有化）的三年成本。

    这是很多企业"当前方案"的真实形态，也是"上自建硬件到底省多少云服务费"的直接对照。
    """
    key = plan.gpu_key if plan.gpu_key in CLOUD_GPU_MONTHLY else "a10"
    monthly_per_card = CLOUD_GPU_MONTHLY[key]
    cards = max(plan.gpu_count, 1)
    monthly = monthly_per_card * cards + 1500.0        # 含云存储与带宽
    three_year = monthly * MONTHS
    return {
        "instance": f"{plan.gpu_name} 云实例 × {cards}",
        "monthly": round(monthly, 0),
        "three_year": round(three_year, 0),
        "saving_vs_cloud_gpu": round(1 - plan.capex_total / three_year, 4) if three_year else 0.0,
        "note": "云 GPU 实例为租赁制，无资产沉淀；自建方案的成本优势随使用年限增长而扩大。",
    }


# ---------------------------------------------------------------------------
# 三方案 36 个月现金流
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    key: str
    label: str
    color: str
    available: bool
    note: str
    monthly: List[float] = field(default_factory=list)
    cumulative: List[float] = field(default_factory=list)
    total_36m: float = 0.0

    def break_even_with(self, other: "Scenario") -> Optional[float]:
        return first_crossover(self.cumulative, other.cumulative)


def first_crossover(cum_a: Sequence[float], cum_b: Sequence[float]) -> Optional[float]:
    """返回 cum_a 首次小于等于 cum_b 的月份（含线性插值，如 7.5）。"""
    for i in range(len(cum_a)):
        d = cum_a[i] - cum_b[i]
        if d <= 0:
            if i == 0:
                return 1.0
            prev = cum_a[i - 1] - cum_b[i - 1]
            span = prev - d
            frac = (prev / span) if span > 0 else 0.0
            return round(i + frac, 1)
    return None


def build_scenarios(inp: TCOInputs, plan: Optional[HardwarePlan] = None,
                    hybrid_plan: Optional[HardwarePlan] = None) -> Dict[str, Any]:
    wl = estimate_workload(inp)
    plan = plan or recommend_hardware(inp, wl)
    hybrid_plan = hybrid_plan or recommend_hybrid_hardware(inp, wl)

    base_api = api_monthly_cost(inp, wl)
    blocked = inp.secrecy_level == "绝密"

    api_m, local_m, hy_m = [], [], []
    for m in range(1, MONTHS + 1):
        year = (m - 1) // 12
        growth = (1 + inp.usage_growth) ** year
        decline = (1 - inp.api_price_decline) ** year
        api_cost = base_api * growth * decline
        local_cost = plan.monthly_opex * (1 + 0.03) ** year      # 年化通胀 3%
        hy_cost = (hybrid_plan.monthly_opex * (1 + 0.03) ** year
                   + base_api * inp.overflow_ratio * growth * decline)
        api_m.append(api_cost)
        local_m.append(local_cost + (plan.capex_total if m == 1 else 0.0))
        hy_m.append(hy_cost + (hybrid_plan.capex_total if m == 1 else 0.0))

    def cum(xs: Sequence[float]) -> List[float]:
        out, run = [], 0.0
        for x in xs:
            run += x
            out.append(run)
        return out

    api = Scenario("api", "商用闭源 API（按量付费）", "#d95926", not blocked,
                   "无前期投入，随用量线性增长；单价按年下降但业务量按年增长" if not blocked
                   else "绝密级数据禁止出域，该方案不适用")
    local = Scenario("local", "本地私有化服务器（自建）", "#3987e5", True,
                     "一次性 CAPEX + 稳定 OPEX，规模越大单位成本越低")
    hybrid = Scenario("hybrid", "混合云（本地基线 + 云端弹性）", "#199e70", not blocked,
                      f"本地承载基线负载，{inp.overflow_ratio:.0%} 峰值流量溢出到云端" if not blocked
                      else "绝密级数据禁止出域，该方案不适用")
    api.monthly, api.cumulative, api.total_36m = api_m, cum(api_m), round(sum(api_m), 0)
    local.monthly, local.cumulative, local.total_36m = local_m, cum(local_m), round(sum(local_m), 0)
    hybrid.monthly, hybrid.cumulative, hy_m_c = hy_m, cum(hy_m), round(sum(hy_m), 0)
    hybrid.total_36m = hy_m_c

    return {"workload": wl, "plan": plan, "hybrid_plan": hybrid_plan,
            "base_api_monthly": base_api,
            "scenarios": {"api": api, "local": local, "hybrid": hybrid}}


def efficiency_benefit(inp: TCOInputs, wl: Workload) -> Dict[str, Any]:
    """效率收益（影子价值）：故障排查/条款比对耗时下降折算的人力成本节约。

    两道保守化处理，避免把"理论上省下的时间"当成"已经省下的钱"：

    1. **可节省工时封顶**：一个人每天花在知识检索上的时间不可能超过
       `max_lookup_hours_per_day`，因此按此上限截断，防止"全员满负荷提问 × 每次省
       8.5 分钟"堆出数倍于编制规模的虚高工时。
    2. **采纳率爬坡**：第 m 月采纳率 = min(1, m / adoption_ramp_months)，新系统不会
       上线首月就全员满负荷使用。
    """
    blended_min = inp.deep_case_ratio * inp.deep_save_min + (1 - inp.deep_case_ratio) * inp.light_save_min
    raw_hours = wl.monthly_queries * blended_min / 60.0
    capacity = inp.employees * inp.workdays_per_month * inp.max_lookup_hours_per_day
    hours = min(raw_hours, capacity)
    capped = raw_hours > capacity
    gross = hours * inp.hourly_cost
    net = gross * inp.benefit_discount

    ramp = max(1, int(inp.adoption_ramp_months))
    adoption = [min(1.0, m / ramp) for m in range(1, MONTHS + 1)]
    three_year = sum(net * a for a in adoption)

    return {
        "blended_save_min": round(blended_min, 2),
        "monthly_hours_saved": round(hours, 0),
        "raw_monthly_hours": round(raw_hours, 0),
        "capacity_hours": round(capacity, 0),
        "capped_by_capacity": capped,
        "monthly_gross": round(gross, 0),
        "monthly_net": round(net, 0),
        "annual_net": round(net * 12, 0),
        "three_year_net": round(three_year, 0),
        "adoption_ramp_months": ramp,
        "avg_adoption_36m": round(sum(adoption) / len(adoption), 3),
    }


def sensitivity(inp: TCOInputs) -> List[Dict[str, Any]]:
    """敏感性分析：换用不同云端模型档位时，私有化的回本周期与三年节约率。"""
    wl = estimate_workload(inp)
    plan = recommend_hardware(inp, wl)
    rows = []
    for key, m in API_MODELS.items():
        if key == "private-cloud-gpu":
            continue
        base = api_monthly_cost(inp, wl, key)
        local_cum = plan.capex_total + sum(
            plan.monthly_opex * (1.03) ** ((k) // 12) for k in range(MONTHS))
        api_cum = sum(base * (1 + inp.usage_growth) ** (k // 12) * (1 - inp.api_price_decline) ** (k // 12)
                      for k in range(MONTHS))
        be = first_crossover([plan.capex_total + sum(plan.monthly_opex * (1.03) ** (j // 12) for j in range(i + 1))
                              for i in range(MONTHS)],
                             [sum(base * (1 + inp.usage_growth) ** (j // 12) * (1 - inp.api_price_decline) ** (j // 12)
                                  for j in range(i + 1)) for i in range(MONTHS)])
        rows.append({
            "model_key": key, "model_name": m["name"],
            "api_monthly": round(base, 0),
            "three_year_api": round(api_cum, 0),
            "three_year_local": round(local_cum, 0),
            "saving_ratio": round(1 - local_cum / api_cum, 4) if api_cum else 0.0,
            "break_even_month": be,
            "verdict": ("私有化在 3 年内更省" if local_cum < api_cum else "3 年内 API 更省，应走合规/自主可控论证"),
        })
    rows.sort(key=lambda r: r["three_year_api"])
    return rows


# ---------------------------------------------------------------------------
# 汇总 KPI
# ---------------------------------------------------------------------------

def summarize(inp: TCOInputs, result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = result or build_scenarios(inp)
    wl, plan = result["workload"], result["plan"]
    sc = result["scenarios"]
    eff = efficiency_benefit(inp, wl)

    api, local, hybrid = sc["api"], sc["local"], sc["hybrid"]
    savings_vs_api = 1 - local.total_36m / api.total_36m if api.total_36m else 0.0
    savings_hybrid_vs_api = 1 - hybrid.total_36m / api.total_36m if api.total_36m else 0.0
    be_local = local.break_even_with(api)
    be_hybrid = hybrid.break_even_with(api)
    roi_3y = (eff["three_year_net"] - local.total_36m) / local.total_36m if local.total_36m else 0.0
    net_benefit = eff["three_year_net"] - local.total_36m

    return {
        "workload": asdict(wl),
        "plan": asdict(plan),
        "api_monthly": round(result["base_api_monthly"], 0),
        "api_model_name": API_MODELS[inp.api_model]["name"],
        "local_monthly_opex": plan.monthly_opex,
        "local_capex": plan.capex_total,
        "three_year_api": api.total_36m,
        "three_year_local": local.total_36m,
        "three_year_hybrid": hybrid.total_36m,
        "savings_vs_api": round(savings_vs_api, 4),
        "savings_hybrid_vs_api": round(savings_hybrid_vs_api, 4),
        "break_even_local": be_local,
        "break_even_hybrid": be_hybrid,
        "tco_per_query_local": round(local.total_36m / (wl.monthly_queries * MONTHS), 4) if wl.monthly_queries else 0.0,
        "tco_per_query_api": round(api.total_36m / (wl.monthly_queries * MONTHS), 4) if wl.monthly_queries else 0.0,
        "efficiency": eff,
        # 绝对净收益比"收益率"更好沟通：几千个百分点的 ROI 只会让评审失去判断力
        "three_year_net_benefit": round(net_benefit, 0),
        "roi_3y_including_benefit": round(roi_3y, 4),
        "secrecy_note": SECRECY_COMPLIANCE_NOTE.get(inp.secrecy_level, ""),
        "compliance_blocked": inp.secrecy_level == "绝密",
    }


def monthly_rows(inp: TCOInputs, result: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    result = result or build_scenarios(inp)
    sc = result["scenarios"]
    rows = []
    for i in range(MONTHS):
        rows.append({
            "月份": i + 1,
            "API月度成本": round(sc["api"].monthly[i], 0),
            "API累计成本": round(sc["api"].cumulative[i], 0),
            "本地月度成本": round(sc["local"].monthly[i], 0),
            "本地累计成本": round(sc["local"].cumulative[i], 0),
            "混合云月度成本": round(sc["hybrid"].monthly[i], 0),
            "混合云累计成本": round(sc["hybrid"].cumulative[i], 0),
            "累计节省(本地vs API)": round(sc["api"].cumulative[i] - sc["local"].cumulative[i], 0),
        })
    return rows


def to_csv(rows: Sequence[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    if not rows:
        return ""
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 《致管理层》报告
# ---------------------------------------------------------------------------

def _wan(x: float) -> str:
    return f"{x / 10000:,.1f} 万元"


def render_report_markdown(inp: TCOInputs, result: Optional[Dict[str, Any]] = None) -> str:
    result = result or build_scenarios(inp)
    kpi = summarize(inp, result)
    wl, plan = result["workload"], result["plan"]
    eff = kpi["efficiency"]
    be = kpi["break_even_local"]
    be_txt = f"第 {be} 个月" if be else "36 个月内未回本"
    be_h = kpi["break_even_hybrid"]
    be_h_txt = f"第 {be_h} 个月" if be_h else "36 个月内未回本"

    lines: List[str] = []
    lines.append("# 致管理层：AI 知识库私有化立项 ROI 评估报告")
    lines.append("")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')}　|　"
                 f"测算工具：OnPrem-RAG-Kit TCO Calculator　|　货币单位：人民币")
    lines.append("")
    lines.append("## 一、结论摘要")
    lines.append("")
    lines.append(f"1. **回本临界点**：在给定业务负载下，本地私有化方案于 **{be_txt}** 起低于云端 API 按量付费的累计成本"
                 f"（混合云方案为 **{be_h_txt}**）。")
    lines.append(f"1. **成本对比结论**：本地私有化三年 TCO 为 {_wan(kpi['three_year_local'])}，"
                 f"对比「{kpi['api_model_name']}」API 方案（{_wan(kpi['three_year_api'])}）"
                 + (f"**节约 {kpi['savings_vs_api']:.1%}**" if kpi['savings_vs_api'] >= 0
                    else f"**高出 {abs(kpi['savings_vs_api']):.1%}**——本场景下不应以「省钱」作为立项依据，"
                         f"而应论证数据合规与自主可控")
                 + f"；混合云方案" + (f"节约 {kpi['savings_hybrid_vs_api']:.1%}。" if kpi['savings_hybrid_vs_api'] >= 0
                                      else f"高出 {abs(kpi['savings_hybrid_vs_api']):.1%}。"))
    lines.append(f"3. **单位成本**：本地化单次问答综合成本 ¥{kpi['tco_per_query_local']:.4f}，"
                 f"API 方案 ¥{kpi['tco_per_query_api']:.4f}。")
    lines.append(f"4. **效率收益（影子价值，不计入上方 TCO 对比）**：按 {eff['blended_save_min']:.1f} 分钟/次的"
                 f"混合节省口径，每月释放约 {eff['monthly_hours_saved']:,.0f} 人·小时"
                 + ("（已按人均每日 "
                    f"{inp.max_lookup_hours_per_day:.1f} 小时的检索总工时封顶）" if eff["capped_by_capacity"] else "")
                 + f"，按 {inp.benefit_discount:.0%} 价值实现率折算后年化约 {_wan(eff['annual_net'])}；"
                 f"计入 {eff['adoption_ramp_months']} 个月采纳爬坡后，三年净收益约 "
                 f"{_wan(kpi['three_year_net_benefit'])}。"
                 f"**该数字高度依赖「人均节省时长」与「价值实现率」两项假设，仅用于内部沟通，不作为立项依据。**")
    lines.append(f"5. **数据合规**：当前数据保密等级为「{inp.secrecy_level}」——{kpi['secrecy_note']}。")
    lines.append("")
    lines.append("## 二、业务负载测算")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 覆盖员工数 | {inp.employees:,} 人 |")
    lines.append(f"| 人均日提问量 | {inp.queries_per_user_day:.1f} 次 |")
    lines.append(f"| 日均问答量 | {wl.daily_queries:,.0f} 次 |")
    lines.append(f"| 月均问答量（{inp.workdays_per_month} 工作日） | {wl.monthly_queries:,.0f} 次 |")
    lines.append(f"| 单次平均 Token | {inp.tokens_per_query:,}（输入 {inp.input_ratio:.0%} / 输出 {1 - inp.input_ratio:.0%}） |")
    lines.append(f"| 月均 Token 消耗 | {wl.monthly_tokens / 1e6:,.1f} M tokens |")
    lines.append(f"| 平均 QPS / 峰值 QPS | {wl.avg_qps:.3f} / {wl.peak_qps:.3f} |")
    lines.append(f"| 峰值输出吞吐需求 | {wl.required_tps:,.0f} tokens/s（含 {inp.headroom:.1f}× 余量） |")
    lines.append("")
    lines.append("## 三、硬件选型与部署架构建议")
    lines.append("")
    lines.append(f"* **推理模型**：{plan.model_name}（{plan.quant}）")
    lines.append(f"* **GPU 配置**：{plan.gpu_name} × {plan.gpu_count}（{plan.servers} 台服务器，"
                 f"显存合计 {plan.vram_total_gb:.0f} GB，实测吞吐能力 {plan.tps_capacity:,.0f} tokens/s）")
    lines.append(f"* **推理引擎**：vLLM（PagedAttention + Continuous Batching）；量化权重经 AWQ 校准，"
                 f"精度损失 <1%，显存占用下降约 70%")
    lines.append(f"* **检索架构**：BM25 + 向量混合召回（RRF 融合）→ 重排 → 父子分段整表回溯")
    lines.append(f"* **选型理由**：{plan.rationale}")
    lines.append(f"* **一次性投入**：硬件与平台 {_wan(plan.hardware_capex)}、存储 {_wan(plan.storage_capex)}、"
                 f"网络 {_wan(plan.network_capex)}、备件 {_wan(plan.spare_capex)}、"
                 f"实施与合规 {_wan(plan.impl_capex)}，合计 **{_wan(plan.capex_total)}**")
    lines.append("")
    lines.append("**月度运营成本明细**")
    lines.append("")
    lines.append("| 项目 | 金额（元/月） |")
    lines.append("| --- | --- |")
    for k, v in plan.opex_breakdown.items():
        lines.append(f"| {k} | {v:,.0f} |")
    lines.append(f"| **合计** | **{plan.monthly_opex:,.0f}** |")
    lines.append("")
    lines.append("## 四、三年 TCO 对比")
    lines.append("")
    lines.append("| 方案 | 三年总成本 | 相对 API 节约 | 回本临界点 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(f"| 商用闭源 API（{kpi['api_model_name']}） | {_wan(kpi['three_year_api'])} | — | — |")
    lines.append(f"| 本地私有化服务器 | {_wan(kpi['three_year_local'])} | {kpi['savings_vs_api']:.1%} | {be_txt} |")
    lines.append(f"| 混合云（本地 + 云端弹性） | {_wan(kpi['three_year_hybrid'])} | "
                 f"{kpi['savings_hybrid_vs_api']:.1%} | {be_h_txt} |")
    lines.append("")
    lines.append("## 五、敏感性分析（云端模型档位）")
    lines.append("")
    lines.append("| 云端模型 | API 月成本 | 三年 API 成本 | 三年本地成本 | 节约率 | 回本临界点 | 结论 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in sensitivity(inp):
        be_r = f"第 {r['break_even_month']} 个月" if r["break_even_month"] else "未回本"
        lines.append(f"| {r['model_name']} | {r['api_monthly']:,.0f} 元 | {_wan(r['three_year_api'])} | "
                     f"{_wan(r['three_year_local'])} | {r['saving_ratio']:.1%} | {be_r} | {r['verdict']} |")
    lines.append("")
    lines.append("> **决策提示**：若云端选用极低价模型（如 DeepSeek-V3），API 单价可低至本地化的十分之一，"
                 "此时私有化的立项依据**不应是省钱**，而应论证：① 数据不出域的合规要求；"
                 "② 峰值并发与限流不可控带来的业务中断风险；③ 微调与模型自主迭代能力；"
                 "④ 三年后随业务量增长的成本曲线反超。")
    lines.append("")
    lines.append("## 六、风险与建议")
    lines.append("")
    lines.append("1. **用量增长风险**：本模型假设业务量年增长 "
                 f"{inp.usage_growth:.0%}。用量增长会同时推高 API 成本并摊薄本地固定成本，"
                 "对私有化方案是利好；反之若用量萎缩，本地方案的固定成本将被浪费。")
    lines.append("2. **API 降价风险**：本模型假设 API 单价年降 "
                 f"{inp.api_price_decline:.0%}。若降幅显著超过业务量增速，回本周期将被拉长。")
    lines.append("3. **实施风险**：本地化的隐性成本集中在知识库工程与运维人力（本模型已按 "
                 f"{inp.ops_fte} FTE × {inp.ops_salary_wan:.0f} 万元/年计入），"
                 "建议首期按 8-12 周交付，先跑通 3 个高价值场景再全量推广。")
    lines.append("4. **建议路径**：机密级及以上数据先本地化，非敏感场景保留 API 通道（混合云），"
                 "既满足合规又不牺牲弹性。")
    lines.append("")
    lines.append("## 附录：关键假设")
    lines.append("")
    lines.append("| 参数 | 取值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 电价 / PUE | {PRICE_KWH} 元/度 / {PUE} |")
    lines.append(f"| 机房托管 | {COLOCATION_PER_CABINET_MONTH:,.0f} 元/柜/月 |")
    lines.append(f"| 运维人力 | {inp.ops_fte} FTE × {inp.ops_salary_wan:.0f} 万元/年 |")
    lines.append(f"| 硬件采购折扣 | {inp.hardware_discount:.2f} |")
    lines.append(f"| 冗余策略 | {inp.redundancy} |")
    lines.append(f"| 效率折算单价 | {inp.hourly_cost:.0f} 元/人·小时，价值实现率 {inp.benefit_discount:.0%} |")
    lines.append(f"| 采纳率爬坡 | {inp.adoption_ramp_months} 个月（第 m 月采纳率 = min(1, m/{inp.adoption_ramp_months})） |")
    lines.append(f"| 人均每日检索工时上限 | {inp.max_lookup_hours_per_day:.1f} 小时（用于封顶效率收益） |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 OnPrem-RAG-Kit 自动生成。所有价格与参数均可在测算器中调整，"
                 "结论随参数实时更新；建议以厂商正式报价替换默认参考价后提交立项。*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 场景预设
# ---------------------------------------------------------------------------

SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "轻量型（合规驱动）": {
        "employees": 800, "queries_per_user_day": 3.0, "tokens_per_query": 3000,
        "secrecy_level": "机密", "api_model": "gpt-4o", "ops_fte": 0.3,
        "api_price_decline": 0.20, "usage_growth": 0.15, "redundancy": "cold",
        "benefit_discount": 0.3, "hardware_discount": 1.0,
    },
    "平衡型（推荐）": {
        "employees": 3000, "queries_per_user_day": 5.0, "tokens_per_query": 4000,
        "secrecy_level": "机密", "api_model": "gpt-4o", "ops_fte": 0.5,
        "api_price_decline": 0.15, "usage_growth": 0.15, "redundancy": "cold",
        "benefit_discount": 0.5, "hardware_discount": 0.95,
    },
    "规模化（强推荐）": {
        "employees": 5000, "queries_per_user_day": 8.0, "tokens_per_query": 5000,
        "secrecy_level": "内部", "api_model": "qwen-max", "ops_fte": 0.4,
        "api_price_decline": 0.08, "usage_growth": 0.25, "redundancy": "hot",
        "benefit_discount": 0.6, "hardware_discount": 0.90,
    },
}


def preset(name: str, base: Optional[TCOInputs] = None) -> TCOInputs:
    base = base or TCOInputs()
    data = SCENARIO_PRESETS.get(name, {})
    return TCOInputs.from_dict({**base.to_dict(), **data})


if __name__ == "__main__":
    for name in SCENARIO_PRESETS:
        inp = preset(name)
        res = build_scenarios(inp)
        kpi = summarize(inp, res)
        be = kpi["break_even_local"]
        print("=" * 78)
        print(f"[{name}]  {inp.employees} 人 × {inp.queries_per_user_day:.1f} 次/日 × "
              f"{inp.tokens_per_query} tokens")
        print(f"  API 月成本 {kpi['api_monthly']:,.0f} 元 | 本地 CAPEX {kpi['local_capex']:,.0f} 元 | "
              f"本地 OPEX {kpi['local_monthly_opex']:,.0f} 元/月")
        print(f"  三年: API {kpi['three_year_api'] / 10000:.1f}万 / 本地 {kpi['three_year_local'] / 10000:.1f}万 / "
              f"混合 {kpi['three_year_hybrid'] / 10000:.1f}万")
        print(f"  节约 {kpi['savings_vs_api']:.1%} | 回本临界点 "
              f"{('第 %.1f 个月' % be) if be else '未回本'} | 选型 {kpi['plan']['gpu_name']} × {kpi['plan']['gpu_count']}")
    print("=" * 78)
