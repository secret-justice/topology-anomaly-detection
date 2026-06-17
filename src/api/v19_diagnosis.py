# -*- coding: utf-8 -*-
"""
v19 增强版诊断API端点
使用EnhancedKnowledgeBase, AnomalyCorrelator, CorrectionAdvisor, DiagnosticReporter
"""
from fastapi import APIRouter
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v19", tags=["v19 Diagnosis"])


@router.post("/diagnose/enhanced")
async def diagnose_enhanced(request: dict):
    """
    增强版诊断: 关联分析 + 修正建议 + 专业报告
    
    输入: {"anomalies": [...], "network_context": {...}}
    输出: 完整诊断报告
    """
    from llm_assistant.enhanced_llm_v19 import (
        EnhancedKnowledgeBase, AnomalyCorrelator, 
        CorrectionAdvisor, DiagnosticReporter
    )
    
    anomalies = request.get("anomalies", [])
    network_context = request.get("network_context", {})
    
    # 初始化组件
    kb = EnhancedKnowledgeBase()
    correlator = AnomalyCorrelator(kb)
    advisor = CorrectionAdvisor(kb)
    reporter = DiagnosticReporter(kb)
    
    # 关联分析
    correlation = correlator.correlate(anomalies, network_context)
    
    # 修正建议
    correction = advisor.advise(anomalies, correlation, network_context)
    
    # 诊断报告
    report = reporter.generate(anomalies, correlation, correction, network_context)
    
    return {
        "status": "success",
        "anomaly_count": len(anomalies),
        "correlation": correlation,
        "correction": correction,
        "report": report,
    }


@router.post("/diagnose/markdown")
async def diagnose_markdown_v19(request: dict):
    """
    生成Markdown格式的诊断报告
    
    输入: {"anomalies": [...], "network_context": {...}}
    输出: {"markdown": "..."}
    """
    from llm_assistant.enhanced_llm_v19 import (
        EnhancedKnowledgeBase, AnomalyCorrelator, 
        CorrectionAdvisor, DiagnosticReporter
    )
    
    anomalies = request.get("anomalies", [])
    network_context = request.get("network_context", {})
    
    kb = EnhancedKnowledgeBase()
    correlator = AnomalyCorrelator(kb)
    advisor = CorrectionAdvisor(kb)
    reporter = DiagnosticReporter(kb)
    
    correlation = correlator.correlate(anomalies, network_context)
    correction = advisor.advise(anomalies, correlation, network_context)
    report = reporter.generate(anomalies, correlation, correction, network_context)
    
    return {"markdown": report.get("detailed_report", "")}


@router.post("/diagnose/professional")
async def diagnose_professional(request: dict):
    """
    生成专业报告(含术语解释和因果链)
    
    输入: {"anomalies": [...], "network_context": {...}}
    输出: {"professional_report": "..."}
    """
    from llm_assistant.enhanced_llm_v19 import (
        EnhancedKnowledgeBase, AnomalyCorrelator, 
        CorrectionAdvisor, DiagnosticReporter
    )
    
    anomalies = request.get("anomalies", [])
    network_context = request.get("network_context", {})
    
    kb = EnhancedKnowledgeBase()
    correlator = AnomalyCorrelator(kb)
    advisor = CorrectionAdvisor(kb)
    reporter = DiagnosticReporter(kb)
    
    correlation = correlator.correlate(anomalies, network_context)
    correction = advisor.advise(anomalies, correlation, network_context)
    report = reporter.generate(anomalies, correlation, correction, network_context)
    
    return {"professional_report": report.get("professional_report", "")}


@router.get("/knowledge/{anomaly_type}")
async def get_knowledge(anomaly_type: str):
    """
    获取异常类型知识
    
    输入: anomaly_type (如 "topo_interrupt")
    输出: 异常类型详细知识
    """
    from llm_assistant.enhanced_llm_v19 import EnhancedKnowledgeBase
    
    kb = EnhancedKnowledgeBase()
    knowledge = kb.get_knowledge(anomaly_type)
    
    if not knowledge:
        return {"error": f"Unknown anomaly type: {anomaly_type}"}
    
    return {
        "type": anomaly_type,
        "name_cn": knowledge.name_cn,
        "name_en": knowledge.name_en,
        "category": knowledge.category,
        "severity": knowledge.severity.value,
        "description": knowledge.description,
        "possible_causes": knowledge.possible_causes,
        "physical_effects": knowledge.physical_effects,
        "detection_methods": knowledge.detection_methods,
        "correction_strategies": knowledge.correction_strategies,
        "related_anomalies": knowledge.related_anomalies,
        "causal_chains": knowledge.causal_chains,
        "professional_terms": knowledge.professional_terms,
    }


@router.get("/glossary")
async def get_glossary():
    """
    获取专业术语表
    
    输出: {"glossary": {"PT": "电压互感器", ...}}
    """
    from llm_assistant.enhanced_llm_v19 import EnhancedKnowledgeBase
    
    kb = EnhancedKnowledgeBase()
    return {"glossary": kb.professional_glossary}
