# -*- coding: utf-8 -*-
"""
v20 诊断API端点 - 使用新的DiagnosticEngine

API端点:
  - POST /api/v20/diagnose: 完整诊断
  - POST /api/v20/diagnose/markdown: Markdown报告
  - POST /api/v20/diagnose/professional: 专业报告
  - GET /api/v20/knowledge/{type}: 知识查询
  - GET /api/v20/glossary: 术语表
  - GET /api/v20/engine/status: 引擎状态
"""
from fastapi import APIRouter
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v20", tags=["v20 Diagnosis"])

# 全局诊断引擎实例（懒加载）
_engine = None


def get_engine(use_llm: bool = False, llm_model_path: Optional[str] = None):
    """获取诊断引擎实例（单例）"""
    global _engine
    if _engine is None:
        from llm_assistant.diagnostic_engine import DiagnosticEngine
        _engine = DiagnosticEngine(use_llm=use_llm, llm_model_path=llm_model_path)
    return _engine


@router.post("/diagnose")
async def diagnose(request: dict):
    """
    完整诊断
    
    输入: {
        "anomalies": [...],
        "network_context": {...},
        "use_llm": false,  // 可选，是否启用LLM润色
        "llm_model_path": null  // 可选，LLM模型路径
    }
    输出: 完整诊断结果
    """
    anomalies = request.get("anomalies", [])
    network_context = request.get("network_context", {})
    use_llm = request.get("use_llm", False)
    llm_model_path = request.get("llm_model_path")
    
    engine = get_engine(use_llm=use_llm, llm_model_path=llm_model_path)
    result = engine.diagnose(anomalies, network_context)
    
    return result


@router.post("/diagnose/markdown")
async def diagnose_markdown(request: dict):
    """
    生成Markdown格式的诊断报告
    
    输入: {
        "anomalies": [...],
        "network_context": {...}
    }
    输出: {"markdown": "..."}
    """
    anomalies = request.get("anomalies", [])
    network_context = request.get("network_context", {})
    
    engine = get_engine()
    markdown = engine.diagnose_markdown(anomalies, network_context)
    
    return {"markdown": markdown}


@router.post("/diagnose/professional")
async def diagnose_professional(request: dict):
    """
    生成专业报告
    
    输入: {
        "anomalies": [...],
        "network_context": {...}
    }
    输出: {"professional_report": "..."}
    """
    anomalies = request.get("anomalies", [])
    network_context = request.get("network_context", {})
    
    engine = get_engine()
    professional = engine.diagnose_professional(anomalies, network_context)
    
    return {"professional_report": professional}


@router.get("/knowledge/{anomaly_type}")
async def get_knowledge(anomaly_type: str):
    """
    获取异常类型知识
    
    输入: anomaly_type (如 "topo_interrupt")
    输出: 异常类型详细知识
    """
    engine = get_engine()
    knowledge = engine.get_knowledge(anomaly_type)
    
    if not knowledge:
        return {"error": f"Unknown anomaly type: {anomaly_type}"}
    
    return knowledge


@router.get("/glossary")
async def get_glossary():
    """
    获取专业术语表
    
    输出: {"glossary": {"PT": "电压互感器", ...}}
    """
    engine = get_engine()
    return {"glossary": engine.get_glossary()}


@router.get("/engine/status")
async def engine_status():
    """
    获取引擎状态
    
    输出: {
        "backend": "rules" | "llamacpp",
        "llm_available": true/false,
        "model_path": "...",
        "components": {...}
    }
    """
    engine = get_engine()
    
    return {
        "backend": engine.llm.get_backend() if engine.llm else "rules",
        "llm_available": engine.llm.is_available() if engine.llm else False,
        "use_llm": engine.use_llm,
        "components": {
            "knowledge_base": True,
            "correlator": True,
            "advisor": True,
            "reporter": True,
            "llm": engine.llm is not None,
        }
    }
