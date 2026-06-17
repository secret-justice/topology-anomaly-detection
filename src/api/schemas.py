# -*- coding: utf-8 -*-
"""
Pydantic 数据模型
定义 API 请求/响应的结构化数据，自动校验 + 自动生成 OpenAPI 文档
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class AnomalyType(str, Enum):
    """五类拓扑异常"""
    MODEL_MISMATCH = "\u56fe\u6a21\u4e0d\u7b26"
    TOPO_INTERRUPT = "\u62d3\u6251\u4e2d\u65ad"
    VIRTUAL_FAULTY = "\u865a\u63a5/\u9519\u63a5"
    TELE_TOPO_MISMATCH = "\u9065\u6d4b!=\u62d3\u6251"
    TELE_SIGNAL_MISMATCH = "\u9065\u4fe1!=\u9065\u6d4b"


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("ok", description="\u670d\u52a1\u72b6\u6001")
    version: str = Field("1.0.0", description="API \u7248\u672c\u53f7")
    modules: Dict[str, bool] = Field(default_factory=dict, description="\u5404\u6a21\u5757\u53ef\u7528\u6027")


class AnomalyItem(BaseModel):
    """单条异常记录"""
    type: str = Field(..., description="\u5f02\u5e38\u7c7b\u578b")
    location: str = Field(..., description="\u5f02\u5e38\u4f4d\u7f6e")
    confidence: float = Field(..., ge=0.0, le=1.0, description="\u7f6e\u4fe1\u5ea6")
    layer: str = Field(..., description="\u68c0\u6d4b\u5c42")
    details: str = Field("", description="\u5f02\u5e38\u8be6\u60c5")


class CorrectionItem(BaseModel):
    """单条修正方案"""
    anomaly_type: str = Field(..., description="\u5bf9\u5e94\u5f02\u5e38\u7c7b\u578b")
    action: str = Field(..., description="\u5efa\u8bae\u52a8\u4f5c")
    priority: str = Field(..., description="\u4f18\u5148\u7ea7")
    target: str = Field(..., description="\u4fee\u6b63\u76ee\u6807")
    description: str = Field("", description="\u4fee\u6b63\u63cf\u8ff0")
    steps: List[str] = Field(default_factory=list, description="\u4fee\u6b63\u6b65\u9aa4")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="\u7f6e\u4fe1\u5ea6")


class DetectResponse(BaseModel):
    """异常检测响应"""
    success: bool = Field(True, description="\u662f\u5426\u6210\u529f")
    anomaly_count: int = Field(..., ge=0, description="\u5f02\u5e38\u6570\u91cf")
    anomalies: List[AnomalyItem] = Field(default_factory=list, description="\u5f02\u5e38\u5217\u8868")
    summary: Dict[str, Any] = Field(default_factory=dict, description="\u68c0\u6d4b\u6458\u8981")


class CorrectionResponse(BaseModel):
    """修正方案响应"""
    success: bool = Field(True, description="\u662f\u5426\u6210\u529f")
    correction_count: int = Field(..., ge=0, description="\u4fee\u6b63\u65b9\u6848\u6570\u91cf")
    corrections: List[CorrectionItem] = Field(default_factory=list, description="\u4fee\u6b63\u5217\u8868")
    summary: Dict[str, int] = Field(default_factory=dict, description="\u4fee\u6b63\u7edf\u8ba1")


class UploadResponse(BaseModel):
    """文件上传响应"""
    success: bool = Field(True, description="\u662f\u5426\u6210\u529f")
    message: str = Field("", description="\u6d88\u606f")
    cim_devices: int = Field(0, description="CIM \u8bbe\u5907\u6570")
    svg_devices: int = Field(0, description="SVG \u8bbe\u5907\u6570")
    switches: int = Field(0, description="\u5f00\u5173\u6570")
    nodes: int = Field(0, description="\u8282\u70b9\u6570")


class DetectRequest(BaseModel):
    """异常检测请求"""
    network_name: str = Field("example_simple", description="PandaPower \u7f51\u7edc\u540d")
    use_rule_engine: bool = Field(True, description="\u542f\u7528\u89c4\u5219\u5f15\u64ce")
    use_state_estimator: bool = Field(True, description="\u542f\u7528\u72b6\u6001\u4f30\u8ba1")
    inject_anomalies: bool = Field(False, description="\u6ce8\u5165\u5408\u6210\u5f02\u5e38")
    anomaly_count: int = Field(3, ge=1, le=20, description="\u6ce8\u5165\u6570\u91cf")
    random_seed: int = Field(42, description="\u968f\u673a\u79cd\u5b50")


class CorrectRequest(BaseModel):
    """修正请求"""
    network_name: str = Field("example_simple", description="PandaPower \u7f51\u7edc\u540d")
    use_rule_engine: bool = Field(True)
    use_state_estimator: bool = Field(True)
    inject_anomalies: bool = Field(True)
    anomaly_count: int = Field(3, ge=1, le=20)
    random_seed: int = Field(42)


class BatchDetectRequest(BaseModel):
    """批量检测请求"""
    network_names: List[str] = Field(..., min_length=1, max_length=10)
    use_rule_engine: bool = Field(True)
    use_state_estimator: bool = Field(True)
    inject_anomalies: bool = Field(False)
    anomaly_count: int = Field(3, ge=1, le=20)
    random_seed: int = Field(42)


class BatchDetectResponse(BaseModel):
    """批量检测响应"""
    success: bool = Field(True)
    total_networks: int = Field(...)
    results: Dict[str, DetectResponse] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """统一错误响应"""
    success: bool = Field(False)
    error_code: int = Field(...)
    error_type: str = Field(...)
    message: str = Field(...)
    detail: Optional[str] = Field(None)
