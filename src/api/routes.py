# -*- coding: utf-8 -*-
"""
REST API 路由层
定义所有 HTTP 端点，负责参数校验和响应格式化
业务逻辑委托给 service 层处理
"""
from __future__ import annotations
import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse, DetectRequest, DetectResponse, AnomalyItem,
    CorrectRequest, CorrectionResponse, CorrectionItem,
    UploadResponse, BatchDetectRequest, BatchDetectResponse, ErrorResponse,
)
from api import service

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# GET /api/v1/health - 健康检查
# ============================================================
@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="返回服务状态和各模块可用性。503 表示服务降级。",
    responses={503: {"model": ErrorResponse, "description": "服务降级"}},
)
async def health_check():
    """
    健康检查端点。
    返回 status ("ok"/"degraded")、version 和 modules 可用性字典。
    """
    result = service.get_health()
    status_code = 200 if result["status"] == "ok" else 503
    return JSONResponse(content=result, status_code=status_code)


# ============================================================
# POST /api/v1/data/upload - 上传 CIM/SVG 数据文件
# ============================================================
@router.post(
    "/api/v1/data/upload",
    response_model=UploadResponse,
    summary="上传 CIM/SVG 数据",
    description="上传 CIM/RDF 或 SVG 文件，服务端解析并返回设备统计。",
    responses={
        400: {"model": ErrorResponse, "description": "请求参数错误"},
        422: {"model": ErrorResponse, "description": "文件解析失败"},
        500: {"model": ErrorResponse, "description": "服务内部错误"},
    },
)
async def upload_data(
    cim_file: Optional[UploadFile] = File(None, description="CIM/RDF 文件"),
    svg_file: Optional[UploadFile] = File(None, description="SVG 图形文件"),
):
    """
    上传 CIM/SVG 数据文件。
    支持同时上传 CIM 和 SVG 文件；至少上传一个，否则返回 400。
    """
    if cim_file is None and svg_file is None:
        raise HTTPException(
            status_code=400,
            detail="至少上传一个文件（cim_file 或 svg_file）"
        )

    cim_path = None
    svg_path = None
    temp_files = []

    try:
        if cim_file:
            suffix = os.path.splitext(cim_file.filename or ".xml")[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            content = await cim_file.read()
            tmp.write(content)
            tmp.close()
            cim_path = tmp.name
            temp_files.append(cim_path)

        if svg_file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
            content = await svg_file.read()
            tmp.write(content)
            tmp.close()
            svg_path = tmp.name
            temp_files.append(svg_path)

        result = service.upload_data(cim_path=cim_path, svg_path=svg_path)
        return UploadResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"数据解析失败: {str(e)}")
    except Exception as e:
        logger.exception(f"upload error: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass


# ============================================================
# POST /api/v1/detect - 异常检测
# ============================================================
@router.post(
    "/api/v1/detect",
    response_model=DetectResponse,
    summary="执行异常检测",
    description="对指定 PandaPower 网络执行拓扑异常检测。",
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def detect_anomalies(request: DetectRequest):
    """
    异常检测端点。
    请求体: network_name, use_rule_engine, use_state_estimator,
            inject_anomalies, anomaly_count, random_seed
    """
    try:
        result = service.run_detect(
            network_name=request.network_name,
            use_rule_engine=request.use_rule_engine,
            use_state_estimator=request.use_state_estimator,
            inject_anomalies=request.inject_anomalies,
            anomaly_count=request.anomaly_count,
            random_seed=request.random_seed,
        )
        return DetectResponse(
            success=result["success"],
            anomaly_count=result["anomaly_count"],
            anomalies=[AnomalyItem(**a) for a in result["anomalies"]],
            summary=result["summary"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"依赖模块不可用: {str(e)}")
    except Exception as e:
        logger.exception(f"detect error: {e}")
        raise HTTPException(status_code=500, detail=f"检测执行失败: {str(e)}")


# ============================================================
# POST /api/v1/correct - 检测 + 修正
# ============================================================
@router.post(
    "/api/v1/correct",
    response_model=CorrectionResponse,
    summary="执行异常检测并生成修正方案",
    description="先执行异常检测，再基于检测结果生成修正建议。",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def correct_anomalies(request: CorrectRequest):
    """检测 + 修正端点"""
    try:
        result = service.run_correct(
            network_name=request.network_name,
            use_rule_engine=request.use_rule_engine,
            use_state_estimator=request.use_state_estimator,
            inject_anomalies=request.inject_anomalies,
            anomaly_count=request.anomaly_count,
            random_seed=request.random_seed,
        )
        return CorrectionResponse(
            success=result["success"],
            correction_count=result["correction_count"],
            corrections=[CorrectionItem(**c) for c in result["corrections"]],
            summary=result["correction_summary"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"correct error: {e}")
        raise HTTPException(status_code=500, detail=f"修正执行失败: {str(e)}")


# ============================================================
# POST /api/v1/batch/detect - 批量检测
# ============================================================
@router.post(
    "/api/v1/batch/detect",
    response_model=BatchDetectResponse,
    summary="批量异常检测",
    description="对多个 PandaPower 网络依次执行异常检测。",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def batch_detect(request: BatchDetectRequest):
    """
    批量检测端点。
    单个网络失败不影响其他网络，错误记录在 errors 字段中。
    """
    results = {}
    errors = {}

    for name in request.network_names:
        try:
            r = service.run_detect(
                network_name=name,
                use_rule_engine=request.use_rule_engine,
                use_state_estimator=request.use_state_estimator,
                inject_anomalies=request.inject_anomalies,
                anomaly_count=request.anomaly_count,
                random_seed=request.random_seed,
            )
            results[name] = DetectResponse(
                success=r["success"],
                anomaly_count=r["anomaly_count"],
                anomalies=[AnomalyItem(**a) for a in r["anomalies"]],
                summary=r["summary"],
            )
        except Exception as e:
            logger.warning(f"batch detect {name} failed: {e}")
            errors[name] = str(e)

    return BatchDetectResponse(
        success=len(errors) == 0,
        total_networks=len(request.network_names),
        results=results,
        errors=errors,
    )

# ============================================================
# v9 API Endpoints - 25-type detection pipeline
# ============================================================
from api.v9_service import run_v9_detect, get_network_topology, list_networks


@router.get(
    "/api/v1/networks",
    summary="列出可用网络",
    description="返回所有可用的PandaPower测试网络及其元数据。",
)
async def get_networks():
    """返回可用网络列表。"""
    return {"networks": list_networks()}


@router.get(
    "/api/v1/network/{network_name}/topology",
    summary="获取网络拓扑",
    description="返回指定网络的拓扑数据（节点和边），供前端D3.js渲染。",
)
async def get_topology(network_name: str):
    """返回网络拓扑图数据。"""
    try:
        return get_network_topology(network_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"网络加载失败: {str(e)}")


@router.post(
    "/api/v1/v9/detect",
    summary="v9异常检测(25种类型)",
    description="执行完整的v9检测管线，支持25种异常类型。",
)
async def v9_detect(
    network_name: str = "case33bw",
    inject_anomalies: bool = True,
    anomaly_count: int = 3,
    random_seed: int = 42,
):
    """执行v9 25类异常检测。"""
    try:
        result = run_v9_detect(
            network_name=network_name,
            inject_anomalies=inject_anomalies,
            anomaly_count=anomaly_count,
            random_seed=random_seed,
        )
        return result
    except Exception as e:
        logger.exception(f"v9 detect error: {e}")
        raise HTTPException(status_code=500, detail=f"v9检测失败: {str(e)}")


@router.post(
    "/api/v1/v9/batch",
    summary="v9批量检测",
    description="对多个网络执行v9检测。",
)
async def v9_batch_detect(
    network_names: str = "case9,case14,case33bw",
    inject_anomalies: bool = True,
    anomaly_count: int = 3,
):
    """批量v9检测。"""
    results = {}
    errors = {}
    for name in network_names.split(","):
        name = name.strip()
        try:
            results[name] = run_v9_detect(
                network_name=name,
                inject_anomalies=inject_anomalies,
                anomaly_count=anomaly_count,
            )
        except Exception as e:
            errors[name] = str(e)
    return {
        "success": len(errors) == 0,
        "total": len(network_names.split(",")),
        "results": results,
        "errors": errors,
    }
