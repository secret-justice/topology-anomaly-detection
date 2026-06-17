# -*- coding: utf-8 -*-
"""
配电网拓扑智能识别与修正系统 v17 - REST API
OpenAPI documentation: http://localhost:8000/docs
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from api.routes import router
from api.v19_diagnosis import router as v19_router
from api.v20_diagnosis import router as v20_router
from api.v9_service import run_v9_detect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with OpenAPI docs
app = FastAPI(
    title="配电网拓扑智能识别与修正系统",
    description="""
    ## 功能概述
    
    基于多层检测架构的配电网拓扑智能识别与修正系统，支持28种异常类型检测。
    
    ### 检测层
    - **Layer 1**: Rule Engine (规则引擎)
    - **Layer 2**: State Estimation (状态估计)
    - **Layer 3**: GNN (图神经网络)
    - **Layer 4**: v8 Specialized Detectors
    - **Layer 5**: v9 Specialized Detectors
    - **Layer 6**: v16 New Types (bus_section_mismatch, bypass_operation, load_transfer_residual)
    
    ### 异常类型 (28种)
    原始5种 + v8扩充10种 + v9新增10种 + v16新增3种
    
    ### 技术栈
    - PandaPower (电力系统仿真)
    - PyTorch (GNN模型)
    - FastAPI (REST API)
    """,
    version="17.0.0",
    contact={
        "name": "CP-202606 Team",
        "url": "https://github.com/cp202606/power-topology",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v9", tags=["Detection"])
app.include_router(v19_router)
app.include_router(v20_router)

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle WebSocket messages
            logger.info(f"WebSocket received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend UI."""
    frontend_path = Path(__file__).parent.parent.parent / "03_前端界面" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found</h1>")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "17.0.0"}


# ============================================================
# LLM辅助诊断端点 (v18)
# ============================================================

@app.post("/api/v9/diagnose", tags=["Diagnosis"])
async def diagnose_network(request: dict):
    """
    运行检测并生成智能诊断报告
    
    输入: {"network_name": "case33bw", "inject_anomalies": true, "anomaly_count": 3}
    输出: 检测结果 + 关联分析 + 修正建议 + 诊断报告
    """
    from llm_assistant import DiagnosticReporter
    
    network_name = request.get("network_name", "case33bw")
    inject = request.get("inject_anomalies", True)
    n_anomalies = request.get("anomaly_count", 3)
    seed = request.get("random_seed", 42)
    
    # 1. 运行检测
    result = run_v9_detect(
        network_name=network_name,
        inject_anomalies=inject,
        anomaly_count=n_anomalies,
        random_seed=seed,
    )
    
    detections = result.get("detections", [])
    
    # 2. 生成诊断报告
    reporter = DiagnosticReporter(use_llm=False)
    diagnosis = reporter.generate(
        anomalies=detections,
        network_context={
            "bus_count": result.get("bus_count", 0),
            "line_count": result.get("line_count", 0),
            "network_name": network_name,
        }
    )
    
    return {
        "detection": result,
        "diagnosis": diagnosis,
    }


@app.post("/api/v9/diagnose/markdown", tags=["Diagnosis"])
async def diagnose_markdown(request: dict):
    """生成Markdown格式的诊断报告"""
    from llm_assistant import DiagnosticReporter
    
    network_name = request.get("network_name", "case33bw")
    result = run_v9_detect(network_name=network_name)
    detections = result.get("detections", [])
    
    reporter = DiagnosticReporter(use_llm=False)
    report = reporter.generate(anomalies=detections, network_context={
        "bus_count": result.get("bus_count", 0),
        "line_count": result.get("line_count", 0),
        "network_name": network_name,
    })
    
    return {"markdown": reporter.to_markdown(report)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


