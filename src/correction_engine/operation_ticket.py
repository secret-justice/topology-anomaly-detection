# -*- coding: utf-8 -*-
"""
v16: 操作票格式输出 — 符合电力调度操作规范
将修正方案格式化为标准操作票
"""
import datetime
import json


def generate_operation_ticket(correction_plan, network_name="unknown"):
    """
    生成标准操作票
    
    Args:
        correction_plan: list of dict, 每个包含 {type, target, action, before_state, after_state}
        network_name: 网络名称
        
    Returns:
        dict: 标准操作票格式
    """
    now = datetime.datetime.now()
    
    ticket = {
        "ticket_id": f"OP-{now.strftime('%Y%m%d%H%M%S')}",
        "network": network_name,
        "created_at": now.isoformat(),
        "status": "pending_approval",
        "summary": {
            "total_operations": len(correction_plan),
            "line_operations": sum(1 for c in correction_plan if c.get("type") == "line"),
            "switch_operations": sum(1 for c in correction_plan if c.get("type") == "switch"),
            "transformer_operations": sum(1 for c in correction_plan if c.get("type") == "transformer"),
        },
        "operations": [],
        "safety_check": {
            "n1_verified": False,
            "kcl_verified": False,
            "notes": [],
        },
    }
    
    for i, correction in enumerate(correction_plan, 1):
        op = {
            "sequence": i,
            "object_type": correction.get("type", "unknown"),
            "object_id": correction.get("target", "unknown"),
            "object_name": correction.get("name", ""),
            "operation": correction.get("action", "unknown"),
            "before_state": correction.get("before_state", {}),
            "after_state": correction.get("after_state", {}),
            "reason": correction.get("reason", ""),
            "confidence": correction.get("confidence", 0.0),
        }
        ticket["operations"].append(op)
    
    return ticket


def format_ticket_text(ticket):
    """将操作票格式化为文本(供调度员阅读)"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"配电网拓扑修正操作票")
    lines.append(f"票号: {ticket['ticket_id']}")
    lines.append(f"网络: {ticket['network']}")
    lines.append(f"时间: {ticket['created_at']}")
    lines.append(f"操作总数: {ticket['summary']['total_operations']}")
    lines.append("=" * 60)
    lines.append("")
    
    for op in ticket["operations"]:
        lines.append(f"  序号 {op['sequence']}: [{op['object_type']}] {op['object_name'] or op['object_id']}")
        lines.append(f"    操作: {op['operation']}")
        lines.append(f"    原因: {op['reason']}")
        lines.append(f"    置信度: {op['confidence']:.2f}")
        lines.append(f"    操作前: {json.dumps(op['before_state'], ensure_ascii=False)}")
        lines.append(f"    操作后: {json.dumps(op['after_state'], ensure_ascii=False)}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"安全校验: N-1={'通过' if ticket['safety_check']['n1_verified'] else '待校验'}")
    lines.append(f"          KCL={'通过' if ticket['safety_check']['kcl_verified'] else '待校验'}")
    lines.append("=" * 60)
    
    return "\n".join(lines)
