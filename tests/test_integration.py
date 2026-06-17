import sys, json
sys.path.insert(0, r"E:\项目大全\电力拓扑图修正\02_算法代码")

from api.service import run_detect, run_correct, get_health, list_available_networks

print("=" * 60)
print("  集成测试: API 服务层")
print("=" * 60)

# 1. Health
h = get_health()
print(f"\n[1] Health: {h}")

# 2. Networks
nets = list_available_networks()
print(f"[2] Networks: {list(nets.keys())}")

# 3. Detect (无注入)
r1 = run_detect(network_name="case33bw", inject_anomalies=False)
print(f"[3] Detect (no inject): anomaly_count={r1.get('anomaly_count', '?')}, success={r1.get('success')}")

# 4. Detect (有注入)
r2 = run_detect(network_name="case33bw", inject_anomalies=True, anomaly_count=5, random_seed=42)
print(f"[4] Detect (inject 5): anomaly_count={r2.get('anomaly_count', '?')}, success={r2.get('success')}")
if 'summary' in r2:
    s = r2['summary']
    print(f"    Summary: {json.dumps(s, ensure_ascii=False, default=str)[:200]}")

# 5. Correct
r3 = run_correct(network_name="case33bw", inject_anomalies=True, anomaly_count=3, random_seed=42)
print(f"[5] Correct: anomaly_count={r3.get('anomaly_count','?')}, correction_count={r3.get('correction_count','?')}")

# 6. 多网络测试
for net_name in ["case14", "example_simple"]:
    try:
        r = run_detect(network_name=net_name, inject_anomalies=True, anomaly_count=3, random_seed=42)
        print(f"[6] {net_name}: anomaly_count={r.get('anomaly_count','?')}, success={r.get('success')}")
    except Exception as e:
        print(f"[6] {net_name}: ERROR - {e}")

print("\n" + "=" * 60)
print("  集成测试完成")
print("=" * 60)
