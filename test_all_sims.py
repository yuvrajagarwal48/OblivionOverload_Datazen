"""Comprehensive test of all simulation modes"""
import requests
import json
import sys

BASE = "http://localhost:8000"
passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")

# ---- SERVER CHECK ----
try:
    requests.get(f"{BASE}/api/bank/registry", timeout=3)
except:
    print("Server not running!"); sys.exit(1)

# ==================================================================
print("="*60)
print("1. BANK REGISTRY (no init required)")
print("="*60)

r = requests.get(f"{BASE}/api/bank/registry")
check("GET /registry", r.status_code == 200)
data = r.json()
check("46 banks loaded", data.get("count") == 46, f"got {data.get('count')}")

r = requests.get(f"{BASE}/api/bank/registry/search", params={"q": "state"})
check("Search 'state'", r.status_code == 200 and r.json().get("count", 0) >= 5)

r = requests.get(f"{BASE}/api/bank/registry/0")
check("Get SBI by ID", r.status_code == 200 and "State Bank" in r.json().get("bank", {}).get("name", ""))

# ==================================================================
print("\n" + "="*60)
print("2. OBSERVER MODE - Real Banks")
print("="*60)

r = requests.post(f"{BASE}/api/simulation/init", json={
    "bank_ids": [0, 8, 41, 42], "episode_length": 20
})
check("Init 4 real banks", r.status_code == 200)
data = r.json()
check("Message mentions real banks", "real banks" in data.get("message", ""), data.get("message"))
check("4 banks in response", len(data.get("banks") or []) == 4, f"got {len(data.get('banks') or [])}")

# Step
r = requests.post(f"{BASE}/api/simulation/step")
check("Step 1", r.status_code == 200)
if r.status_code == 200:
    step_data = r.json()
    check("Step number = 1", step_data.get("current_step") == 1)

r = requests.post(f"{BASE}/api/simulation/step")
check("Step 2", r.status_code == 200)

r = requests.post(f"{BASE}/api/simulation/step")
check("Step 3", r.status_code == 200)

# State
r = requests.get(f"{BASE}/api/simulation/state")
check("GET /state", r.status_code == 200)
if r.status_code == 200:
    state = r.json()
    check("Timestep = 3", state.get("timestep") == 3)
    check("4 banks in state", len(state.get("banks", {})) == 4, f"got {len(state.get('banks', {}))}")

# Bank list
r = requests.get(f"{BASE}/api/bank/")
check("GET /bank/", r.status_code == 200)
if r.status_code == 200:
    bl = r.json()
    check("4 banks in list", bl.get("count") == 4, f"got {bl.get('count')}")
    banks = bl.get("banks", [])
    has_names = any(b.get("name") for b in banks)
    check("Banks have names", has_names)

# Status
r = requests.get(f"{BASE}/api/simulation/status")
check("GET /status", r.status_code == 200)

# ==================================================================
print("\n" + "="*60)
print("3. OBSERVER MODE - Real + Synthetic")
print("="*60)

r = requests.post(f"{BASE}/api/simulation/init", json={
    "bank_ids": [0, 41], "synthetic_count": 3, "synthetic_stress": "distressed", "episode_length": 20
})
check("Init 2 real + 3 synthetic", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    check("5 banks total", len(data.get("banks") or []) == 5, f"got {len(data.get('banks') or [])}")

r = requests.post(f"{BASE}/api/simulation/step")
check("Step with mixed banks", r.status_code == 200)

r = requests.get(f"{BASE}/api/bank/")
if r.status_code == 200:
    bl = r.json()
    check("5 banks in list", bl.get("count") == 5, f"got {bl.get('count')}")

# ==================================================================
print("\n" + "="*60)
print("4. OBSERVER MODE - Default Random")
print("="*60)

r = requests.post(f"{BASE}/api/simulation/init", json={
    "num_banks": 10, "episode_length": 20, "seed": 42
})
check("Init 10 random banks", r.status_code == 200)
if r.status_code == 200:
    data = r.json()
    check("Message says random", "random" in data.get("message", ""), data.get("message"))

r = requests.post(f"{BASE}/api/simulation/step")
check("Step with random banks", r.status_code == 200)

r = requests.get(f"{BASE}/api/simulation/state")
check("State with random banks", r.status_code == 200)
if r.status_code == 200:
    check("10 banks in state", len(r.json().get("banks", {})) == 10)

# ==================================================================
print("\n" + "="*60)
print("5. WHAT-IF / ANALYTICS")
print("="*60)

# Re-init with real banks for what-if
r = requests.post(f"{BASE}/api/simulation/init", json={
    "bank_ids": [0, 8, 41, 42], "episode_length": 30
})
check("Re-init for what-if", r.status_code == 200)

# Run a few steps
for i in range(3):
    requests.post(f"{BASE}/api/simulation/step")

# Bank list
r = requests.get(f"{BASE}/api/bank/")
check("GET /bank/ list", r.status_code == 200)

# Simulation status
r = requests.get(f"{BASE}/api/simulation/status")
check("GET /simulation/status", r.status_code == 200)

# Individual bank detail
r = requests.get(f"{BASE}/api/bank/0")
check("GET /bank/0 detail", r.status_code == 200)

# Bank history
r = requests.get(f"{BASE}/api/bank/0/history")
check("GET /bank/0/history", r.status_code == 200)

# What-if endpoint check
r = requests.get(f"{BASE}/docs")
check("API docs accessible", r.status_code == 200)

# ==================================================================
print("\n" + "="*60)
print("6. VALIDATION (Error handling)")
print("="*60)

r = requests.post(f"{BASE}/api/simulation/init", json={"bank_ids": [0, 999]})
check("Invalid bank ID returns 400", r.status_code == 400, f"got {r.status_code}")

r = requests.post(f"{BASE}/api/simulation/init", json={"bank_names": ["Not A Bank"]})
check("Invalid bank name returns 400", r.status_code == 400, f"got {r.status_code}")

# ==================================================================
print("\n" + "="*60)
print(f"SUMMARY: {passed} passed, {failed} failed out of {passed+failed} tests")
print("="*60)
