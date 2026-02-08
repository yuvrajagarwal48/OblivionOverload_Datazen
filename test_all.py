# -*- coding: utf-8 -*-
"""Full test of all simulation modes: Observer, What-If, Registry, State"""
import requests
import json
import sys
import time
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

BASE = "http://localhost:8000"
RESULTS = {}

def safe(val, fmt=".2f"):
    """Safely format a value"""
    if val is None:
        return "None"
    try:
        return f"{val:{fmt}}"
    except:
        return str(val)

# =========================================================================
# TEST GROUP 1: BANK REGISTRY
# =========================================================================
def test_registry_list():
    r = requests.get(f"{BASE}/api/bank/registry")
    assert r.status_code == 200, f"Status {r.status_code}"
    data = r.json()
    assert data["count"] == 46, f"Expected 46 banks, got {data['count']}"
    print(f"  -> {data['count']} banks, source: {data['source']}")
    # Spot check SBI
    sbi = next((b for b in data["banks"] if b["name"] == "State Bank of India"), None)
    assert sbi is not None, "SBI not found"
    print(f"  -> SBI: tier={sbi['tier']}, assets={safe(sbi['initial_assets'])}")
    return True

def test_registry_search():
    r = requests.get(f"{BASE}/api/bank/registry/search", params={"q": "hdfc"})
    assert r.status_code == 200
    data = r.json()
    assert data["count"] >= 1, "No HDFC results"
    print(f"  -> Found {data['count']} results for 'hdfc'")
    return True

def test_registry_single():
    r = requests.get(f"{BASE}/api/bank/registry/0")
    assert r.status_code == 200
    data = r.json()
    b = data.get("bank", data)
    assert b["name"] == "State Bank of India"
    print(f"  -> Bank 0: {b['name']}, CRAR={b.get('metadata',{}).get('crar','?')}")
    return True

# =========================================================================
# TEST GROUP 2: OBSERVER MODE (init, step, state, history)
# =========================================================================
def test_observer_init_real_banks():
    r = requests.post(f"{BASE}/api/simulation/init", json={
        "bank_ids": [0, 8, 41, 42],  # SBI, BoB, HDFC, ICICI
        "episode_length": 30
    })
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
    data = r.json()
    banks = data.get("banks", [])
    assert len(banks) == 4, f"Expected 4 banks, got {len(banks)}"
    names = [b["name"] for b in banks]
    print(f"  -> Initialized: {', '.join(names)}")
    return True

def test_observer_step():
    r = requests.post(f"{BASE}/api/simulation/step")
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:300]}"
    data = r.json()
    step = data.get("current_step", "?")
    done = data.get("is_done", "?")
    rewards = data.get("rewards", {})
    print(f"  -> Step {step}, done={done}, rewards for {len(rewards)} agents")
    return True

def test_observer_multi_step():
    for i in range(4):
        r = requests.post(f"{BASE}/api/simulation/step")
        assert r.status_code == 200, f"Step {i+1} failed: {r.status_code}"
    data = r.json()
    print(f"  -> Ran 4 more steps, now at step {data.get('current_step')}")
    return True

def test_observer_state():
    r = requests.get(f"{BASE}/api/simulation/state")
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:300]}"
    data = r.json()
    ts = data.get("timestep", "?")
    banks = data.get("banks", {})
    exch = data.get("exchanges", [])
    ccps = data.get("ccps", [])
    print(f"  -> Timestep {ts}, {len(banks)} banks, {len(exch)} exchanges, {len(ccps)} CCPs")
    # Check bank details
    for bid, b in list(banks.items())[:2]:
        print(f"     Bank {bid}: equity={safe(b['equity'])}, cash={safe(b['cash'])}, cap_ratio={safe(b['capital_ratio'],'.4f')}")
    return True

def test_observer_status():
    r = requests.get(f"{BASE}/api/simulation/status")
    assert r.status_code == 200, f"Status {r.status_code}"
    data = r.json()
    print(f"  -> Status: {data.get('status')}, step={data.get('current_step')}")
    return True

def test_observer_bank_list():
    r = requests.get(f"{BASE}/api/bank/")
    assert r.status_code == 200
    data = r.json()
    banks = data.get("banks", [])
    print(f"  -> {data.get('count')} banks listed")
    for b in banks[:3]:
        print(f"     {b.get('name','?')}: equity={safe(b.get('equity'))}, risk={safe(b.get('risk_score'),'.1f')}")
    return True

def test_observer_bank_detail():
    r = requests.get(f"{BASE}/api/bank/0")
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
    data = r.json()
    print(f"  -> Bank 0: {data.get('name','?')}, active={data.get('is_active')}")
    return True

def test_observer_analytics():
    r = requests.get(f"{BASE}/api/analytics/risk")
    print(f"  -> Risk analytics: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        for k in list(data.keys())[:4]:
            print(f"     {k}: {safe(data[k])}")
    return r.status_code == 200

def test_observer_market():
    r = requests.get(f"{BASE}/api/market/state")
    print(f"  -> Market state: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"     Price={safe(data.get('price'))}, Vol={safe(data.get('volatility',data.get('vol')),'.4f')}")
    return r.status_code == 200

# =========================================================================
# TEST GROUP 3: WHAT-IF / COUNTERFACTUAL
# =========================================================================
def test_whatif_endpoints_exist():
    r = requests.get(f"{BASE}/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = list(spec.get("paths", {}).keys())
    whatif_paths = [p for p in paths if "whatif" in p.lower() or "what" in p.lower()]
    print(f"  -> What-If related paths: {whatif_paths if whatif_paths else 'None found'}")
    # Also check all available paths
    all_paths = sorted(paths)
    print(f"  -> All API paths ({len(all_paths)}):")
    for p in all_paths:
        methods = list(spec["paths"][p].keys())
        print(f"     {', '.join(m.upper() for m in methods)} {p}")
    return True

def test_whatif_analyze():
    """Try the what-if analyze endpoint with correct schema"""
    # First get the openapi spec to find the right schema
    r = requests.get(f"{BASE}/openapi.json")
    spec = r.json()
    
    # Find what-if related paths
    whatif_paths = {k: v for k, v in spec.get("paths", {}).items() 
                   if "whatif" in k.lower() or "what" in k.lower()}
    
    if not whatif_paths:
        print("  -> No what-if endpoints found")
        return False
    
    # Try each what-if POST endpoint
    for path, methods in whatif_paths.items():
        if "post" in methods:
            # Get the schema for the request body
            post_info = methods["post"]
            body = post_info.get("requestBody", {})
            content = body.get("content", {}).get("application/json", {})
            schema_ref = content.get("schema", {}).get("$ref", "")
            schema_name = schema_ref.split("/")[-1] if schema_ref else "unknown"
            print(f"  -> POST {path} expects: {schema_name}")
            
            # Find schema details
            if schema_name in spec.get("components", {}).get("schemas", {}):
                schema = spec["components"]["schemas"][schema_name]
                props = schema.get("properties", {})
                required = schema.get("required", [])
                print(f"     Properties: {list(props.keys())}")
                print(f"     Required: {required}")
    
    return True

# =========================================================================
# TEST GROUP 4: INIT VARIANTS  
# =========================================================================
def test_init_with_synthetic():
    r = requests.post(f"{BASE}/api/simulation/init", json={
        "bank_ids": [0, 41],
        "synthetic_count": 2,
        "synthetic_stress": "distressed",
        "episode_length": 20
    })
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
    data = r.json()
    banks = data.get("banks", [])
    print(f"  -> {len(banks)} banks (2 real + 2 synthetic distressed)")
    for b in banks:
        print(f"     {b['name']} [Tier {b['tier']}]")
    return True

def test_init_default_random():
    r = requests.post(f"{BASE}/api/simulation/init", json={
        "num_banks": 10,
        "episode_length": 20,
        "seed": 42
    })
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
    data = r.json()
    print(f"  -> Default init: {data.get('message')}")
    
    # Verify bank count
    r2 = requests.get(f"{BASE}/api/bank/")
    banks = r2.json().get("banks", [])
    print(f"  -> {len(banks)} banks in system")
    return True

def test_init_invalid_ids():
    r = requests.post(f"{BASE}/api/simulation/init", json={
        "bank_ids": [0, 999],
        "episode_length": 20
    })
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print(f"  -> Correctly rejected: {r.json().get('detail')}")
    return True

def test_init_by_name():
    r = requests.post(f"{BASE}/api/simulation/init", json={
        "bank_names": ["State Bank of India", "ICICI Bank"],
        "episode_length": 20
    })
    if r.status_code == 200:
        data = r.json()
        banks = data.get("banks", [])
        print(f"  -> Init by name: {[b['name'] for b in banks]}")
        return True
    else:
        print(f"  -> Status {r.status_code}: {r.text[:200]}")
        return False

# =========================================================================
# TEST GROUP 5: SIMULATION FLOW (end-to-end)
# =========================================================================
def test_full_flow():
    """Full observer flow: init -> step 10x -> state -> analytics"""
    # Init
    r = requests.post(f"{BASE}/api/simulation/init", json={
        "bank_ids": [0, 1, 2, 8, 41, 42],
        "episode_length": 50
    })
    assert r.status_code == 200
    
    # Step 10 times
    for i in range(10):
        r = requests.post(f"{BASE}/api/simulation/step")
        assert r.status_code == 200, f"Step {i+1} failed: {r.status_code}"
    
    data = r.json()
    print(f"  -> After 10 steps: step={data.get('current_step')}")
    
    # Get state
    r = requests.get(f"{BASE}/api/simulation/state")
    assert r.status_code == 200
    state = r.json()
    banks = state.get("banks", {})
    active = sum(1 for b in banks.values() if b["status"] == "active")
    print(f"  -> {active}/{len(banks)} banks active")
    for bid, b in banks.items():
        print(f"     Bank {bid}: equity={safe(b['equity'])}, status={b['status']}")
    
    # Bank list
    r = requests.get(f"{BASE}/api/bank/")
    assert r.status_code == 200
    bank_list = r.json()
    print(f"  -> Bank list: {bank_list.get('count')} banks")
    print(f"     Avg capital ratio: {safe(bank_list.get('summary',{}).get('avg_capital_ratio'),'.4f')}")
    print(f"     At risk: {bank_list.get('summary',{}).get('at_risk')}")
    
    return True

# =========================================================================
# RUNNER
# =========================================================================
if __name__ == "__main__":
    # Wait for server
    print("Checking server...")
    for attempt in range(8):
        try:
            r = requests.get(f"{BASE}/api/bank/registry", timeout=2)
            if r.status_code == 200:
                print("Server is UP\n")
                break
        except:
            pass
        print(f"  Waiting... ({attempt+1}/8)")
        time.sleep(3)
    else:
        print("Server not responding!")
        sys.exit(1)
    
    tests = [
        # Registry
        ("1.1 Registry: List all banks",        test_registry_list),
        ("1.2 Registry: Search",                test_registry_search),
        ("1.3 Registry: Single bank",           test_registry_single),
        # Observer mode
        ("2.1 Observer: Init real banks",       test_observer_init_real_banks),
        ("2.2 Observer: Single step",           test_observer_step),
        ("2.3 Observer: Multi step",            test_observer_multi_step),
        ("2.4 Observer: Get state",             test_observer_state),
        ("2.5 Observer: Status",                test_observer_status),
        ("2.6 Observer: Bank list",             test_observer_bank_list),
        ("2.7 Observer: Bank detail",           test_observer_bank_detail),
        ("2.8 Observer: Analytics",             test_observer_analytics),
        ("2.9 Observer: Market state",          test_observer_market),
        # What-If
        ("3.1 WhatIf: Endpoints exist",         test_whatif_endpoints_exist),
        ("3.2 WhatIf: Analyze schema",          test_whatif_analyze),
        # Init variants
        ("4.1 Init: Real + Synthetic",          test_init_with_synthetic),
        ("4.2 Init: Default random",            test_init_default_random),
        ("4.3 Init: Invalid IDs (400)",         test_init_invalid_ids),
        ("4.4 Init: By bank name",              test_init_by_name),
        # Full flow
        ("5.1 Full: 6 banks, 10 steps",         test_full_flow),
    ]
    
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            result = fn()
            RESULTS[name] = result
            print(f"  >> {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"  >> EXCEPTION: {e}")
            RESULTS[name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    for name, passed in RESULTS.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}")
    
    total = len(RESULTS)
    passed = sum(1 for v in RESULTS.values() if v)
    print(f"\n  {passed}/{total} tests passed")
    
    if passed < total:
        print("\n  Failed tests need attention!")
    else:
        print("\n  All simulations working!")
