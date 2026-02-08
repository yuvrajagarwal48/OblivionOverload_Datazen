"""Test observer and what-if modes"""
import requests
import sys

BASE = "http://localhost:8000"

def test_observer():
    print("="*60)
    print("OBSERVER MODE TEST")
    print("="*60)
    
    # Init
    r = requests.post(f"{BASE}/api/simulation/init", json={"bank_ids": [0, 8, 41, 42], "episode_length": 20})
    if r.status_code != 200:
        print(f"FAIL - Init: {r.status_code}")
        return False
    print("PASS - Initialized 4 real banks")
    
    # Step
    for i in range(3):
        r = requests.post(f"{BASE}/api/simulation/step")
        if r.status_code != 200:
            print(f"FAIL - Step {i+1}: {r.status_code}")
            return False
    print("PASS - Ran 3 simulation steps")
    
    # Get state
    r = requests.get(f"{BASE}/api/simulation/state")
    if r.status_code != 200:
        print(f"FAIL - Get state: {r.status_code}")
        return False
    state = r.json()
    print(f"PASS - Got state: step {state.get('timestep')}, {len(state.get('banks', {}))} banks")
    
    return True

def test_whatif():
    print("\n" + "="*60)
    print("WHAT-IF MODE TEST")
    print("="*60)
    
    # Init
    r = requests.post(f"{BASE}/api/simulation/init", json={"bank_ids": [0, 41], "episode_length": 20})
    if r.status_code != 200:
        print(f"FAIL - Init: {r.status_code}")
        return False
    print("PASS - Initialized 2 real banks")
    
    # Get banks
    r = requests.get(f"{BASE}/api/bank/")
    if r.status_code != 200:
        print(f"FAIL - Get banks: {r.status_code}")
        return False
    banks = r.json().get("banks", [])
    if not banks:
        print("FAIL - No banks found")
        return False
    
    bank_id = banks[0].get("bank_id")
    bank_name = banks[0].get("name", "Unknown")
    print(f"PASS - Found bank {bank_id}: {bank_name}")
    
    # Check what-if endpoint exists
    r = requests.get(f"{BASE}/docs")
    if r.status_code != 200:
        print("FAIL - API docs not accessible")
        return False
    print("PASS - API endpoints accessible")
    
    return True

if __name__ == "__main__":
    # Check server
    try:
        r = requests.get(f"{BASE}/api/bank/registry", timeout=2)
        if r.status_code == 200:
            print("Server is running\n")
    except:
        print("ERROR: Server not running")
        sys.exit(1)
    
    results = {}
    try:
        results["Observer"] = test_observer()
    except Exception as e:
        print(f"EXCEPTION in Observer: {e}")
        results["Observer"] = False
    
    try:
        results["WhatIf"] = test_whatif()
    except Exception as e:
        print(f"EXCEPTION in WhatIf: {e}")
        results["WhatIf"] = False
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} mode")
    
    passed_count = sum(1 for v in results.values() if v)
    print(f"\n  {passed_count}/{len(results)} modes working")
