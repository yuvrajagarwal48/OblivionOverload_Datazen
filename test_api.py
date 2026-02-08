"""Quick API test script"""
import requests
import json
import sys
import time

BASE = "http://localhost:8000"

def test_registry():
    print("=" * 60)
    print("TEST 1: GET /api/bank/registry")
    print("=" * 60)
    r = requests.get(f"{BASE}/api/bank/registry")
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Count: {data.get('count')}")
    print(f"  Source: {data.get('source')}")
    for b in data.get("banks", [])[:5]:
        print(f"    Bank {b['id']}: {b['name']} [Tier {b['tier']}] cash={b['initial_cash']:.0f} assets={b['initial_assets']:.0f}")
    print()
    return r.status_code == 200

def test_registry_search():
    print("=" * 60)
    print("TEST 2: GET /api/bank/registry/search?q=state")
    print("=" * 60)
    r = requests.get(f"{BASE}/api/bank/registry/search", params={"q": "state"})
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Results: {data.get('count')}")
    for b in data.get("results", []):
        print(f"    Bank {b['id']}: {b['name']}")
    print()
    return r.status_code == 200

def test_registry_single():
    print("=" * 60)
    print("TEST 3: GET /api/bank/registry/0  (SBI)")
    print("=" * 60)
    r = requests.get(f"{BASE}/api/bank/registry/0")
    data = r.json()
    print(f"  Status: {r.status_code}")
    b = data.get("bank", data)
    print(f"  Name: {b.get('name')}")
    print(f"  Tier: {b.get('tier')}")
    print(f"  Cash: {b.get('initial_cash')}")
    print(f"  Assets: {b.get('initial_assets')}")
    print(f"  Ext Liab: {b.get('initial_external_liabilities')}")
    meta = b.get("metadata", {})
    print(f"  CRAR: {meta.get('crar')}, NPA: {meta.get('net_npa_ratio')}, RoA: {meta.get('roa')}")
    print()
    return r.status_code == 200

def test_init_with_real_banks():
    print("=" * 60)
    print("TEST 4: POST /api/simulation/init (real banks: SBI, BoB, HDFC, ICICI)")
    print("=" * 60)
    payload = {
        "bank_ids": [0, 8, 41, 42],  # SBI, BoB, HDFC, ICICI
        "episode_length": 50
    }
    r = requests.post(f"{BASE}/api/simulation/init", json=payload)
    data = r.json()
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  Message: {data.get('message')}")
        print(f"  Num banks: {data.get('num_banks')}")
        banks = data.get("banks", [])
        for b in banks:
            print(f"    Bank {b.get('id')}: {b.get('name', 'N/A')} [Tier {b.get('tier', '?')}]")
    else:
        print(f"  Error: {data}")
    print()
    return r.status_code == 200

def test_init_with_synthetic():
    print("=" * 60)
    print("TEST 5: POST /api/simulation/init (real + synthetic)")
    print("=" * 60)
    payload = {
        "bank_ids": [0, 41, 42],  # SBI, HDFC, ICICI
        "synthetic_count": 3,
        "synthetic_stress": "distressed",
        "episode_length": 50
    }
    r = requests.post(f"{BASE}/api/simulation/init", json=payload)
    data = r.json()
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  Message: {data.get('message')}")
        print(f"  Num banks: {data.get('num_banks')}")
        banks = data.get("banks", [])
        for b in banks:
            print(f"    Bank {b.get('id')}: {b.get('name', 'N/A')} [Tier {b.get('tier', '?')}]")
    else:
        print(f"  Error: {data}")
    print()
    return r.status_code == 200

def test_bank_list_after_init():
    print("=" * 60)
    print("TEST 6: GET /api/bank/ (after init - should show names)")
    print("=" * 60)
    r = requests.get(f"{BASE}/api/bank/")
    data = r.json()
    print(f"  Status: {r.status_code}")
    banks = data.get("banks", [])
    print(f"  Total banks: {len(banks)}")
    for b in banks[:6]:
        print(f"    Bank {b.get('id')}: name={b.get('name', 'N/A')} cash={b.get('cash', '?')} assets={b.get('total_assets', '?')} equity={b.get('equity', '?')}")
    print()
    return r.status_code == 200

def test_init_default():
    print("=" * 60)
    print("TEST 7: POST /api/simulation/init (default - no real banks)")
    print("=" * 60)
    payload = {
        "num_banks": 10,
        "episode_length": 50,
        "seed": 42
    }
    r = requests.post(f"{BASE}/api/simulation/init", json=payload)
    data = r.json()
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  Message: {data.get('message')}")
        print(f"  Num banks: {data.get('num_banks')}")
    else:
        print(f"  Error: {data}")
    print()
    return r.status_code == 200

def test_invalid_bank_ids():
    print("=" * 60)
    print("TEST 8: POST /api/simulation/init (invalid bank IDs)")
    print("=" * 60)
    payload = {
        "bank_ids": [0, 999, 1000],
        "episode_length": 50
    }
    r = requests.post(f"{BASE}/api/simulation/init", json=payload)
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(data, indent=2)[:300]}")
    print()
    return r.status_code == 400  # Should be 400 for invalid IDs

if __name__ == "__main__":
    results = {}
    
    # Check server is up
    print("Checking server...")
    for attempt in range(5):
        try:
            requests.get(f"{BASE}/api/bank/registry", timeout=3)
            print("Server is up!\n")
            break
        except:
            print(f"  Attempt {attempt+1}/5 - waiting...")
            time.sleep(3)
    else:
        print("Server not responding. Start it first.")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Registry List", test_registry),
        ("Registry Search", test_registry_search),
        ("Registry Single", test_registry_single),
        ("Init Real Banks", test_init_with_real_banks),
        ("Bank List After Init", test_bank_list_after_init),
        ("Init Real+Synthetic", test_init_with_synthetic),
        ("Bank List After Init 2", test_bank_list_after_init),
        ("Init Default", test_init_default),
        ("Invalid Bank IDs", test_invalid_bank_ids),
    ]
    
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}\n")
            results[name] = False
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")
