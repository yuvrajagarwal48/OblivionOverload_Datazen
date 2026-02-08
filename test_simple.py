"""Simple focused test for the two modes"""
import requests
import json

BASE = "http://localhost:8000"

print("=" * 70)
print("INIT TEST WITH 4 REAL BANKS")
print("=" * 70)

r = requests.post(f"{BASE}/api/simulation/init", json={"bank_ids": [0, 8, 41, 42], "episode_length": 20})
print(f"Init: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Message: {data.get('message')}")
    print(f"Banks returned in init response: {len(data.get('banks', []))}")
    for b in data.get("banks", []):
        print(f"  - {b.get('name')} (ID {b.get('id')}, Tier {b.get('tier')})")
    
    # Now check bank list
    print("\nBank list endpoint:")
    r2 = requests.get(f"{BASE}/api/bank/")
    print(f"Status: {r2.status_code}")
    if r2.status_code == 200:
        data2 = r2.json()
        print(f"Total banks in system: {data2.get('count')}")
        print(f"Banks:")
        for b in data2.get("banks", [])[:6]:
            print(f"  Bank ID {b.get('bank_id')}: name={b.get('name', 'N/A')}, tier={b.get('tier')}, equity={b.get('equity')}")

