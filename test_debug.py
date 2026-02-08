"""Debug step failure"""
import requests
import json

BASE = "http://localhost:8000"

# Init
r = requests.post(f"{BASE}/api/simulation/init", json={"bank_ids": [0, 41], "episode_length": 20})
print(f"Init: {r.status_code}")

# Try step and get full error
r = requests.post(f"{BASE}/api/simulation/step")
print(f"\nStep: {r.status_code}")
if r.status_code != 200:
    print("Response text:")
    print(r.text)
