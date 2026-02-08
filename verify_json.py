import json
with open('data/real_banks.json') as f:
    data = json.load(f)
print(f"Banks: {len(data['banks'])}")
for b in data['banks']:
    if b['name'] in ['State Bank of India', 'HDFC Bank', 'Nainital Bank', 'ICICI Bank']:
        print(json.dumps(b, indent=2))
