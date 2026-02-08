"""
Complete bank data extraction from APB30091213F.pdf
Extracts all individual banks, using text extraction.
Maps RBI data to simulation balance sheet schema.
"""
import pdfplumber
import json
import re
import os

# Bank pages mapping from TOC (0-indexed PDF pages)
BANK_PAGES = {
    # SBI & Associates
    "State Bank of India": 14,
    "State Bank of Bikaner & Jaipur": 15,
    "State Bank of Hyderabad": 16,
    "State Bank of Mysore": 17,
    "State Bank of Patiala": 18,
    "State Bank of Travancore": 19,
    # Nationalised Banks
    "Allahabad Bank": 20,
    "Andhra Bank": 21,
    "Bank of Baroda": 22,
    "Bank of India": 23,
    "Bank of Maharashtra": 24,
    "Canara Bank": 25,
    "Central Bank of India": 26,
    "Corporation Bank": 27,
    "Dena Bank": 28,
    "IDBI Bank Ltd.": 29,
    "Indian Bank": 30,
    "Indian Overseas Bank": 31,
    "Oriental Bank of Commerce": 32,
    "Punjab and Sind Bank": 33,
    "Punjab National Bank": 34,
    "Syndicate Bank": 35,
    "UCO Bank": 36,
    "Union Bank of India": 37,
    "United Bank of India": 38,
    "Vijaya Bank": 39,
    # Old Private Sector Banks
    "Catholic Syrian Bank": 40,
    "City Union Bank": 41,
    "Dhanlaxmi Bank": 42,
    "Federal Bank": 43,
    "ING Vysya Bank": 44,
    "Jammu & Kashmir Bank": 45,
    "Karnataka Bank": 46,
    "Karur Vysya Bank": 47,
    "Lakshmi Vilas Bank": 48,
    "Nainital Bank": 49,
    "Ratnakar Bank": 50,
    "South Indian Bank": 51,
    "Tamilnad Mercantile Bank": 52,
    # New Private Sector Banks
    "Axis Bank": 53,
    "Development Credit Bank": 54,
    "HDFC Bank": 55,
    "ICICI Bank": 56,
    "IndusInd Bank": 57,
    "Kotak Mahindra Bank": 58,
    "Yes Bank": 59,
}


def parse_bank_page(page_text, bank_name):
    """Parse a single bank's page text and extract 2012-13 data using label matching."""
    lines = page_text.strip().split('\n')

    # Label patterns to match each row reliably
    label_patterns = {
        'offices': r'No\.\s*of\s*offices',
        'employees': r'No\.\s*of\s*employees',
        'business_per_employee': r'Business\s*per\s*employee',
        'profit_per_employee': r'Profit\s*per\s*employee',
        'capital_reserves': r'Capital\s*and\s*Reserves',
        'deposits': r'^Deposits',
        'investments': r'^Investments',
        'advances': r'^Advances',
        'interest_income': r'^Interest\s*Income',
        'other_income': r'^Other\s*income',
        'interest_expended': r'^Interest\s*expended',
        'operating_expenses': r'^Operating\s*expenses',
        'net_interest_margin': r'^Net\s*Interest\s*Margin',
        'cost_of_funds': r'^Cost\s*of\s*Funds',
        'return_on_advances_adj': r'^Return\s*on\s*advances\s*adjusted',
        'wages_pct': r'^Wages\s*as',
        'return_on_equity': r'^Return\s*on\s*Equity',
        'return_on_assets': r'^Return\s*on\s*Assets',
        'crar': r'^CRAR',
        'net_npa_ratio': r'^Net\s*NPA',
    }

    data = {}
    for line in lines:
        line = line.strip()
        for key, pattern in label_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                numbers = re.findall(r'[\d,]+\.?\d*', line)
                if len(numbers) >= 5:
                    val_str = numbers[4].replace(',', '')  # 2012-13 is the 5th number
                    try:
                        data[key] = float(val_str)
                    except ValueError:
                        data[key] = 0.0
                elif numbers:
                    val_str = numbers[-1].replace(',', '')
                    try:
                        data[key] = float(val_str)
                    except ValueError:
                        data[key] = 0.0
                break

    return data


def extract_all_banks():
    """Extract data for all banks from the PDF."""
    pdf = pdfplumber.open('APB30091213F.pdf')
    all_banks = {}

    for bank_name, page_idx in BANK_PAGES.items():
        try:
            page = pdf.pages[page_idx]
            text = page.extract_text()
            if text:
                data = parse_bank_page(text, bank_name)
                all_banks[bank_name] = data
                print(f"OK: {bank_name} -> {len(data)} items extracted")
                if 'deposits' in data and 'investments' in data:
                    print(f"    Deposits: {data.get('deposits', 0):,.0f}, "
                          f"Investments: {data.get('investments', 0):,.0f}, "
                          f"Advances: {data.get('advances', 0):,.0f}, "
                          f"Capital: {data.get('capital_reserves', 0):,.0f}")
            else:
                print(f"FAIL: {bank_name} -> No text on page {page_idx}")
        except Exception as e:
            print(f"ERROR: {bank_name} -> {e}")

    pdf.close()
    return all_banks


def map_to_simulation_schema(bank_name, raw_data, bank_id):
    """
    Map extracted RBI data to simulation Bank schema.

    Bank constructor needs: bank_id, tier, initial_cash, initial_assets,
                           initial_external_liabilities, min_capital_ratio

    All amounts in PDF are in Millions of Rupees.
    Convert to crores by dividing by 10 (1 crore = 10 million).
    Then scale down for simulation range.
    """
    MILLION_TO_CRORE = 10.0

    deposits = raw_data.get('deposits', 0) / MILLION_TO_CRORE
    investments = raw_data.get('investments', 0) / MILLION_TO_CRORE
    advances = raw_data.get('advances', 0) / MILLION_TO_CRORE
    capital_reserves = raw_data.get('capital_reserves', 0) / MILLION_TO_CRORE
    roa = raw_data.get('return_on_assets', 1.0)
    crar = raw_data.get('crar', 12.0)
    npa_ratio = raw_data.get('net_npa_ratio', 2.0)
    roe = raw_data.get('return_on_equity', 10.0)

    total_assets = deposits + capital_reserves
    cash_like = total_assets - investments - advances
    initial_cash = max(cash_like * 0.25, capital_reserves * 0.1)
    
    # Illiquid assets = Investments + Advances (both are non-cash assets on the balance sheet)
    # In the simulation, illiquid_assets is the main asset class
    initial_assets = investments + advances
    
    initial_external_liabilities = deposits

    CORE_BANKS = {
        "State Bank of India", "ICICI Bank", "HDFC Bank", "Bank of Baroda",
        "Punjab National Bank", "Canara Bank", "Bank of India", "Union Bank of India",
        "IDBI Bank Ltd.", "Central Bank of India", "Indian Overseas Bank", "Axis Bank"
    }
    tier = 1 if bank_name in CORE_BANKS else 2

    # Scale: config has cash [5000,15000], assets [10000,30000], ext_liab [3000,10000]
    # SBI deposits ~ 1202739 crores -> target ~ 20000 => scale_factor = 60
    SCALE_FACTOR = 60.0

    initial_cash_scaled = initial_cash / SCALE_FACTOR
    initial_assets_scaled = initial_assets / SCALE_FACTOR
    initial_ext_liab_scaled = initial_external_liabilities / SCALE_FACTOR

    initial_cash_scaled = max(initial_cash_scaled, 500)
    initial_assets_scaled = max(initial_assets_scaled, 1000)
    initial_ext_liab_scaled = max(initial_ext_liab_scaled, 500)

    return {
        "bank_id": bank_id,
        "name": bank_name,
        "tier": tier,
        "initial_cash": round(initial_cash_scaled, 2),
        "initial_assets": round(initial_assets_scaled, 2),
        "initial_external_liabilities": round(initial_ext_liab_scaled, 2),
        "min_capital_ratio": 0.08,
        "metadata": {
            "crar": crar,
            "net_npa_ratio": npa_ratio,
            "return_on_equity": roe,
            "return_on_assets": roa,
            "deposits_crores": round(deposits, 2),
            "investments_crores": round(investments, 2),
            "advances_crores": round(advances, 2),
            "capital_reserves_crores": round(capital_reserves, 2),
            "source": "RBI APB 2012-13",
            "data_year": "2012-13"
        }
    }


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    print("Extracting bank data from PDF...")
    raw_banks = extract_all_banks()

    print(f"\n{'='*80}")
    print(f"Total banks extracted: {len(raw_banks)}")
    print(f"{'='*80}\n")

    sim_banks = []
    for i, (name, raw) in enumerate(raw_banks.items()):
        mapped = map_to_simulation_schema(name, raw, i)
        sim_banks.append(mapped)
        print(f"{i:2d}. {name:40s} | tier={mapped['tier']} | "
              f"cash={mapped['initial_cash']:10.2f} | "
              f"assets={mapped['initial_assets']:10.2f} | "
              f"ext_liab={mapped['initial_external_liabilities']:10.2f}")

    output = {
        "description": "Real Indian bank data extracted from RBI A Profile of Banks 2012-13",
        "source": "Reserve Bank of India - APB30091213F.pdf",
        "data_year": "2012-13",
        "unit_note": "Simulation values scaled from actual crore values (scale_factor=60)",
        "banks": sim_banks
    }

    with open('data/real_banks.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(sim_banks)} banks to data/real_banks.json")
