import requests
import json
import random

url='http://localhost:8000/predict'

def test(req_data):
    r = requests.post(url, json={'data': [req_data]})
    res = r.json().get('predictions', [{}])[0]
    return res

results = {'A': None, 'B': None, 'C': None, 'D': None}

for i in range(1000):
    data = {
        'bureau_score': random.choice([300, 400, 500, 600, 700, 800, 850]),
        'age': random.choice([25, 30, 40, 50, 60]),
        'past_due_months': random.choice([0, 1, 2, 3, 4, 5]),
        'marital_status': random.choice(['Menikah', 'Belum menikah', 'Cerai']),
        'employment_type': random.choice(['Pensiunan', 'ASN aktif', 'ASN kontrak', 'PNS Purna Bakti', 'Pegawai BUMN']),
        'salary_source': random.choice(['Taspen', 'Pemerintah Daerah', 'Kementerian']),
        'stability': random.choice(['Stable', 'Unstable']),
        'credit_history_length': random.choice([1, 5, 10, 15, 20]),
        'collateral_flag': random.choice(['1', '0']),
        'life_insurance_coverage': random.choice([0, 5000000, 50000000]),
        'insurance_premium_monthly': random.choice([0, 150000, 500000]),
        'insurance_status': random.choice(['Ada', 'Lapsed', 'Tidak ada']),
        'health_risk_score': random.choice([0.1, 0.5, 0.9]),
        'medical_expense_ratio': random.choice([0.1, 0.3, 0.5])
    }
    
    req_data = data.copy()
    req_data['income_stability'] = 1.0 if req_data['stability'] == 'Stable' else 0.0
    del req_data['stability']
    
    res = test(req_data)
    grade = res.get('grade')
    if grade in results and results[grade] is None:
        results[grade] = data
        
    if all(v is not None for v in results.values()):
        break

print(json.dumps(results, indent=2))
