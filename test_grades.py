import requests
import json

url='http://localhost:8000/predict'

def test(data):
    req_data = data.copy()
    req_data['income_stability'] = 1.0 if req_data['stability'] == 'Stable' else 0.0
    del req_data['stability']
    r = requests.post(url, json={'data': [req_data]})
    res = r.json().get('predictions', [{}])[0]
    return f"Grade {res.get('grade')}: {res.get('probability', 0)*100:.1f}%"

base = {
    'marital_status': 'Menikah', 
    'employment_type': 'Pegawai Swasta', 
    'salary_source': 'Taspen', 
    'collateral_flag': '1', 
    'life_insurance_coverage': 5000000, 
    'insurance_premium_monthly': 150000, 
    'insurance_status': 'Ada', 
    'health_risk_score': 0.1, 
    'medical_expense_ratio': 0.1
}

A = {**base, 'bureau_score': 850, 'age': 40, 'past_due_months': 0, 'stability': 'Stable', 'credit_history_length': 15}
B = {**base, 'bureau_score': 720, 'age': 30, 'past_due_months': 0, 'stability': 'Stable', 'credit_history_length': 10}
C = {**base, 'bureau_score': 600, 'age': 25, 'past_due_months': 1, 'stability': 'Unstable', 'credit_history_length': 5}
D = {**base, 'bureau_score': 400, 'age': 25, 'past_due_months': 3, 'stability': 'Unstable', 'credit_history_length': 1}

print('A:', test(A))
print('B:', test(B))
print('C:', test(C))
print('D:', test(D))
