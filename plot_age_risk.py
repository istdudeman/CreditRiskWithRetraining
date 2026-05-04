import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DEFAULT_FEATURES = {
    'credit_history_length': 9.0, 'income_stability': 1.0, 'interest_rate': 0.05, 
    'salary_volatility': 0.05, 'avg_account_balance': 2816651.0, 'monthly_income': 3203345.0, 
    'macro_unemployment': 0.03, 'medical_check_grade': 'Hijau', 'pension_amount': 0.0, 
    'recent_cash_withdrawals': 0.0, 'outstanding_balance': 1834289.0, 'macro_property_index': 101.7, 
    'employer_risk_score': 0.03, 'digital_payment_frequency': 4.0, 'loan_amount': 5004438.0, 
    'macro_inflation': 0.015, 'macro_interest_rate': 0.054, 'card_spending': 0.0, 
    'total_existing_installments': 0.0, 'product_type': 'Vehicle Loan', 'ltv': 0.0,
    'marital_status': 'Menikah', 'employment_type': 'Pegawai Swasta', 'salary_source': 'Payroll',
    'medical_expense_ratio': 0.1, 'health_risk_score': 0.2, 'age': 40, 'life_insurance_coverage': 0,
    'insurance_status': 'Tidak ada', 'insurance_premium_monthly': 0, 'bureau_score': 650, 
    'past_due_months': 0, 'collateral_flag': '0'
}

ages = list(range(20, 71)) # Ages 20 to 70
probabilities = []
shaps = []

for age in ages:
    features = DEFAULT_FEATURES.copy()
    features['age'] = age
    try:
        response = requests.post("http://localhost:8000/predict", json={"data": [features]})
        if response.status_code == 200:
            res = response.json()["predictions"][0]
            probabilities.append(res['probability'] * 100) # Percentage
            
            age_shap = next((s['value'] for s in res['shap_values'] if 'age' in s['feature'].lower()), 0)
            shaps.append(age_shap)
        else:
            print(f"Error for age {age}")
            probabilities.append(np.nan)
            shaps.append(np.nan)
    except Exception as e:
        print(f"Conn failed: {e}")
        probabilities.append(np.nan)
        shaps.append(np.nan)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(ages, probabilities, marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=5)

# Highlight specific dots
highlight_ages = [30, 45, 60]
for ha in highlight_ages:
    if ha in ages:
        idx = ages.index(ha)
        plt.scatter(ages[idx], probabilities[idx], color='red', s=100, zorder=5)
        plt.annotate(f"{probabilities[idx]:.2f}%", (ages[idx], probabilities[idx]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='red')

plt.title('Probability of Default vs Applicant Age', fontsize=16)
plt.xlabel('Age (Years)', fontsize=14)
plt.ylabel('Base Probability of Default (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvspan(35, 55, color='green', alpha=0.1, label='Lowest Risk Demographic')
plt.legend()

plt.tight_layout()
# Save to the conversation artifact directory
output_path = r"C:\Users\daber\.gemini\antigravity\brain\6c3fa09d-df26-479d-8e0e-fa3e33928cce\age_probability_curve.png"
plt.savefig(output_path, dpi=150)
print(f"Saved plot plot to {output_path}")

df = pd.DataFrame({'Age': ages, 'Default_Probability_Pct': probabilities, 'Age_SHAP_Impact': shaps})
print(df[df['Age'].isin([25, 30, 45, 60, 65])].to_markdown())
