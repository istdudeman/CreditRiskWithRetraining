import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker('id_ID')

def generate_credit_risk_data(n_samples=10000, seed=42):
    """
    Generates a synthetic credit risk dataset tailored for Indonesian banking (Mandiri Taspen style).
    """
    np.random.seed(seed)
    random.seed(seed)
    
    data = {}


    data['customer_id'] = [f'CUST{i:05d}' for i in range(n_samples)]
    data['name_masked'] = [fake.name() for _ in range(n_samples)] #

    # var demografik
    data['age'] = np.random.randint(45, 70, n_samples)
    data['gender'] = np.random.choice(['Laki-laki', 'Perempuan'], n_samples, p=[0.55, 0.45])
    data['marital_status'] = np.random.choice(['Menikah', 'Cerai', 'Belum menikah'], n_samples, p=[0.7, 0.2, 0.1])
    
    # employment dan detail pensiun
    employment_types = ['Pensiunan', 'ASN aktif', 'ASN kontrak', 'PNS Purna Bakti', 'Pegawai BUMN']
    data['employment_type'] = np.random.choice(employment_types, n_samples, p=[0.6, 0.2, 0.05, 0.1, 0.05])
    data['pension_status'] = np.random.choice(['Pensiun', 'Aktif', 'Menjelang pensiun'], n_samples, p=[0.7, 0.2, 0.1])
    
    # Pension/Income 
    base_income = np.random.normal(5000000, 1500000, n_samples)
    data['monthly_income'] = base_income * (1 + (data['age'] < 55) * 0.5) # if younger high base income
    data['pension_amount'] = np.where(data['employment_type'] == 'Pensiunan', 
                                     data['monthly_income'] * 0.8, 0).round(0)
    
    data['salary_source'] = np.random.choice(['Taspen', 'Pemerintah Daerah', 'Kementerian'], n_samples)
    
    # Assign employer risk score (lower better, govt employed lower risk)
    risk_map = {'Pensiunan': 0.1, 'ASN aktif': 0.05, 'ASN kontrak': 0.2, 'PNS Purna Bakti': 0.15, 'Pegawai BUMN': 0.08}
    data['employer_risk_score'] = np.array([
        risk_map[e] + np.random.uniform(-0.02, 0.02) 
        for e in data['employment_type']
    ])
    data['employer_type'] = np.random.choice(['Government', 'BUMN', 'Swasta'], n_samples, p=[0.7, 0.2, 0.1])

    # Income and financial cap
    data['income_stability'] = np.clip(np.random.normal(0.8, 0.15, n_samples), 0.5, 1.0)
    data['medical_expense_ratio'] = np.clip(0.1 + (data['age'] - 40) * 0.01 + np.random.normal(0, 0.05, n_samples), 0.05, 0.45)
    data['salary_volatility'] = np.clip(0.1 + (1 - data['income_stability']) * 0.5 + np.random.normal(0, 0.05, n_samples), 0.05, 0.5)
    data['avg_account_balance'] = data['monthly_income'] * np.random.uniform(0.5, 3.0, n_samples)
    data['total_existing_installments'] = np.random.randint(0, 5, n_samples)
    data['other_active_loans_count'] = np.random.randint(0, 3, n_samples)
    data['DTI'] = np.clip(data['total_existing_installments'] * 0.05 + data['monthly_income'] * 0.000005 * np.random.rand(n_samples), 0.1, 0.6)

    # Credit behaviour and ojk data
    data['bureau_score'] = np.clip(np.random.normal(700, 100, n_samples), 500, 850).astype(int)
    data['past_due_months'] = np.random.poisson(0.5, n_samples)
    data['credit_history_length'] = np.clip(np.random.normal(10, 5, n_samples), 1, 30).astype(int)

    # Health risk and medical data
    data['health_risk_score'] = np.clip(0.2 + (data['age'] - 40) * 0.015 + np.random.normal(0, 0.1, n_samples), 0.0, 1.0)
    data['has_chronic_condition'] = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    data['medical_check_grade'] = np.random.choice(['Hijau', 'Kuning', 'Merah'], n_samples, p=[0.5, 0.35, 0.15])

    # Insurance and protection
    data['insurance_status'] = np.random.choice(['Ada', 'Lapsed', 'Tidak ada'], n_samples, p=[0.6, 0.15, 0.25])
    data['life_insurance_coverage'] = np.where(data['insurance_status'] == 'Ada', 
                                            data['monthly_income'] * np.random.uniform(10, 30, n_samples), 0)
    data['insurance_company'] = np.random.choice(['A', 'B', 'C', 'N/A'], n_samples, p=[0.3, 0.2, 0.2, 0.3])
    data['insurance_premium_monthly'] = np.where(data['insurance_status'] == 'Ada', 
                                               data['monthly_income'] * np.random.uniform(0.01, 0.05, n_samples), 0)

    # Loan info
    data['loan_id'] = [f'LOAN{i:05d}' for i in range(n_samples)]
    data['product_type'] = np.random.choice(['Kredit Pensiun', 'Payroll loan', 'KTA', 'Vehicle Loan'], n_samples)
    data['issue_date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, n_samples), unit='D')
    data['loan_amount'] = np.random.uniform(5000000, 500000000, n_samples)
    data['term_months'] = np.random.choice([12, 24, 36, 60], n_samples, p=[0.1, 0.2, 0.4, 0.3])
    data['interest_rate'] = np.clip(0.08 + data['loan_amount'] / 1e9 * 0.05 + np.random.normal(0, 0.02, n_samples), 0.05, 0.2)
    data['collateral_flag'] = np.where(data['product_type'] == 'Vehicle Loan', 1, 0)
    data['ltv'] = np.where(data['collateral_flag'] == 1, np.random.uniform(0.5, 0.9, n_samples), 0)
    
    # Outsanding balance, though it has to be < than loan ammount
    data['outstanding_balance'] = data['loan_amount'] * np.random.uniform(0.3, 1.0, n_samples)

    # Transactional behaviour
    data['recent_cash_withdrawals'] = data['monthly_income'] * np.random.uniform(0.1, 0.5, n_samples)
    data['digital_payment_frequency'] = np.random.poisson(5, n_samples)
    data['card_spending'] = data['monthly_income'] * np.random.uniform(0.2, 0.8, n_samples)
    
    # Risk model outputs basically
    
    # Base PD function: PD is influenced negatively by Bureau Score, Income, Insurance, and positively by DTI, Health Risk, Past Due
    
    # Introduce explicit feature interaction pattern for the ML model to learn:
    # Elderly applicants (age >= 60) taking out long-lasting loans (term_months >= 60) should have extremely high risk.
    elderly_long_term_penalty = np.where((data['age'] >= 60) & (data['term_months'] >= 60), 10.0, 0.0)

    logits = (
        -2.0
        - 0.01 * (data['bureau_score'] - 650)
        + 3.0 * data['DTI']
        + 1.5 * data['health_risk_score']
        + 1.0 * data['past_due_months']
        - 1.0 * (data['insurance_status'] == 'Ada')
        + 1.5 * (data['marital_status'] == 'Cerai')
        + 1.0 * data['employer_risk_score']
        + elderly_long_term_penalty
    )
    base_pd = 1 / (1 + np.exp(-logits))
    
    # noise and clip
    data['PD'] = np.clip(base_pd + np.random.normal(0, 0.015, n_samples), 0.005, 0.999)

    # Binary Target Variable (missed_payment_flag) - The label for classification models
    data['missed_payment_flag'] = np.random.binomial(1, data['PD'])
    
    # Loss Given Default - Influenced by insurance and collateral
    data['LGD'] = np.clip(0.4 + 0.1 * (data['collateral_flag'] == 0) - 0.15 * (data['insurance_status'] == 'Ada') + np.random.normal(0, 0.05, n_samples), 0.1, 0.8)
    
    # Exposure at Default - Assume a fraction of outstanding balance
    data['EAD'] = data['outstanding_balance'] * np.random.uniform(0.9, 1.0, n_samples)
    
    # Expected Loss 
    data['expected_loss'] = data['PD'] * data['LGD'] * data['EAD']
    
    # Risk Grade based on PD (for demonstration)
    data['risk_grade'] = pd.cut(data['PD'], 
                                bins=[0, 0.03, 0.07, 0.15, 1], 
                                labels=['A', 'B', 'C', 'D'], 
                                right=False)

    # --- 11. Macro-Economic Indicators (Constant for the initial batch) ---
    data['macro_inflation'] = np.random.normal(3.5, 0.5, n_samples) / 100 
    data['macro_unemployment'] = np.random.normal(5.0, 0.5, n_samples) / 100 
    data['macro_interest_rate'] = np.random.normal(6.25, 0.25, n_samples) / 100 
    data['macro_property_index'] = np.random.normal(120, 5, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Final cleanup and type casting
    df['monthly_income'] = df['monthly_income'].round(0).astype(int)
    df['avg_account_balance'] = df['avg_account_balance'].round(0).astype(int)
    df['loan_amount'] = df['loan_amount'].round(0).astype(int)
    df['outstanding_balance'] = df['outstanding_balance'].round(0).astype(int)
    df['EAD'] = df['EAD'].round(0).astype(int)
    df['life_insurance_coverage'] = df['life_insurance_coverage'].round(0).astype(int)

    return df

# Generate the data
df_credit = generate_credit_risk_data(n_samples=10000)

# Display the first few rows and check data types
print("--- Synthetic Credit Risk Dataset Head ---")
print(df_credit.head())
print("\n--- DataFrame Info ---")
df_credit.info()

# Save the dataset to a CSV file (your 'Data Ingestion' source)
df_credit.to_csv('synthetic_credit_risk_data.csv', index=False)
print("\nDataset saved to 'synthetic_credit_risk_data.csv'")