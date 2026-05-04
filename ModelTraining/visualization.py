import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Konfigurasi style visualisasi
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Simulasi Load Data (Ganti dengan file asli Anda)
df = pd.read_csv('synthetic_credit_risk_data.csv')

# Dummy data generator untuk keperluan contoh running script
def generate_dummy_data(n=500):
    np.random.seed(42)
    data = {
        'bureau_score': np.random.normal(650, 100, n).clip(300, 850),
        'monthly_income': np.random.lognormal(16, 0.5, n),
        'loan_amount': np.random.lognormal(18, 0.6, n),
        'risk_grade': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.4, 0.4, 0.2]),
        'missed_payment_flag': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'age': np.random.randint(45, 75, n),
        'PD': np.random.uniform(0, 0.3, n)
    }
    return pd.DataFrame(data)

df = generate_dummy_data()

# --- VISUALISASI 1: Distribusi Skor Kredit (Bureau Score) berdasarkan Risk Grade ---
plt.figure()
sns.histplot(data=df, x='bureau_score', hue='risk_grade', multiple="stack", palette='viridis')
plt.title('Distribusi Bureau Score (SLIK) Berdasarkan Risk Grade', fontsize=15)
plt.xlabel('Credit Score')
plt.ylabel('Frekuensi')
plt.savefig('distribusi_bureau_score.png')

# --- VISUALISASI 2: Korelasi antara Pendapatan dan Jumlah Pinjaman ---
plt.figure()
sns.scatterplot(data=df, x='monthly_income', y='loan_amount', hue='risk_grade', alpha=0.6)
plt.title('Hubungan Monthly Income vs Loan Amount', fontsize=15)
plt.xscale('log') # Menggunakan skala log untuk data finansial
plt.yscale('log')
plt.xlabel('Monthly Income (Log Scale)')
plt.ylabel('Loan Amount (Log Scale)')
plt.savefig('income_vs_loan.png')

# --- VISUALISASI 3: Persentase Target Label (Default vs Non-Default) ---
plt.figure()
df['missed_payment_flag'].value_counts().plot.pie(
    autopct='%1.1f%%', 
    colors=['skyblue', 'salmon'], 
    labels=['Lancar (0)', 'Gagal Bayar (1)'],
    explode=(0, 0.1)
)
plt.title('Proporsi Gagal Bayar (Target Label)', fontsize=15)
plt.ylabel('')
plt.savefig('proporsi_default.png')

# --- VISUALISASI 4: Boxplot Usia Nasabah pada Segmen Pensiunan ---
plt.figure()
sns.boxplot(data=df, x='risk_grade', y='age', palette='Set2')
plt.title('Distribusi Usia Nasabah Per Kategori Risiko', fontsize=15)
plt.xlabel('Risk Grade')
plt.ylabel('Usia')
plt.savefig('boxplot_usia_risiko.png')

print("Visualisasi berhasil dibuat. Silakan cek file gambar di direktori kerja.")