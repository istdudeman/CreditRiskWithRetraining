import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap

# ==============================================================================
# SCRIPT TEMPLATE: GENERATE GRAFIK EVALUASI UNTUK BAB IV
# Pastikan untuk mengganti 'y_true', 'y_pred_prob', dan 'X_test' dengan variabel 
# aktual dari sistem Anda sebelum menjalankan script ini.
# ==============================================================================

def plot_combined_roc_curve(y_true, y_prob_raw, y_prob_woe, save_path="roc_curve_comparison.png"):
    """
    Menghasilkan grafik komparasi ROC Curve (Skenario B: Ablation Study)
    """
    plt.figure(figsize=(8, 6))
    
    # Menghitung curve untuk XGBoost tanpa WoE
    fpr_raw, tpr_raw, _ = roc_curve(y_true, y_prob_raw)
    auc_raw = auc(fpr_raw, tpr_raw)
    plt.plot(fpr_raw, tpr_raw, color='red', lw=2, label=f'XGBoost (Raw Data) - AUC = {auc_raw:.3f}')
    
    # Menghitung curve untuk XGBoost dengan WoE
    fpr_woe, tpr_woe, _ = roc_curve(y_true, y_prob_woe)
    auc_woe = auc(fpr_woe, tpr_woe)
    plt.plot(fpr_woe, tpr_woe, color='blue', lw=2, label=f'XGBoost (WoE Data) - AUC = {auc_woe:.3f}')
    
    # Garis diagonal (random guess)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Perbandingan ROC Curve: Dampak Transformasi WoE', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Kurva ROC disimpan di: {save_path}")
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, save_path="confusion_matrix.png"):
    """
    Menghasilkan Heatmap Confusion Matrix (Bukti Evaluasi Model)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good Loan (0)', 'Bad Loan (1)'],
                yticklabels=['Good Loan (0)', 'Bad Loan (1)'],
                annot_kws={"size": 14})
    
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix - XGBoost (WoE)', fontsize=14)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Confusion Matrix disimpan di: {save_path}")
    plt.show()

def generate_shap_summary(model, X_test, save_path="shap_summary.png"):
    """
    Menghasilkan SHAP Summary Plot (Bukti Interpretabilitas)
    X_test di sini adalah dataframe pandas dengan nama fitur.
    """
    print("Menghitung nilai SHAP (ini mungkin memakan waktu)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 8))
    # Buat summary plot (tidak ditampilkan langsung agar bisa disave)
    shap.summary_plot(shap_values, X_test, show=False)
    
    plt.title('SHAP Summary Plot: Pengaruh Fitur terhadap Risiko Kredit', fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] SHAP Summary disimpan di: {save_path}")
    plt.show()

if __name__ == "__main__":
    import os
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np

    # Membuat folder untuk menyimpan gambar
    out_dir = "bab 3 gambar"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Mulai men-generate gambar laporan dan menyimpannya di folder '{out_dir}'...")
    
    # ---------------------------------------------------------
    # 1. Dummy Data untuk ROC dan Confusion Matrix
    # ---------------------------------------------------------
    np.random.seed(42)
    y_true_dummy = np.random.randint(0, 2, 1000)
    y_prob_raw_dummy = np.random.rand(1000)
    # Simulasi data WoE memiliki performa sedikit lebih baik (AUC lebih tinggi)
    y_prob_woe_dummy = np.random.rand(1000) + (y_true_dummy * 0.3)
    y_prob_woe_dummy = np.clip(y_prob_woe_dummy, 0, 1)
    y_pred_woe_dummy = (y_prob_woe_dummy > 0.5).astype(int)
    
    plot_combined_roc_curve(
        y_true_dummy, 
        y_prob_raw_dummy, 
        y_prob_woe_dummy, 
        save_path=os.path.join(out_dir, "roc_curve_comparison.png")
    )
    
    plot_confusion_matrix_heatmap(
        y_true_dummy, 
        y_pred_woe_dummy, 
        save_path=os.path.join(out_dir, "confusion_matrix.png")
    )

    # ---------------------------------------------------------
    # 2. Dummy Model & Data untuk SHAP Summary Plot
    # ---------------------------------------------------------
    print("Mempersiapkan data dan model dummy untuk plot SHAP...")
    X_dummy = pd.DataFrame({
        'Umur': np.random.randint(20, 65, 200),
        'Pendapatan': np.random.randint(3000, 15000, 200),
        'Riwayat_Tunggakan': np.random.randint(0, 5, 200),
        'Jumlah_Pinjaman': np.random.randint(1000, 50000, 200)
    })
    # Target klasifikasi dummy
    y_dummy_model = np.random.randint(0, 2, 200)
    
    # Latih model ringan (Random Forest) untuk generate tree SHAP
    dummy_model = RandomForestClassifier(max_depth=3, n_estimators=10, random_state=42)
    dummy_model.fit(X_dummy, y_dummy_model)
    
    generate_shap_summary(
        dummy_model, 
        X_dummy, 
        save_path=os.path.join(out_dir, "shap_summary.png")
    )
    
    print(f"\n[OK] Selesai! Semua gambar berhasil disimpan di folder: {out_dir}/")
