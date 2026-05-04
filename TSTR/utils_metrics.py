import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

def calculate_inverted_ks_statistic(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, columns: list = None) -> float:
    """
    Calculates the Inverted Kolmogorov-Smirnov (KS) Statistic across continuous columns.
    The regular KS statistic measures the maximum distance between the CDFs of two distributions.
    Inverted KS = 1 - KS. 
    Higher Inverted KS means higher fidelity (distributions are more similar).
    
    Returns the average Inverted KS across all specified columns.
    """
    if columns is None:
        # Use numerical columns by default
        columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
    ks_results = []
    
    for col in columns:
        if col in synthetic_data.columns:
            # KS statistic range is [0, 1]
            stat, p_value = ks_2samp(real_data[col], synthetic_data[col])
            inverted_ks = 1.0 - stat
            ks_results.append(inverted_ks)
            
    if not ks_results:
        warnings.warn("No overlapping columns found for KS statistic calculation.")
        return 0.0
        
    mean_inverted_ks = np.mean(ks_results)
    return mean_inverted_ks

def calculate_dcr(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, columns: list = None) -> dict:
    """
    Calculates the Distance to Closest Record (DCR).
    DCR measures the Euclidean distance from each synthetic record to its nearest real record.
    A DCR of 0 indicates an exact copy (memorization/data leakage).
    
    Returns a dictionary summarizing the DCR distribution:
      - mean_dcr
      - min_dcr
      - perfectly_copied_fraction (percentage of synthetic records with DCR = 0)
    """
    if columns is None:
        columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
    # Take only the relevant columns and drop NaNs
    real_subset = real_data[columns].dropna()
    syn_subset = synthetic_data[columns].dropna()
    
    if len(real_subset) == 0 or len(syn_subset) == 0:
        return {"mean_dcr": float('nan'), "min_dcr": float('nan'), "perfectly_copied_fraction": float('nan')}
    
    # Scale data to ensure distances are comparable across features
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_subset)
    syn_scaled = scaler.transform(syn_subset)
    
    # Use NearestNeighbors to find the closest real record for each synthetic record
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
    nn.fit(real_scaled)
    
    distances, indices = nn.kneighbors(syn_scaled)
    
    # Analyze the distances
    mean_dcr = np.mean(distances)
    min_dcr = np.min(distances)
    
    # Consider distances very close to 0 as copies (due to float precision)
    copies = np.sum(distances < 1e-6)
    copied_fraction = copies / len(syn_subset)
    
    return {
        "mean_dcr": mean_dcr,
        "min_dcr": min_dcr,
        "perfectly_copied_fraction": copied_fraction
    }
