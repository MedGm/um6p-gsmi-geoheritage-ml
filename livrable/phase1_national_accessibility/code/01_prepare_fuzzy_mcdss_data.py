import os
import re
import numpy as np
import pandas as pd

P1_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "geosites_accessibility_labeled.csv"))
OUT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "geosites_fuzzy_mcdss_master.csv"))

def clean_region_names(region_str):
    if pd.isna(region_str):
        return "Unknown"
    s = str(region_str).strip()
    if re.search(r"B[eé]ni\s+Mellal[- ]Kh[eé]nifra", s, re.IGNORECASE):
        return "Béni Mellal-Khénifra"
    elif re.search(r"Tanger[- ]T[eé]touan[- ]Al\s+Hoce[ïi]ma", s, re.IGNORECASE):
        return "Tanger-Tétouan-Al Hoceïma"
    elif re.search(r"Dr[aâ]a[- ]Tafilalet", s, re.IGNORECASE):
        return "Draâ-Tafilalet"
    elif re.search(r"Souss[- ]Massa", s, re.IGNORECASE):
        return "Souss-Massa"
    elif re.search(r"Marrakech[- ]Safi", s, re.IGNORECASE):
        return "Marrakech-Safi"
    elif re.search(r"F[eèé]s[- ]M[eèé]kn[eèé]s", s, re.IGNORECASE):
        return "Fès-Meknès"
    elif re.search(r"Rabat[- ]Sal[eé][- ]K[eé]nitra", s, re.IGNORECASE):
        return "Rabat-Salé-Kénitra"
    elif re.search(r"Oriental", s, re.IGNORECASE):
        return "l'Oriental"
    elif re.search(r"Dakhla", s, re.IGNORECASE):
        return "Dakhla-Oued Eddahab"
    elif re.search(r"La[aâ]youne", s, re.IGNORECASE):
        return "Laâyoune-Sakia El Hamra"
    elif re.search(r"Guelmim", s, re.IGNORECASE):
        return "Guelmim-Oued Noun"
    elif re.search(r"Casablanca[- ]Settat", s, re.IGNORECASE):
        return "Casablanca-Settat"
    return s

def compute_fuzzy_accessibility_score(d_hw, d_piste, slope, ruggedness):
    """
    Computes continuous Fuzzy Multi-Criteria Membership Score S_access in [0, 1].
    Smooth sigmoid decay curves prevent artificial sharp step-function border artifacts.
    """
    # Sigmoids for road & track proximity (higher = closer = better access)
    mu_hw = 1.0 / (1.0 + np.exp(np.clip((d_hw - 2000.0) / 400.0, -50, 50)))
    mu_piste = 1.0 / (1.0 + np.exp(np.clip((d_piste - 1000.0) / 250.0, -50, 50)))
    
    # Combined infrastructure proximity
    mu_infra = np.maximum(mu_hw, 0.75 * mu_piste)
    
    # Terrain penalty sigmoids (higher slope/ruggedness = worse access)
    mu_slope_penalty = 1.0 / (1.0 + np.exp(np.clip((slope - 20.0) / 4.0, -50, 50)))
    mu_rugged_penalty = 1.0 / (1.0 + np.exp(np.clip((ruggedness - 200.0) / 40.0, -50, 50)))
    
    # Overall fuzzy accessibility score S_access in [0.0, 1.0]
    s_access = mu_infra * (0.65 * mu_slope_penalty + 0.35 * mu_rugged_penalty)
    return s_access

def main():
    print(f"Loading master labeled dataset from: {P1_CSV}")
    df = pd.read_csv(P1_CSV)
    
    # Drop coordinate duplicates
    df = df.drop_duplicates(subset=['Latitude_WGS84', 'Longitude_WGS84']).copy()
    print(f"Deduplicated master geosites count: {len(df)}")
    
    # Clean region names
    df['Administrative_Region'] = df['Administrative_Region'].apply(clean_region_names)
    
    # Fill missing terrain features with median
    num_cols = ['Elevation_m', 'Slope_deg', 'Ruggedness', 'Dist_to_Highway_m', 'Dist_to_Piste_m', 'Dist_to_Dam_m', 'Dist_to_River_m']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Compute Continuous Fuzzy Accessibility Index
    s_scores = compute_fuzzy_accessibility_score(
        df['Dist_to_Highway_m'].values,
        df['Dist_to_Piste_m'].values,
        df['Slope_deg'].values,
        df['Ruggedness'].values
    )
    
    df['Fuzzy_Accessibility_Score'] = s_scores
    
    # Categorize using continuous smooth thresholds
    categories = []
    for s in s_scores:
        if s >= 0.50:
            categories.append('Easy')
        elif s >= 0.25:
            categories.append('Moderate')
        else:
            categories.append('Difficult')
            
    df['Fuzzy_Accessibility_Category'] = categories
    
    print("\n--- Unified Regional Geosites Counts ---")
    print(df['Administrative_Region'].value_counts())
    
    print("\n--- Continuous Fuzzy Accessibility Categories ---")
    print(df['Fuzzy_Accessibility_Category'].value_counts())
    
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved master fuzzy MCDSS dataset to: {OUT_CSV}")

if __name__ == "__main__":
    main()
