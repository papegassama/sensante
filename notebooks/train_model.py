import pandas as pd
import numpy as np
# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv.csv")
# Verifier les dimensions
print( f"Dataset : {df.shape[0]} patients , {df.shape[1]} colonnes ")
print ( f"\nColonnes : {list(df.columns)}")
print ( f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# Preparer les donnees
from sklearn.preprocessing import LabelEncoder

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])      
df['region_encoded'] = le_region.fit_transform(df['region'])

# Selectionner les features et la cible
features = ['age', 'sexe_encoded', 'température', 'tension_sys', 'toux', 'fatigue', 'maux_tete' ,'region_encoded']

X = df[features]
y = df['diagnostic']

print ( f"Features : {X.shape}")
print ( f"Cible : {y.shape}")