""" 
SenSante - Explorer le fichier patient_dakar.csv
Lab 1 : Git, Python et Structure Projet
"""

import pandas as pd
import matpotlib as plt

# === CHARGER LES DONNEES ===

df = pd.read_csv("./data/patients_dakar.csv.csv") # ouvre le fichier et le charge dans pandas

# === PREMIERS APER US ===

print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

# Dimension du dataset

print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")


# Apercu des 5 premiers lignes
print(f"\n----- Statistiques descriptives ------")
print(df.head())

# ==== STATISTIQUE DES DIAGNOSTIQUES ====

print(df.describe().round(2))

# ==== REPARTITION DES DIAGNOSTIQUES ===

print(f"\n ---- Repartition des diagnostiques ----")

diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items() : 
	pct = count / len(df) * 100
	print(f"  {diag:12s} {count:3d} patients ({pct:.1f}%)")


# === REPARTITION PAR REGION ===

print(f"\n---Repartition par region (top 5) ---")
region_counts = df["region"].value_counts().head(5)
for region , count in region_counts.items():
	print(f"  {region:15s} : {count:3d} patients")

# === TEMPERATURE MOYENNE PAR DIAGNOSTIC ===

print(f"\n---Temperature moyenne par diagnostique ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items() :
	print(f"  {diag:12s} : {temp:.1f} C")



# == REPARTION PAR SEXE ET DIAGNOSTIQUES ===

print(f"\n ---- Repartition par sexe et diagnostique ----")
sexe_diag =df.groupby(["sexe", "diagnostic"]).size()

for (sexe, diag), count in sexe_diag.items() :
	print(f"{sexe:6s} | {diag:12s} : {count:3d} patients")




print("=" * 50)
print("Exploration terminee !!")
print("Prochain lab : entrainer un modele ML")
print("=" * 50)
