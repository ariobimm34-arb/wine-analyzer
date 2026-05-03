# ============================================================
# STEP 2 — Eksplorasi Data (EDA)
# Jalankan: python step2_explore_data.py
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load data ---
df_red   = pd.read_csv("winequality-red.csv",   sep=";")
df_white = pd.read_csv("winequality-white.csv",  sep=";")

df_red["wine_type"]   = "red"
df_white["wine_type"] = "white"
df = pd.concat([df_red, df_white], ignore_index=True)

print("=" * 50)
print("INFO DATASET")
print("=" * 50)
print(f"Total baris  : {len(df)}")
print(f"Red wine     : {len(df_red)}")
print(f"White wine   : {len(df_white)}")
print(f"\nKolom: {list(df.columns)}")

print("\n" + "=" * 50)
print("STATISTIK DESKRIPTIF")
print("=" * 50)
print(df.describe().round(2))

print("\n" + "=" * 50)
print("DISTRIBUSI KUALITAS WINE")
print("=" * 50)
print(df["quality"].value_counts().sort_index())

# Cek missing values
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# --- Visualisasi ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Eksplorasi Data Wine Quality", fontsize=14, fontweight="bold")

# 1. Distribusi kualitas
df["quality"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color="#7F0000", edgecolor="white"
)
axes[0].set_title("Distribusi Nilai Kualitas")
axes[0].set_xlabel("Skor Kualitas")
axes[0].set_ylabel("Jumlah Wine")

# 2. Kualitas per jenis wine
df.groupby(["wine_type", "quality"]).size().unstack().plot(
    kind="bar", ax=axes[1], color=["#C8A84B", "#7F0000"], edgecolor="white"
)
axes[1].set_title("Kualitas per Jenis Wine")
axes[1].set_xlabel("Jenis Wine")
axes[1].legend(title="Kualitas")

# 3. Korelasi dengan kualitas (red wine saja)
corr = df_red.drop("wine_type", axis=1, errors="ignore").corr()["quality"].drop("quality").sort_values()
colors = ["#B71C1C" if x < 0 else "#2E7D32" for x in corr]
corr.plot(kind="barh", ax=axes[2], color=colors, edgecolor="white")
axes[2].set_title("Korelasi Fitur vs Kualitas (Red)")
axes[2].axvline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("eda_result.png", dpi=120, bbox_inches="tight")
plt.show()
print("\nGrafik disimpan: eda_result.png")
print("Lanjut ke: python step3_train_model.py")
