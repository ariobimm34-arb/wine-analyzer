import urllib.request
import os

print("Mengunduh dataset Wine Quality...")

urls = {
    "winequality-red.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "winequality-white.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}

for filename, url in urls.items():
    if os.path.exists(filename):
        print(f"  {filename} sudah ada, skip.")
    else:
        urllib.request.urlretrieve(url, filename)
        print(f"  {filename} berhasil diunduh.")

print("\nSelesai! Dataset siap digunakan.")
print("Lanjut ke: python step3_train_model.py")