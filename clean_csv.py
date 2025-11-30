import pandas as pd
from pathlib import Path
import re
import os


def clean_central_bank_file():
    # --- CORRECTION DU CHEMIN ICI ---
    # 1. On récupère le dossier où se trouve ce script (dossier 'src')
    script_dir = Path(__file__).resolve().parent

    # 2. On remonte d'un cran pour aller à la racine du projet ('gold-price-forecasting-ml')
    project_root = script_dir.parent

    # 3. On construit le chemin vers data/raw
    file_path = project_root / "data" / "raw" / "central_bank_demand.csv"
    # -------------------------------

    print(f"📂 Lecture du fichier : {file_path}")

    if not file_path.exists():
        print(f"❌ Erreur : Fichier introuvable ici : {file_path}")
        print(
            "Vérifie que 'central_bank_demand.csv' est bien dans le dossier 'data/raw' à la racine."
        )
        return

    # Le reste du code reste identique...
    # ...
    try:
        df = pd.read_csv(file_path, sep=None, engine="python", dtype=str)
    except Exception as e:
        print(f"❌ Erreur de lecture : {e}")
        print("💡 CONSEIL : Vérifie que le fichier n'est pas ouvert dans Excel !")
        return

    # (Garde la suite de ton code précédent à partir d'ici : df.columns = ...)
    # Copie-colle la suite du code précédent ici
    df.columns = ["Date", "Demand"]
    print("🛠️  Nettoyage en cours...")

    def parse_quarterly_date(date_str):
        if pd.isna(date_str):
            return None
        date_str = str(date_str).strip()
        match = re.search(r"Q([1-4])['\s]*(\d{2,4})", date_str, re.IGNORECASE)
        if match:
            quarter = int(match.group(1))
            year_str = match.group(2)
            year = int(year_str)
            if year < 100:
                year += 2000
            month_end = {1: 3, 2: 6, 3: 9, 4: 12}
            day_end = {1: 31, 2: 30, 3: 30, 4: 31}
            return pd.Timestamp(
                year=year, month=month_end[quarter], day=day_end[quarter]
            )
        return None

    df["Date"] = df["Date"].apply(parse_quarterly_date)

    def clean_number(num_str):
        if pd.isna(num_str):
            return 0.0
        clean_str = str(num_str).replace(",", ".").replace(" ", "")
        try:
            return float(clean_str)
        except:
            return 0.0

    df["Demand"] = df["Demand"].apply(clean_number)
    df = df.dropna(subset=["Date"]).sort_values("Date")

    df.to_csv(file_path, index=False)

    print("✅ Succès ! Fichier nettoyé.")
    print(df.head())


if __name__ == "__main__":
    clean_central_bank_file()
