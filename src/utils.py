import os
import sys
import pandas as pd
import itertools
from mlxtend.preprocessing import TransactionEncoder

# --- PATH CONFIGURATION ---
DATASET_FOLDER = os.path.join(os.getcwd(), "data2")

if not os.path.isdir(DATASET_FOLDER):
    print(f"[!] DATASET_FOLDER path does not exist: {DATASET_FOLDER}")
    sys.exit(1)
else:
    print("[✓] DATASET_FOLDER OK:", DATASET_FOLDER)

DATASET_FILES = {
    "Amazon": os.path.join(DATASET_FOLDER, "amazon_transactions.csv"),
    "Shoprite": os.path.join(DATASET_FOLDER, "shoprite_transactions.csv"),
    "BestBuy": os.path.join(DATASET_FOLDER, "bestbuy_transactions.csv"),
    "Adidas": os.path.join(DATASET_FOLDER, "adidas_transactions.csv"),
    "Dicks": os.path.join(DATASET_FOLDER, "dicks_sporting_transactions.csv")
}

def load_transactions_from_csv(path, tx_col='transaction_list'):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, dtype=str).fillna('')
    if tx_col not in df.columns:
        raise ValueError(f"Expected column '{tx_col}' in {path}. Found: {df.columns.tolist()}")
    transactions = []
    for val in df[tx_col].astype(str):
        items = [it.strip() for it in val.split(',') if it.strip()]
        transactions.append(items)
    return transactions

def transactions_to_onehot_df(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_).astype(int)

def normalize_support_conf(val_str):
    v = float(val_str)
    if v <= 1:
        return v
    if v <= 100:
        return v / 100
    raise ValueError("Support/confidence must be 0–1 or 1–100%")
