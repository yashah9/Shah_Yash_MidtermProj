import os
import sys
import time
import itertools
import pandas as pd
import warnings
from collections import defaultdict
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------
# CONFIGURATION OF Path
# ------------------------
#DATASET_FOLDER = os.path.join(os.getcwd(), "C:/CS634-Midterm/data2")  # Update this path if needed
DATASET_FOLDER = os.path.join(os.getcwd(), "data2")  # Update this path if needed

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

print("\nDatasets available:")
for i, name in enumerate(DATASET_FILES.keys(), start=1):
    print(f"  {i}. {name} -> {DATASET_FILES[name]}")


# ------------------------
# HELPER FUNCTIONS
# ------------------------
def load_transactions_from_csv(path, tx_col='transaction_list', id_col='transaction_id'):
    """Load CSV where each row has a comma-separated list of items."""
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


def brute_force_frequent_itemsets(transactions, min_support, verbose=False):
    """Brute-force enumeration of frequent itemsets."""
    n_trans = len(transactions)
    min_count = max(1, int(min_support * n_trans + 1e-9))
    items = sorted({it for tx in transactions for it in tx})
    tx_sets = [set(tx) for tx in transactions]

    support_counts_all = {}
    freq_itemsets = {}
    k = 1
    found_any = True

    while found_any:
        found_any = False
        candidates = [(it,) for it in items] if k == 1 else list(itertools.combinations(items, k))
        counts = {}

        for cand in candidates:
            cand_set = set(cand)
            c = sum(1 for tx in tx_sets if cand_set.issubset(tx))
            if c >= min_count:
                counts[tuple(cand)] = c
                support_counts_all[frozenset(cand)] = c

        if counts:
            freq_itemsets[k] = counts
            found_any = True
            if verbose:
                print(f"Found {len(counts)} frequent {k}-itemsets")
            k += 1
        else:
            break

    return freq_itemsets, support_counts_all, n_trans


def generate_rules_from_freq_itemsets(freq_itemsets, support_counts_all, n_trans, min_confidence):
    """Generate association rules from frequent itemsets."""
    rules = []
    for k, items_dict in freq_itemsets.items():
        if k < 2:
            continue
        for itemset_tuple, count in items_dict.items():
            itemset = set(itemset_tuple)
            sup_itemset = count / n_trans
            for s in range(1, k):
                for antecedent in itertools.combinations(itemset_tuple, s):
                    antecedent = set(antecedent)
                    consequent = itemset - antecedent
                    ante_count = support_counts_all.get(frozenset(antecedent), 0)
                    if ante_count == 0:
                        continue
                    confidence = count / ante_count
                    if confidence >= min_confidence:
                        rules.append({
                            "antecedent": tuple(sorted(antecedent)),
                            "consequent": tuple(sorted(consequent)),
                            "support": sup_itemset,
                            "support_count": count,
                            "confidence": confidence
                        })
    return sorted(rules, key=lambda r: (-r['confidence'], -r['support']))


def transactions_to_onehot_df(transactions):
    """Convert list of transactions to one-hot encoded DataFrame."""
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_).astype(int)


def normalize_support_conf(val_str):
    """Convert '20' -> 0.2 or '0.2' -> 0.2."""
    v = float(val_str)
    if v <= 1:
        return v
    if v <= 100:
        return v / 100
    raise ValueError("Support/confidence must be 0–1 or 1–100%")


# ------------------------
# MAIN EXECUTION LOGIC
# ------------------------
def run_for_dataset(dataset_name, tx_file, min_support, min_confidence, verbose=True):
    print(f"\n=== Running on dataset: {dataset_name} ===")
    transactions = load_transactions_from_csv(tx_file)
    print(f"Loaded {len(transactions)} transactions.")

    # --- Brute-force ---
    t0 = time.time()
    freq_itemsets, support_counts_all, n_trans = brute_force_frequent_itemsets(
        transactions, min_support, verbose=False
    )
    brute_time = time.time() - t0
    total_freq_count = sum(len(v) for v in freq_itemsets.values())

    rules = generate_rules_from_freq_itemsets(
        freq_itemsets, support_counts_all, n_trans, min_confidence
    )

    print(f"\n[✓] Brute-force completed in {brute_time:.2f}s")
    print(f"Found {total_freq_count} frequent itemsets and {len(rules)} association rules.\n")

    # --- Apriori ---
    df_onehot = transactions_to_onehot_df(transactions)
    t1 = time.time()
    apr = apriori(df_onehot, min_support=min_support, use_colnames=True)
    apr_rules = association_rules(apr, metric="confidence", min_threshold=min_confidence)
    apr_time = time.time() - t1

    # --- FP-Growth ---
    t2 = time.time()
    fpg = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
    fpg_rules = association_rules(fpg, metric="confidence", min_threshold=min_confidence)
    fpg_time = time.time() - t2

    # --- Output Section ---
    def print_frequent_itemsets(title, itemsets_df):
        print(f"\n=== {title}: Frequent Itemsets ===")
        for _, row in itemsets_df.iterrows():
            items = ", ".join(row["itemsets"])
            print(f"{{{items}}} (support={row['support']:.2f})")

    def print_rules(title, rules_df):
        print(f"\n=== {title}: Association Rules ===")
        for _, row in rules_df.iterrows():
            ant = ", ".join(row["antecedents"])
            cons = ", ".join(row["consequents"])
            print(f"{ant} → {cons} (confidence={row['confidence']:.2f})")

    # --- Print Brute-force Results ---
    if verbose:
        print("\n=== Brute-force Results ===")
        print("Frequent Itemsets:")
        for k, itemsets in freq_itemsets.items():
            for itemset, support_count in itemsets.items():
                support = support_count / n_trans
                print(f"{{{', '.join(itemset)}}} (support={support:.2f})")

        print("\nAssociation Rules:")
        for r in rules[:10]:
            print(f"{', '.join(r['antecedent'])} → {', '.join(r['consequent'])} "
                  f"(confidence={r['confidence']:.2f})")

    # --- Print Apriori & FP-Growth ---
    if not apr.empty:
        print_frequent_itemsets("Apriori", apr)
        print_rules("Apriori", apr_rules)

    if not fpg.empty:
        print_frequent_itemsets("FP-Growth", fpg)
        print_rules("FP-Growth", fpg_rules)

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Brute-force: {total_freq_count} itemsets, {len(rules)} rules, {brute_time:.2f}s")
    print(f"Apriori: {len(apr)} itemsets, {len(apr_rules)} rules, {apr_time:.2f}s")
    print(f"FP-Growth: {len(fpg)} itemsets, {len(fpg_rules)} rules, {fpg_time:.2f}s")

    return {
        "brute_force_rules": rules,
        "apriori_rules": apr_rules,
        "fpgrowth_rules": fpg_rules,
    }




def interactive_run():
    print("\nAvailable datasets:")
    names = list(DATASET_FILES.keys())
    for i, nm in enumerate(names, start=1):
        print(f"{i}. {nm}")

    while True:
        sel = input(f"Choose dataset by number (1-{len(names)}): ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(names):
            chosen = names[int(sel) - 1]
            break
        print("Invalid choice, try again.")

    ms = normalize_support_conf(input("Enter min support (e.g., 0.2 or 20): "))
    mc = normalize_support_conf(input("Enter min confidence (e.g., 0.6 or 60): "))

    results = run_for_dataset(chosen, DATASET_FILES[chosen], ms, mc, verbose=True)
    return results


if __name__ == "__main__":
    interactive_run()
