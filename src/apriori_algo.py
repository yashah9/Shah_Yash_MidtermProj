import time
from mlxtend.frequent_patterns import apriori, association_rules
from utils import load_transactions_from_csv, transactions_to_onehot_df, DATASET_FILES, normalize_support_conf


def run_apriori(dataset_name, min_support, min_confidence):
    print(f"\n=== Running Apriori on {dataset_name} ===")
    transactions = load_transactions_from_csv(DATASET_FILES[dataset_name])
    df = transactions_to_onehot_df(transactions)

    t0 = time.time()
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    elapsed = time.time() - t0

    print(f"\n[✓] Apriori completed in {elapsed:.2f}s")
    print(f"Found {len(freq_itemsets)} frequent itemsets and {len(rules)} association rules.")

    # --- Detailed Output ---
    print("\n=== Apriori Results ===")
    print("Frequent Itemsets:")
    for _, row in freq_itemsets.iterrows():
        print(f"{set(row['itemsets'])} (support={row['support']:.2f})")

    print("\nAssociation Rules:")
    for _, row in rules.iterrows():
        print(f"{set(row['antecedents'])} → {set(row['consequents'])} (confidence={row['confidence']:.2f})")

    return freq_itemsets, rules


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

    run_apriori(chosen, ms, mc)


if __name__ == "__main__":
    interactive_run()
