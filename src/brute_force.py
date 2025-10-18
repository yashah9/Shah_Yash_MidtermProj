import time
import itertools
from utils import load_transactions_from_csv, DATASET_FILES, normalize_support_conf


def brute_force_frequent_itemsets(transactions, min_support):
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
            k += 1
        else:
            break

    return freq_itemsets, support_counts_all, n_trans


def generate_rules(freq_itemsets, support_counts_all, n_trans, min_confidence):
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
                            "confidence": confidence
                        })
    return sorted(rules, key=lambda r: (-r['confidence'], -r['support']))


def run_brute_force(dataset_name, min_support, min_confidence):
    print(f"\n=== Running on dataset: {dataset_name} ===")
    transactions = load_transactions_from_csv(DATASET_FILES[dataset_name])
    print(f"Loaded {len(transactions)} transactions.")

    # --- Brute-force algorithm ---
    t0 = time.time()
    freq_itemsets, support_counts_all, n_trans = brute_force_frequent_itemsets(transactions, min_support)
    rules = generate_rules(freq_itemsets, support_counts_all, n_trans, min_confidence)
    elapsed = time.time() - t0

    print(f"\n[✓] Brute-force completed in {elapsed:.2f}s")
    print(f"Found {sum(len(v) for v in freq_itemsets.values())} frequent itemsets and {len(rules)} association rules.")

    # --- Detailed Output ---
    print("\n=== Brute-force Results ===")
    print("Frequent Itemsets:")
    for k, itemsets in freq_itemsets.items():
        for items, count in itemsets.items():
            print(f"{set(items)} (support={count / n_trans:.2f})")

    print("\nAssociation Rules:")
    for rule in rules:
        ant, cons = rule['antecedent'], rule['consequent']
        print(f"{set(ant)} → {set(cons)} (confidence={rule['confidence']:.2f})")

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

    run_brute_force(chosen, ms, mc)


if __name__ == "__main__":
    interactive_run()
