Sure — here’s a clean, copy-paste-ready version of your **README.md**:

```markdown
# Shah Yash Midterm Project

```

## 🧠 Project Overview
This project performs **association rule mining** on multiple retail transaction datasets using three different algorithms:

- **Brute Force**
- **Apriori Algorithm**
- **FP-Growth Algorithm**

The goal is to find **frequent itemsets** and generate **association rules** from transaction data to identify meaningful product relationships.

---

## How to Run the Code

### 1. Setup Environment (VS Code / Command Line)

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

If `requirements.txt` is missing, manually install the main libraries:

```bash
pip install pandas mlxtend
```

---

### 2. Run Standalone Python Scripts

Each script runs one algorithm independently:

```bash
python .\src\brute_force.py
python .\src\apriori_algo.py
python .\src\fpgrowth_algo.py
```

To run **all algorithms together**, execute:

```bash
python association_rules.py
```

---

### 3. Run Jupyter Notebook

To view the code and outputs interactively:

```bash
jupyter notebook .\Yash_Shah_Midterm.ipynb
```

---

## 📦 Dependencies

* Python 3.x
* pandas
* mlxtend
* jupyter

---

## 📊 Outputs

* Algorithm output screenshots: `screenshots/`
* Final written report: `report/midterm_report.pdf`
* Dataset files: `data2/`

---

## 👨‍💻 Author

**Yash Shah**
MS Computer Science — NJIT

