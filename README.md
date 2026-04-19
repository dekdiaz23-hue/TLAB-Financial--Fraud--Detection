[TLAB] Financial Fraud Detection

This project builds a machine learning pipeline to detect fraudulent bank transactions using a dataset of 6.3 million rows. The goal is to catch as much fraud as possible while keeping false positives low.

# The Dataset

Some context that is important for understanding the data set is that:
1) only 0.13% of transactions are fraud meaning the data is heavily imbalanced.
2) Fraud only occurs in TRANSFER and CASH_OUT transactions
3) The dataset includes a naive fraud flag (`isFlaggedFraud`) that catches less than 0.2% of actual fraud transactions


## Hypothesis

Fraudulent transactions are concentrated in TRANSFER and CASH_OUT transactions where the origin account balance is completely drained to zero after the transaction.


## Project Structure

# Notebook 1:  EDA
Exploratory analysis broken into three sections:

1) Univariate: distributions of amount, transaction type, fraud rate, and time
2) Bivariate: each feature compared against isFraud 
3) Multivariate: relationships between predictors with fraud as an overlay

Key findings post EDA:
1) PAYMENT, CASH_IN, and DEBIT have zero fraud cases
2) Fraud transactions drain the origin account to zero the vast majority of the time
3) Fraudulent transactions average 8x higher in amount than legitimate ones
4) The naive isFlaggedFraud rule catches only 16 out of 8,213 fraud cases


# Notebook 2: Data Cleaning & Preprocessing


- Dropped nameOrig and nameDest — just unique account IDs, no predictive value
- Dropped isFlaggedFraud
- Filtered to TRANSFER and CASH_OUT only since this is where all the fraud is
- Created ('isAccountDrained') flags when origin balance hits zero after transaction
- Created ('errorBalanceOrig') flags when the balance math does not add up
- One-hot encoded transaction type
- Saved cleaned data as cleaned_transactions.csv



# Notebook 3: Model, Tuning & Evaluation
Built a Random Forest Classifier on the cleaned data.

- Used class_weight='balanced' to handle the class imbalance
- Split data 80/20 with stratify=y to preserve the fraud ratio
- Trained a baseline model first, then tuned with RandomizedSearchCV
- Used F1 Score as the primary metric — accuracy is misleading on imbalanced data
- Compared baseline vs tuned F1 to measure improvement
- Feature importance chart confirms that engineered features rank highest

---

# How to Run

1. Clone the repo
2. Add the dataset to your project folder — file is too large for GitHub
3. Run notebooks in order: EDA → Cleaning → Modeling
4. The cleaning notebook saves cleaned_transactions.csv which the modeling notebook loads

NOTE!!! This is a large dataset. If your kernel crashes (like mine did) restart and tune your hyperparamerters but do NOT decrease your test size.



# Imports used 

pandas, numpy, matplotlib, seaborn, scikit-learn


