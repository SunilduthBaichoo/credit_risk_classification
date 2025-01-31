# Credit Risk Classification

Using various supervised learning techniques to train and evaluate a model based on loan risk.

## Overview of the Analysis

### 1. Purpose of the Analysis
The goal of this analysis was to predict loan risk—whether a borrower is likely to default or not. This helps financial institutions and lenders assess risk before issuing loans. By analyzing historical lending data, the model aims to classify loans into two categories:

- **0 (Healthy Loan):** The borrower is likely to repay the loan.
- **1 (High-Risk Loan):** The borrower is at risk of defaulting.

By building a classification model, lenders can make informed decisions, minimize financial risk, and offer appropriate interest rates and credit limits.

### 2. Financial Information in the Data
The dataset contains key financial details about borrowers and their loan applications. Here are the main input data:

| **Input data**        | **Description** |
|---------------------|----------------|
| `loan_size`        | The amount of the loan (in dollars). |
| `interest_rate`    | The percentage interest applied to the loan. |
| `borrower_income`  | The borrower's annual income. |
| `debt_to_income`   | Ratio of total debt to annual income (indicating financial burden). |
| `num_of_accounts`  | Number of accounts held by the borrower. |
| `derogatory_marks` | Number of negative credit marks (e.g., defaults). |
| `total_debt`       | Total outstanding debt of the borrower. |
| `loan_status`      | **Target variable** (0: Healthy Loan, 1: High-Risk Loan). |

The dataset captures financial and credit-related information that affects a borrower's ability to repay loans.

### 3. Distribution of Loan Status (`loan_status`)
Before training the model, we check the distribution of the `loan_status` variable:

```python
# Count distribution of loan status
y.value_counts()
```

| Loan Status | Count |
|-------------|--------|
| 0 (Healthy Loan) | 75,036 |
| 1 (High-Risk Loan) | 2,500 |

**Imbalance Observation:** From the full dataset, there are **75,036 healthy loans** and only **2,500 high-risk loans**, making the dataset highly imbalanced. This means the model needs to carefully handle class imbalances to avoid being biased toward the majority class.

### 4. Machine Learning Process
The machine learning process followed these key stages:

#### A. Data Preparation
- Checked for missing values and cleaned the data if needed.
- Explored the distribution of variables to understand patterns.
- Feature scaling may have been applied (e.g., standardizing loan amounts and debt ratios), if needed.

#### B. Splitting the Data
- Divided the dataset into training and testing sets (e.g., 75% for training, 25% for testing).
- Ensured stratification to maintain class distribution.

#### C. Model Selection
- Chose **Logistic Regression** because it is a well-known algorithm for binary classification problems and provides interpretable coefficients.

#### D. Model Training
- Trained `LogisticRegression` using the training data.
- Adjusted hyperparameters if necessary.

#### E. Model Evaluation
- **Classification Report:** Precision, recall, and F1-score were used to evaluate the model.
- **Accuracy:** Overall model correctness.
- **Class imbalance handling:** Since the dataset had fewer high-risk loans, techniques like weighting or oversampling could be used.

### 5. Methods Used
- **`LogisticRegression`**: A simple yet effective model for binary classification.
- **Evaluation Metrics:**
  - **Precision:** Measures correctness of positive predictions.
  - **Recall:** Measures how well the model identifies actual high-risk loans.
  - **F1-score:** A balance between precision and recall.
  - **Accuracy:** Overall performance.

Other possible methods that could improve classification include:
- **Decision Trees / Random Forests**: Non-linear models that handle complex patterns better.
- **SMOTE (Synthetic Minority Oversampling Technique)**: To handle class imbalance.
- **Gradient Boosting (XGBoost, LightGBM)**: More advanced models that improve accuracy.

## Results

## Model 1 : Logistic Regression Model with the Original Data

### **Class 0 (Healthy Loan) Performance:**
- **Precision:** 1.00 → When the model predicts a loan as healthy, it is correct 100% of the time.
- **Recall:** 0.99 → The model correctly identifies 99% of all actual healthy loans.
- **F1-score:** 1.00 → A strong balance between precision and recall.
- **Support:** 18,759 → The dataset contains significantly more healthy loans.

- **Conclusion:** 
- The model is very accurate for healthy loans, meaning it correctly predicts nearly all 0s.
- The slightly lower recall (0.99) suggests that some healthy loans are misclassified as high-risk (1), though this is minimal.


### **Class 1 (High-Risk Loan) Performance:**
- **Precision:** 0.86 → The model predicts a loan as high-risk 86% of the time correctly.
- **Recall:** 0.95 → The model successfully identifies 95% of all actual high-risk loans.
- **F1-score:** 0.90 → There is a strong balance between precision and recall.
- **Support:** 625 → The high-risk loan class is much smaller, making classification more challenging.
- **Conclusion:** 
- Recall is high (0.95), meaning the model effectively identifies high-risk loans.
- However, precision is lower (0.86), indicating some healthy loans are being misclassified as high-risk (false positives).
- The F1-score of 0.90 suggests a strong balance between catching high-risk loans and avoiding too many false positives.

### **Overall Model Performance:**
- **Accuracy:** 0.99 → The model classifies 99% of all loans correctly.
- **Macro Avg (Average across both classes):**
  - Precision: 0.93
  - Recall: 0.97
  - F1-score: 0.95
- **Weighted Avg (Accounts for class imbalance):**
  - Precision, Recall, and F1-score are all close to 0.99, meaning the model is highly accurate but influenced by the large number of healthy loans.
  
## **Conclusion for Model 1 (Logistic Regression with original data):**
  - The model performs very well in general, with high overall accuracy (99%).
  - The macro recall (0.97) indicates that most loans are correctly classified.
  - Precision for high-risk loans (0.86) is slightly lower, meaning some healthy loans are mistakenly classified as high-risk.
  - There may be some bias toward class 0, which is expected given the dataset imbalance.

## Model 2 : Logistic Regression Model with the Original Data after applying SMOTE
The classification report after applying SMOTE (Synthetic Minority Over-sampling Technique) shows the performance of the model in handling class imbalance. The following are the key metrics.

### **Class 0 (Healthy Loan) Performance:**
- **Precision:** 1.00 → When the model predicts a loan as healthy, it is correct 100% of the time.
- **Recall:** 1.00 → The model successfully identifies 100% of all healthy loans.
- **F1-score:** 1.00 → The perfect balance between precision and recall.
- **Support:** 18,759 → The dataset is still highly dominated by healthy loans.

- **Conclusion:** - The model is performing exceptionally well for healthy loans, as expected.

### **Class 1 (High-Risk Loan) Performance:**
- **Precision:** 0.87 → When the model predicts a loan as high-risk, 87% of those predictions are correct.
- **Recall:** 0.96 → The model correctly identifies 96% of actual high-risk loans.
- **F1-score:**  0.91 → A strong balance between precision and recall.

- **Conclusion:**

- Recall is very high (0.96), meaning the model successfully captures almost all high-risk loans.
- Precision is slightly lower (0.87), indicating that some healthy loans are mistakenly classified as high-risk (false positives).
- The F1-score (0.91) shows a strong balance, meaning the model is effective at both identifying high-risk loans and minimizing misclassification.


### **Overall Model 2 Performance:**

- **Accuracy:** 0.99 → The model correctly classifies 99% of all loans.
- **Macro Average (Across both classes):**
  - Precision: 0.94
  - Recall: 0.98
  - F1-score: 0.96
- **Weighted Average (Accounts for class imbalance):**
  - Precision, Recall, and F1-score remain at 0.99, indicating strong overall model performance.

## Conclusion for Model 2 (after applying SMOTE) :

- The model is highly accurate, with balanced performance across both classes.
- The macro recall (0.98) suggests that almost all loans are classified correctly.
- Precision for high-risk loans (0.87) is slightly lower, meaning some healthy loans are mistakenly flagged as high-risk.


## Summary

Summarizing the results of the machine learning model and recommendations:

- **Which model performs best?**  Logistic Regression Model with the Original Data after applying SMOTE has slightly better performance and is therefore recomended.
- **Does performance depend on the problem we are trying to solve?** Yes, if detecting high-risk loans is more important, we may need to optimize recall and handle class imbalance better.

### **Key Takeaways:**
- The model is highly accurate overall, but class imbalance skews the results. Since most loans are healthy (0), the high accuracy is largely due to correct predictions in that class.
- High-risk loans (1) are detected well, but the **precision (0.87) could be improved** to reduce false positives.

### **Potential Improvements:**
- Adjust decision thresholds to balance precision and recall.
- Use resampling techniques (oversampling high-risk loans or undersampling healthy loans).
- Try alternative models like **Random Forest or Gradient Boosting**, which might handle class imbalance better.

---

This analysis provides a **foundation for credit risk assessment**, helping lenders make informed decisions. Future work may involve **advanced ensemble models** or **deep learning techniques** to further enhance loan risk predictions.

