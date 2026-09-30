# Credit Risk Prediction Model

A machine learning project for predicting whether a loan applicant is likely to default. The project uses the Credit Risk Dataset, performs data cleaning and feature engineering, compares multiple classification algorithms, and saves the selected Random Forest model for reuse.

## Project Overview

The goal of this project is to build a binary classification system for credit-risk prediction:

- `0` → Lower Credit Risk / Likely No Default
- `1` → Higher Credit Risk / Likely Default

The project was developed as part of a **CodeAlpha Machine Learning Internship** project.

> **Note:** This project is for educational and demonstration purposes. Model predictions are estimates and should not be treated as actual financial or lending decisions.

## Objectives

- Explore and understand the credit-risk dataset.
- Clean invalid and inconsistent records.
- Handle missing values.
- Perform exploratory data analysis.
- Engineer additional features.
- Train and compare multiple classification models.
- Evaluate models using standard classification metrics.
- Select a model using ROC-AUC as the comparison criterion.
- Save the trained model and evaluation results for reuse.

## Dataset

The project uses `credit_risk_dataset.csv`.

### Dataset size

- Original records: **32,581**
- Records after preprocessing: **32,409**
- Original columns: **12**
- Features used for modeling: **13**
- Training records: **25,927**
- Testing records: **6,482**

### Original features

| Feature | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Applicant income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_status` | Target variable |
| `loan_percent_income` | Loan amount as a proportion of income |
| `cb_person_default_on_file` | Historical default indicator |
| `cb_person_cred_hist_length` | Length of credit history |

## Data Preprocessing

The notebook performs the following preprocessing steps:

1. Removes duplicate records.
2. Removes unrealistic age values by retaining ages from 18 to 100.
3. Cleans employment-length values using age as a consistency check.
4. Handles missing numerical values using median imputation.
5. Handles missing categorical values using most-frequent imputation.
6. Standardizes numerical features.
7. Applies one-hot encoding to categorical features.
8. Splits the data into training and testing sets.

The dataset initially contained missing values in `person_emp_length` and `loan_int_rate`.

## Feature Engineering

Two additional features were created:

### 1. Income-to-loan ratio

```text
income_loan_ratio = person_income / (loan_amnt + 1)
```

This provides an additional representation of the relationship between applicant income and requested loan amount.

### 2. Credit-history-to-age ratio

```text
credit_history_age_ratio = cb_person_cred_hist_length / (person_age + 1)
```

This represents credit-history length relative to applicant age.

## Machine Learning Models

Three classification algorithms were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**

The project uses a preprocessing pipeline so numerical and categorical variables can be transformed consistently before model training.

## Model Performance

The models were evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8093 | 0.5448 | 0.7807 | 0.6417 | 0.8721 |
| Decision Tree | 0.9156 | 0.8578 | 0.7362 | 0.7924 | 0.9051 |
| Random Forest | 0.9098 | 0.8182 | 0.7553 | 0.7855 | 0.9290 |

### Model selection

The notebook selects the **Random Forest** model based on the highest ROC-AUC among the three tested models.

- Random Forest ROC-AUC: **0.9290**

This selection criterion is implemented directly in the notebook by comparing the ROC-AUC scores.

## Example Prediction

The notebook demonstrates prediction for a sample customer with:

- Age: 30
- Income: 50,000
- Home ownership: RENT
- Employment length: 5 years
- Loan intent: PERSONAL
- Loan grade: B
- Loan amount: 10,000
- Interest rate: 11.5%
- Loan-to-income proportion: 20%
- Previous default on file: N
- Credit history length: 8 years

The saved Random Forest model produced:

- **Prediction:** `0`
- **Estimated default probability:** `24.24%`

Again, this is an example machine-learning output and not a real lending decision.

## Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Anaconda

## Project Structure

A recommended GitHub repository structure is:

```text
credit-risk-prediction/
│
├── credit scoring model project.ipynb
├── credit_risk_dataset.csv
├── credit_risk_random_forest_model.pkl
├── model_comparison_results.csv
├── credit_risk_project_results.txt
├── final_project_summary.txt
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create and activate a virtual environment if desired:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
credit scoring model project.ipynb
```

Run the notebook cells from top to bottom.

Make sure `credit_risk_dataset.csv` is available in the expected project directory before running the notebook.

## Model Evaluation

The project uses the following evaluation metrics:

- **Accuracy** — proportion of total predictions classified correctly.
- **Precision** — proportion of predicted positive/default cases that are actually positive/default.
- **Recall** — proportion of actual positive/default cases detected by the model.
- **F1 Score** — harmonic mean of precision and recall.
- **ROC-AUC** — measures the model's ability to distinguish between the two classes across classification thresholds.

The notebook also generates confusion matrices and ROC curves for model comparison.

## Results Summary

The completed project includes:

- Data exploration and cleaning
- Missing-value handling
- Feature engineering
- Numerical scaling
- Categorical encoding
- Training of three classification models
- Model evaluation
- ROC curve comparison
- Random Forest model selection
- Example customer prediction
- Saved trained model
- Saved model comparison results

## Limitations

- The model is trained on a specific historical dataset and may not generalize to other populations or time periods.
- Credit-risk prediction can involve fairness, privacy, and regulatory considerations that are not addressed by this educational project.
- The example probability is model output, not a calibrated financial risk assessment.
- Further validation, calibration, monitoring, and domain-specific review would be required for real-world deployment.

## Future Improvements

Possible extensions include:

- Hyperparameter tuning with cross-validation.
- Probability calibration.
- Class-imbalance analysis and appropriate resampling strategies.
- Feature importance and model explainability using tools such as SHAP.
- Threshold optimization based on application-specific costs.
- Robust validation on an external dataset.
- A simple web application or API for interactive predictions.
- Model monitoring and periodic retraining.

