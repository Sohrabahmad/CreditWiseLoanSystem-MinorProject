# 💳 CreditWise: Loan Approval Prediction

## 📌 Project Overview

**CreditWise** is a machine learning project that predicts whether a loan application will be approved based on an applicant's financial, personal, and credit-related information.

The project covers data preprocessing, exploratory data analysis (EDA), categorical encoding, feature scaling, model training, evaluation, and feature engineering.

Three machine learning algorithms are compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

## 🎯 Project Objectives

- Analyze loan applicant data and approval patterns.
- Handle missing numerical and categorical values.
- Visualize income, credit scores, and loan approval distributions.
- Encode categorical variables for machine learning.
- Train and compare classification algorithms.
- Evaluate models using accuracy, precision, recall, and F1 score.
- Apply feature engineering to improve model performance.

## 📂 Dataset Description

The project uses `loan_approval_data.csv`.

- **Total records:** 1,000
- **Total columns:** 20
- **Target variable:** `Loan_Approved`

### Features

| Feature | Description |
|---|---|
| `Applicant_ID` | Unique applicant identifier |
| `Applicant_Income` | Applicant's income |
| `Coapplicant_Income` | Co-applicant's income |
| `Employment_Status` | Employment category |
| `Age` | Applicant's age |
| `Marital_Status` | Marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Applicant's credit score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-income ratio |
| `Savings` | Applicant's savings |
| `Collateral_Value` | Value of collateral |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan repayment term |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Property location category |
| `Education_Level` | Education level |
| `Gender` | Applicant's gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target: approval status |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## ⚙️ Project Workflow

1. Load the loan approval dataset.
2. Inspect the data using `head()`, `info()`, `describe()`, and missing-value checks.
3. Impute missing numerical values using the mean.
4. Impute missing categorical values using the most frequent value.
5. Perform exploratory data analysis and visualization.
6. Remove the applicant ID column.
7. Encode categorical features.
8. Separate input features and the target variable.
9. Split the dataset into training and testing sets.
10. Apply feature scaling using `StandardScaler`.
11. Train Logistic Regression, KNN, and Gaussian Naive Bayes.
12. Evaluate the models using classification metrics.
13. Create additional features using feature engineering.
14. Retrain and compare the models.

## 📊 Exploratory Data Analysis

The notebook explores the dataset using the following visualizations:

- Pie chart of loan approval distribution
- Bar chart of education levels
- Histograms of applicant and co-applicant income
- Box plots for income, credit score, DTI ratio, and savings
- Credit score distribution by loan approval status
- Applicant income distribution by loan approval status
- Correlation heatmap for numerical features

These visualizations help identify patterns, distributions, and relationships within the dataset.

## 🧹 Data Preprocessing

### 1. Missing Value Handling

The project uses `SimpleImputer`:

- Numerical columns: mean imputation
- Categorical columns: most-frequent-value imputation

### 2. Categorical Encoding

- `LabelEncoder` is used for `Education_Level` and `Loan_Approved`.
- `OneHotEncoder` is used for employment status, marital status, loan purpose, property area, employer category, and gender.
- `drop="first"` is used to reduce redundant dummy columns.

### 3. Feature Scaling

`StandardScaler` standardizes the features before model training.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted on the training data and then applied to the test data.

## 🤖 Machine Learning Models

### 1. Logistic Regression

A classification algorithm that estimates the probability of loan approval based on applicant features.

### 2. K-Nearest Neighbors (KNN)

A distance-based algorithm that predicts the class based on nearby training examples. The notebook uses `n_neighbors=5`.

### 3. Gaussian Naive Bayes

A probabilistic classifier based on Bayes' theorem, assuming that numerical features follow Gaussian distributions within each class.

## 📈 Model Evaluation

The dataset is split using an 80:20 ratio with `random_state=42`.

### Initial Model Results

The following results were obtained using the notebook's preprocessing and evaluation approach:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.50% | 78.33% | 77.05% | 77.69% |
| KNN (K=5) | 76.00% | 62.75% | 52.46% | 57.14% |
| Gaussian Naive Bayes | 86.50% | 80.36% | 73.77% | 76.92% |

### Results After Feature Engineering

The notebook creates squared features for `DTI_Ratio` and `Credit_Score`, then removes the original versions before retraining the models.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | **87.50%** | 79.03% | **80.33%** | **79.67%** |
| KNN (K=5) | 75.50% | 62.00% | 50.82% | 55.86% |
| Gaussian Naive Bayes | 86.50% | 78.33% | 77.05% | 77.69% |

**Best result:** Logistic Regression after feature engineering achieved 87.50% test accuracy and an F1 score of 79.67% on the held-out test set.

## 🔬 Feature Engineering

The notebook creates two additional features:

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
```

The original `DTI_Ratio` and `Credit_Score` columns are then excluded from the model input, while their squared versions are retained.

This experiment investigates whether nonlinear transformations improve classification performance.

## 📁 Project Structure

```text
CreditWise/
├── credit_wise.ipynb
├── loan_approval_data.csv
└── README.md
```

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd CreditWise
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Notebook

Open `credit_wise.ipynb` and execute the cells sequentially. Ensure `loan_approval_data.csv` is in the notebook's working directory.

## 📚 Learning Outcomes

- Data cleaning and missing-value imputation
- Exploratory data analysis
- Data visualization using Matplotlib and Seaborn
- Label encoding and one-hot encoding
- Feature standardization
- Classification algorithms
- Model evaluation and comparison
- Feature engineering
- Interpreting precision, recall, and F1 score

## ⚠️ Important Considerations

- The notebook imputes missing values in `Loan_Approved` as well as the other categorical columns. In a production workflow, rows with unknown target labels should generally be excluded from supervised training rather than assigned an inferred label.
- The dataset is imputed and encoded before the train-test split. For a more robust evaluation, fit preprocessing steps only on training data using a Scikit-learn pipeline.
- Loan approval is a high-impact financial decision. This model is for educational purposes and should not be used as the sole basis for real lending decisions.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (AI & ML)

---

⭐ If you find this project useful, consider starring the repository!
