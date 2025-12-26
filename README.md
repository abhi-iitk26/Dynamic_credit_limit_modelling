# Dynamic Credit Limit Prediction

## 📌 Project Overview
This project aims to build a machine learning model that dynamically predicts whether a credit card customer's credit limit should be **Increased**, **Maintained**, or **Decreased**. By analyzing demographic data, payment history, and bill statements, the system provides data-driven recommendations to optimize credit risk management and customer satisfaction.

## 📊 Dataset
The project uses a dataset of credit card clients in Taiwan (April 2005 - September 2005).
*   **Target Variable**: `limit_decision` (Categorical: Increase, Maintain, Decrease)
*   **Features**:
    *   **Demographics**: Age, Gender, Education, Marital Status.
    *   **Financials**: Credit Limit (`LIMIT_BAL`), Bill Amounts (`BILL_AMT1` - `BILL_AMT6`), Payment Amounts (`PAY_AMT1` - `PAY_AMT6`).
    *   **History**: Repayment status for the past 6 months (`PAY_0` - `PAY_6`).

## 🛠️ Methodology & Workflow

### 1. Data Preprocessing & Cleaning
*   Loaded dataset and cleaned column names.
*   Checked for missing values and duplicates.
*   Converted categorical variables to appropriate data types.

### 2. Exploratory Data Analysis (EDA)
*   **Univariate Analysis**: Analyzed distributions of Age, Credit Limit, and Bill Amounts. Identified heavy right-skewness in financial features.
*   **Bivariate Analysis**: Examined relationships between features and the target variable (`limit_decision`) using box plots and count plots.
*   **Correlation Analysis**: Detected high multicollinearity among `BILL_AMT` features using a heatmap.

### 3. Feature Engineering
Created new features to capture customer behavior trends:
*   `AVG_UTILIZATION_3M`: Average credit utilization over the last 3 months.
*   `AVG_PAY_RATIO_3M`: Average payment-to-bill ratio.
*   `LATE_PAYMENT_COUNT`: Total number of late payments in history.

### 4. Feature Selection
Applied statistical tests to select the most predictive features:
*   **Correlation Matrix**: Removed highly correlated features (e.g., redundant `BILL_AMT` columns).
*   **Chi-Square Test**: Selected significant categorical features (e.g., `PAY_0`, `EDUCATION`).
*   **ANOVA F-test**: Selected significant numerical features.
*   **VIF (Variance Inflation Factor)**: Iteratively removed features with VIF > 6 to eliminate multicollinearity.

### 5. Data Transformation Pipeline
Built a Scikit-Learn `ColumnTransformer` pipeline:
*   **Numerical Features**: Applied **Yeo-Johnson Power Transformation** to handle skewness and outliers, followed by **StandardScaler**.
*   **Categorical Features**: Applied **One-Hot Encoding** for nominal variables (`SEX`, `EDUCATION`, `MARRIAGE`).

### 6. Handling Class Imbalance
*   Identified imbalance in the target classes (Majority: 'Maintain').
*   Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data to balance the classes and prevent model bias.

### 7. Model Training & Evaluation
Evaluated multiple baseline models:
*   Logistic Regression
*   Decision Tree
*   Random Forest
*   AdaBoost
*   Support Vector Machine (SVM)
*   **XGBoost (Extreme Gradient Boosting)**

### 8. Hyperparameter Tuning
*   Selected **XGBoost** as the top-performing candidate.
*   Performed **GridSearchCV** to optimize hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`).
*   Evaluated the final model using **Accuracy**, **Classification Report**, and **Confusion Matrix**.

## 🚀 Technologies Used
*   **Language**: Python
*   **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE).

## 📈 Key Results
*   The project successfully addressed data skewness using Power Transformers.
*   SMOTE effectively balanced the training data, allowing the model to learn minority classes ('Increase'/'Decrease') better.
*   The tuned **XGBoost** model provided the best performance in classifying credit limit recommendations.
