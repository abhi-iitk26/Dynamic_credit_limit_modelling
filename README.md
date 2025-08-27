# Dynamic_credit_limit_modelling

## Project Overview
A machine learning project that transforms traditional credit risk assessment by predicting optimal credit limit adjustments (Increase, Maintain, Decrease) based on customer payment behavior and financial patterns.

## Problem Statement
Traditional credit limit management relies on binary default prediction. This project innovates by creating a multi-class classification system that provides actionable business insights for proactive credit limit optimization.

## Dataset
- **Source**: UCI Credit Card Dataset
- **Size**: 30,000 customers
- **Features**: 23 attributes including payment history, bill amounts, demographic data
- **Target**: Custom engineered variable with 3 classes (Increase, Maintain, Decrease)

## Key Features

### Business Logic Implementation
- **Decrease Limit**: Customers with payment defaults
- **Increase Limit**: No recent payment delays + high credit utilization (>40%)
- **Maintain Limit**: All other non-defaulting customers

### Advanced Feature Engineering
- Average Credit Utilization (3-month rolling)
- Payment Ratio Analysis
- Late Payment Count (6-month window)
- Statistical significance testing

### Statistical Analysis
- **Chi-Square Test**: Categorical feature relationships
- **ANOVA F-test**: Numerical feature significance
- **VIF Analysis**: Multicollinearity detection and removal

## Technical Implementation

### Data Preprocessing Pipeline
```python
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer for unified preprocessing
- Train-test split with stratification
```

### Machine Learning Models
- **Classical ML**: Logistic Regression, Decision Tree, Random Forest, AdaBoost, SVM
- **Ensemble Methods**: XGBoost with hyperparameter tuning
- **Optimization**: GridSearchCV for parameter selection

### Performance Metrics
- **Best Model**: XGBoost Classifier
- **Accuracy**: 81%
- **Evaluation**: Classification reports, confusion matrices
- **Cross-validation**: 3-fold CV for hyperparameter tuning

## Key Technologies
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Statistical Tools**: SciPy, Statsmodels
- **Development Environment**: Jupyter Notebook

## Business Impact
- **Proactive Risk Management**: Early identification of credit limit adjustment needs
- **Customer Retention**: Appropriate limit increases for good customers
- **Revenue Optimization**: Better credit utilization through data-driven decisions
- **Risk Mitigation**: Systematic approach to limit decreases for high-risk customers

## Project Structure
```
dynamic_credit_limit_prediction/
├── dynamic_credit_limit.ipynb    # Main analysis notebook
├── UCI_Credit_Card.csv          # Dataset
├── PROJECT_DESCRIPTION.md       # This file
└── README.md                    # Setup instructions
```

## Future Enhancements
- Real-time prediction API development
- Advanced ensemble techniques (Stacking, Voting)
- Deep learning model implementation
- A/B testing framework for business validation
- Integration with banking systems

## Skills Demonstrated
- **Statistical Analysis**: Hypothesis testing, feature selection
- **Machine Learning**: Multi-class classification, hyperparameter tuning
- **Data Engineering**: Feature engineering, preprocessing pipelines
- **Business Intelligence**: Domain-specific problem solving
- **Performance Optimization**: Model selection and evaluation

---

*This project showcases the application of advanced machine learning techniques to solve real-world financial problems, demonstrating both technical proficiency and business acumen.*

