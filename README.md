# Data Analysis 3 — Predictive Modeling and Applied Machine Learning

This repository contains three independent projects developed for the **Data Analysis 3** course at CEU.  
They progressively advance from linear regression and feature interpretation to machine learning–based prediction and business-oriented model optimization.  
All projects emphasize **data preparation, model evaluation, and visualization**.

---

## Project 1 — Predicting Earnings of Business Operations Specialists

**Goal:**  
Estimate hourly earnings based on education, experience, and demographics using OLS regression.

**Methods:**  
- Stepwise model building with interaction and quadratic terms  
- Evaluation using RMSE, CV-RMSE, and BIC  
- Visualization of model fit and residual patterns  

**Key Insight:**  
Model 2 (education + experience + gender + age) provides the best trade-off between interpretability and accuracy.  
Additional nonlinear terms do not improve prediction.

**Tools:** `Python`, `pandas`, `statsmodels`, `matplotlib`, `seaborn`


---

## Project 2 — Airbnb Price Prediction

**Goal:**  
Predict nightly Airbnb listing prices using property and host attributes.

**Methods:**  
- Data cleaning, missing-value imputation, and feature engineering  
- Encoding of categorical variables (location, room type, amenities)  
- Model comparison: Linear Regression, Ridge, Lasso, and Random Forest  
- Evaluation metrics: MAE, RMSE, R²  
- Visualization: feature importance, price distribution, prediction vs. actual plots  

**Key Insight:**  
Nonlinear models (especially Random Forest) outperform linear ones by capturing location and property heterogeneity.  
Feature importance shows **location** and **room type** as dominant drivers of price variation.

**Tools:** `Python`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`


---

## Project 3 — Predicting High-Growth Firms (Bisnode Panel)

**Goal:**  
Identify high-growth firms using financial indicators from Bisnode panel data (2010–2015).

**Methods:**  
- Models: Logistic Regression, Random Forest, XGBoost  
- Cost-sensitive loss function (FP = 100, FN = 1000)  
- Threshold tuning via expected loss minimization  
- 5-fold cross-validation for robustness  

**Results:**  
- **XGBoost** achieved **AUC = 0.887** with the lowest expected loss  
- Stronger predictability for service-sector firms  
- Outputs include ranked firm lists for early-stage investment targeting  

**Tools:** `Python`, `xgboost`, `scikit-learn`, `pandas`, `matplotlib`


---

## Summary of Skills
- Regression & Machine Learning  
- Feature Engineering & Model Evaluation  
- Cost-sensitive classification  
- Visualization & Business Interpretation  
- Reproducible Jupyter Workflows


# Data Analysis 4 — Causal Inference and Policy Evaluation

## Project: Impact of Malaria Vaccination on Child Mortality in Sub-Saharan Africa

### Overview
This project investigates the **causal impact** of the RTS,S malaria vaccine on under-five mortality rates across Sub-Saharan Africa (2003–2023).  
Using country-year panel data, it applies a **Difference-in-Differences (DiD)** approach with **two-way fixed effects** to isolate the vaccine’s effect.

---

### Data
- **Sources:** World Bank, UNDP  
- **Sample:** 37 countries × 20 years = 740 observations  
- **Key variables:** under-five mortality rate, malaria incidence, GDP per capita, education, sanitation, water access  
- **Cleaning:** Outlier removal, log transformation, panel balancing

---

### Methodology
$$
Mortality_{it} = \alpha + \beta (Post_t \times Treat_i) + \gamma_i + \delta_t + X_{it}'\theta + \varepsilon_{it}
$$
- **β** captures the average treatment effect of the vaccine program  
- Controls for both **country** and **year** fixed effects  
- Robustness checks include alternate samples and lag structures

---

### Key Findings
- **β = –10.64**, significant at 1% level → substantial reduction in child mortality  
- Adjusted R² = 0.946 → high model fit  
- Results robust to socioeconomic controls and fixed effects  



### Tools
`Python`, `pandas`, `linearmodels`, `statsmodels`, `matplotlib`
