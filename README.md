# Concrete Compressive Strength Prediction

This repository contains an end-to-end machine learning project aimed at predicting the compressive strength of concrete (`csMPa`) based on its component ingredients and age. The project follows a complete machine learning workflow from data exploration and feature engineering to model training, cross-validation, and hyperparameter tuning.

## Project Overview
The objective is to build a regression model that accurately estimates concrete compressive strength. Multiple algorithms—Linear Regression, Decision Tree Regressor, and Random Forest Regressor—are evaluated using 10-fold cross-validation and tuned using grid search.

## Dataset
The project uses the `Concrete_Data_Yeh.csv` dataset (1030 instances). The target variable is `csMPa` (Concrete Compressive Strength in Megapascals).

Input features include:
* **Cement** ($kg/m^3$)
* **Blast Furnace Slag** ($kg/m^3$)
* **Fly Ash** ($kg/m^3$)
* **Water** ($kg/m^3$)
* **Superplasticizer** ($kg/m^3$)
* **Coarse Aggregate** ($kg/m^3$)
* **Fine Aggregate** ($kg/m^3$)
* **Age** (Days)

## Workflow & Methodology

1. **Exploratory Data Analysis (EDA)**
   * Analyzed feature distributions using histograms (noted tail-heavy distributions for cement, slag, fly ash, and age).
   * Computed Pearson correlation coefficients. Identified strong positive correlations with `cement` and `superplasticizer`, and a negative correlation with `water`.

2. **Feature Engineering**
   Engineered Domain-specific features to capture chemical and physical relationships:
   * `w/c`: Water-to-cement ratio ($water / cement$) — strong negative correlation with strength.
   * `total_binder`: Sum of cement, slag, and fly ash ($cement + slag + flyash$).
   * `log(age)`: Logarithmic transform of age to linearize the curing curve.

3. **Preprocessing Pipeline**
   * Train/Test Split: 80% training, 20% testing using `train_test_split(random_state=67)`.
   * Data Pipeline: Built a Scikit-Learn `Pipeline` with `SimpleImputer(strategy="median")` and `StandardScaler()`.

4. **Model Training & Evaluation**
   Evaluated base models using Root Mean Squared Error (RMSE) and 10-fold cross-validation (`cv=10`):
   * **Linear Regression**: Baseline model.
   * **Decision Tree Regressor**: Captures non-linear dynamics.
   * **Random Forest Regressor**: Best performing base model (~4.70 Cross-Validation RMSE).

5. **Hyperparameter Tuning**
   * Performed `GridSearchCV` on `RandomForestRegressor` testing parameters for `n_estimators`, `max_features`, and `bootstrap`.

## Dependencies
* Python 3.x
* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`

## How to Run
1. Clone this repository to your local environment.
2. Ensure `Concrete_Data_Yeh.csv` is present in the working directory.
3. Open and run `My work.ipynb` in Jupyter Notebook or JupyterLab.
