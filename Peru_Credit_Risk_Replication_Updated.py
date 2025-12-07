
# Libraries imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, lasso_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. DATA LOADING & CLEANING
def load_and_process_exact(filepath):
    print("Loading and processing data...")
    df = pd.read_csv(filepath, header=2)
    df.columns = df.columns.str.strip()
    
    # Clean Numerical Columns
    def clean_currency(x):
        if isinstance(x, str):
            return float(x.replace('"', '').replace(',', '').strip())
        return x
    
    df['MONTO COBERTURADO (S/)'] = df['MONTO COBERTURADO (S/)'].apply(clean_currency)
    
    # Generate Risk Level (Hard-coded thresholds)
    def calculate_risk_level(amount):
        if pd.isna(amount): return np.nan
        if amount <= 4890.2: return 1
        elif amount <= 11760: return 2
        elif amount <= 30079.7: return 3
        else: return 4

    df['risk_level'] = df['MONTO COBERTURADO (S/)'].apply(calculate_risk_level)
    
    # Select Features
    feature_cols = [
        'SECTOR ECONÓMICO', 
        'NOMBRE DE ENTIDAD OTORGANTE DEL CRÉDITO', 
        'DEPARTAMENTO',
        'MONTO COBERTURADO (S/)' 
    ]
    
    # Drop missing values
    df_clean = df.dropna(subset=feature_cols + ['risk_level']).copy()
    
    # Ordinal Encoding
    cat_cols = ['SECTOR ECONÓMICO', 'NOMBRE DE ENTIDAD OTORGANTE DEL CRÉDITO', 'DEPARTAMENTO']
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype('category')
        df_clean[col] = df_clean[col].cat.codes

    # Prepare X and y
    X = df_clean[feature_cols].values
    y = df_clean['risk_level'].values # Keep as 1D array
    
    return X, y, feature_cols

# 2. REGRESSION WITH MODIFIED NORMALIZATION (Unscaled Y)
def run_optimized_regression(X, y, feature_names):
    # 1. Min-Max Scaling (Only X, keep Y original 1-4)
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Y is NOT scaled to [0,1], kept as [1, 2, 3, 4] to match paper coefficients magnitude
    y_train_raw = y
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_train_raw, test_size=0.3, random_state=42)
    
    print("\nFinding Optimal Lambdas (Alphas) via Cross-Validation...")
    
    # 3. Ridge Optimization
    alphas_ridge = np.logspace(-5, 1, 100)
    ridge_cv = RidgeCV(alphas=alphas_ridge, scoring='neg_mean_squared_error', cv=5)
    ridge_cv.fit(X_train, y_train)
    best_alpha_ridge = ridge_cv.alpha_
    print(f"Best Ridge Alpha: {best_alpha_ridge:.6f}")

    # 4. Lasso Optimization
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    best_alpha_lasso = lasso_cv.alpha_
    print(f"Best Lasso Alpha: {best_alpha_lasso:.6f}")
    
    # 5. Final Models Comparison
    models = {
        'OLS': LinearRegression(),
        'Ridge (Opt)': Ridge(alpha=best_alpha_ridge),
        'Lasso (Opt)': Lasso(alpha=best_alpha_lasso),
        # Using paper alphas might behave differently due to scaling, but we include them
        'Ridge (Paper)': Ridge(alpha=0.00910),
        'Lasso (Paper)': Lasso(alpha=0.00038)
    }
    
    results = {}
    print("\nResults (Target: Original Risk Level [1-4]):")
    print("-" * 80)
    print(f"{'Model':<15} | {'RMSE':<8} | {'R²':<8} | {'Intercept':<9} | {'Coefficients'}")
    print("-" * 80)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) 
        r2 = r2_score(y_test, y_pred)
        
        coef_str = ", ".join([f"{c:.2f}" for c in model.coef_])
        print(f"{name:<15} | {rmse:.5f}   | {r2:.5f}   | {model.intercept_:.2f}      | [{coef_str}]")
        
        results[name] = {'rmse': rmse, 'r2': r2, 'coefficients': model.coef_, 'model': model}

    return results, X_test, y_test

# 3. RANDOM FOREST & VISUALIZATION
def run_and_plot_comparison(X, y, feature_names, linear_results, X_test_lin, y_test_lin):
    print("\nRunning Random Forest...")
    
    # Scale X same as linear model for fairness, Y unscaled
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print(f"Random Forest   | {rmse_rf:.5f}   | {r2_rf:.5f}")
    
    # Visualization: Amount vs Risk (Actual vs Linear vs RF)
    # We need to recover the 'Amount' values from X_test (it was scaled)
    # Feature 3 (index 3) is MONTO COBERTURADO
    amount_col_idx = 3
    
    # Inverse transform is tricky if we only have X_test subset, so we just use the scaled value for x-axis
    # or re-fit scaler to reverse it. For visualization, scaled X is fine (0 to 1 represents min to max amount)
    
    x_vals = X_test[:, amount_col_idx]
    
    # Get Linear Predictions (using Lasso Opt)
    lasso_model = linear_results['Lasso (Opt)']['model']
    y_pred_lin = lasso_model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    
    # Plot a sample to avoid clutter (e.g., 200 points)
    sample_indices = np.random.choice(len(x_vals), 200, replace=False)
    
    plt.scatter(x_vals[sample_indices], y_test[sample_indices], color='black', label='Actual Risk', alpha=0.5, marker='o')
    plt.scatter(x_vals[sample_indices], y_pred_lin[sample_indices], color='red', label='Linear (Lasso)', alpha=0.5, marker='x')
    plt.scatter(x_vals[sample_indices], y_pred_rf[sample_indices], color='green', label='Random Forest', alpha=0.5, marker='^')
    
    plt.xlabel('Scaled Amount (Monto Coberturado)')
    plt.ylabel('Risk Level')
    plt.title('Why RF Wins: Modeling the "Step Function"')
    plt.legend()
    plt.grid(True)
    plt.show()

# Execution Block
try:
    X, y, feat_names = load_and_process_exact('Reactiva_Peru.csv')
    linear_results, X_test, y_test = run_optimized_regression(X, y, feat_names)
    run_and_plot_comparison(X, y, feat_names, linear_results, X_test, y_test)
except Exception as e:
    print(f"Error: {e}")

