import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load the dataset
file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-radiodata.csv'
df = pd.read_csv(file_path)

# Print the initial shape of the DataFrame
print("Initial DataFrame Shape:", df.shape)

# Data Cleaning and Preprocessing
numeric_columns = ['Ord. Buy Rate', '$ ORD', '$ SPENT', 'UNQ >= :01', 'Length', 'Ord Spots', 'Spots Ran']

# Function to clean and convert numeric columns
def clean_numeric(x):
    if isinstance(x, str):
        return pd.to_numeric(x.replace('$', '').replace(',', '').replace('%', ''), errors='coerce')
    return x

# Apply cleaning function to numeric columns
for col in numeric_columns:
    df[col] = df[col].apply(clean_numeric)

# Convert 'Week' to datetime
df['Week Of'] = pd.to_datetime(df['Week Of'], errors='coerce')

# Convert 'Days' and 'Times' to categorical
df['Days'] = df['Days'].astype('category')
df['Times'] = df['Times'].astype('category')

# Check for infinite values and replace with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Print summary of missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop rows with NaN values in numeric columns
df_cleaned = df.dropna(subset=numeric_columns)

print("\nDataFrame Shape After Cleaning:", df_cleaned.shape)

# Print summary statistics
print("\nSummary Statistics:")
print(df_cleaned[numeric_columns].describe())

# Check for remaining infinite values
inf_check = np.isinf(df_cleaned[numeric_columns]).sum()
print("\nInfinite values after cleaning:")
print(inf_check)

# Feature Engineering
df_cleaned['Efficiency'] = df_cleaned['$ ORD'] / df_cleaned['$ SPENT']
df_cleaned['Efficiency'] = df_cleaned['Efficiency'].replace([np.inf, -np.inf], np.nan)

# One-hot encoding for categorical features
df_encoded = pd.get_dummies(df_cleaned, columns=['Days', 'Times', 'Station', 'Product'], drop_first=True)


# Select features and target
features = ['Ord. Buy Rate', '$ ORD', 'UNQ >= :01', 'Length', 'Ord Spots', 'Spots Ran', 'Efficiency']
X = df_encoded[features + [col for col in df_encoded.columns if col.startswith(('Days_', 'Times_', 'Station_', 'Product_'))]]
y = df_encoded['$ SPENT']
#
# Remove any rows with NaN or infinite values
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y[X.index]
#
# Important: Align df_cleaned with X
df_cleaned = df_cleaned.loc[X.index]

# Print feature information
print("\nFeature Information:")
print(X.info())

# Check for any remaining NaN or infinite values
print("\nRemaining NaN values:")
print(X.isnull().sum())
print("\nRemaining infinite values:")
print(np.isinf(X).sum())

# Remove any rows with NaN or infinite values
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y[X.index]

print("\nFinal shape of X:", X.shape)
print("Final shape of y:", y.shape)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ridge Regression Model
ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print("\nRidge Regression MSE:", ridge_mse)
print("Ridge Regression R2 Score:", ridge_r2)

# XGBoost Model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print("\nXGBoost MSE:", xgb_mse)
print("XGBoost R2 Score:", xgb_r2)

###
df_cleaned['Ridge_Predicted_$ SPENT'] = ridge_model.predict(X_poly)
df_cleaned['XGBoost_Predicted_$ SPENT'] = xgb_model.predict(X_poly)
###

# Visualizations
plt.figure(figsize=(12, 6))
sns.kdeplot(y_test, color="r", label="Actual Value")
sns.kdeplot(ridge_pred, color="b", label="Ridge Predicted Values")
sns.kdeplot(xgb_pred, color="g", label="XGBoost Predicted Values")
plt.legend()
plt.title("Distribution of Actual vs Predicted $ SPENT")
plt.show()

# Feature Importance for XGBoost
plt.figure(figsize=(12, 8))
feature_importance = xgb_model.feature_importances_
feature_names = poly.get_feature_names_out(X.columns)
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

# Efficiency Analysis
df_cleaned['Ridge_Predicted_$ SPENT'] = ridge_model.predict(X_poly)
df_cleaned['XGBoost_Predicted_$ SPENT'] = xgb_model.predict(X_poly)
df_cleaned['Ridge_Efficiency'] = df_cleaned['$ ORD'] / df_cleaned['Ridge_Predicted_$ SPENT']
df_cleaned['XGBoost_Efficiency'] = df_cleaned['$ ORD'] / df_cleaned['XGBoost_Predicted_$ SPENT']

# Find the most efficient combinations
efficient_combinations = df_cleaned.sort_values('XGBoost_Efficiency', ascending=False).head(20)

print("\nMost Efficient Combinations based on XGBoost model:")
print(efficient_combinations[['Station', 'Product', 'Days', 'Times', '$ SPENT', 'XGBoost_Predicted_$ SPENT', 'XGBoost_Efficiency']])

# Scatter plot of Actual vs Predicted $ SPENT
plt.figure(figsize=(10, 8))
plt.scatter(y_test, ridge_pred, alpha=0.5, label='Ridge')
plt.scatter(y_test, xgb_pred, alpha=0.5, label='XGBoost')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual $ SPENT')
plt.ylabel('Predicted $ SPENT')
plt.title('Actual vs Predicted $ SPENT')
plt.legend()
plt.show()