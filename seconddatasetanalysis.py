import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

# Load the dataset
file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-gadata.csv'
df = pd.read_csv(file_path)

# Print the initial shape of the DataFrame
print("Initial DataFrame Shape:", df.shape)

# Convert 'Week of' to datetime
df['Week of'] = pd.to_datetime(df['Week of'])

# Drop columns with too many missing values
df.drop(columns=['Sessions - GA4, event based', 'Event value - GA4 (USD)'], inplace=True)

# Handle missing values by dropping rows with any NaN values in remaining numeric columns
df_cleaned = df.dropna(subset=['Event count - GA4'])

# Print remaining data types and column names
print("Data Types:")
print(df_cleaned.dtypes)
print("\nColumn Names:")
print(df_cleaned.columns)

# Define features and target variable
features = [
  "Session source - GA4", 
  "Session medium - GA4", 
  "Session campaign - GA4", 
  "Event name - GA4", 
  "Traffic source", 
  "Paid / Organic"
]
X = df_cleaned[features]
y = df_cleaned['Event count - GA4'] # Target

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Define the R^2 scorer
r2_scorer = make_scorer(r2_score)

# Cross-validation with polynomial features and Ridge regression
alpha_value = 10 # Using a higher alpha value for stronger regularization
model = Ridge(alpha=alpha_value)
scores = cross_val_score(model, X_poly, y, cv=3, scoring=r2_scorer)
average_r2 = np.mean(scores)
print("Average R^2 Score with Polynomial Features:", average_r2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, shuffle=False, random_state=42)

# Fit the Ridge regression model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Convert predictions to a pandas Series to match indices
y_pred_series = pd.Series(y_pred, index=y_test.index)

# Calculate Mean Squared Error and R^2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score with Polynomial Features:", r2)

# Compare actual and predicted values
comparison_df = pd.DataFrame({'Actual Event Count': y_test, 'Predicted Event Count': y_pred_series})
print(comparison_df.head(20))

# Plot distribution of actual and predicted values
plt.figure(figsize=(10, 6))
ax1 = sns.kdeplot(df_cleaned['Event count - GA4'], color="r", label="Actual Value")
sns.kdeplot(y_pred, color="b", label="Predicted Value", ax=ax1)
plt.legend()
plt.title("Distribution of Actual vs. Predicted Event Count")
plt.show()

# Perform Grid Search for Ridge regression with polynomial features
parameters1 = [{'alpha': [0.0001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, parameters1, cv=4)
Grid1.fit(X_poly, y)

# Output the best model
print("Best Estimator with Polynomial Features:", Grid1.best_estimator_)

# View scores for each alpha
scores = Grid1.cv_results_['mean_test_score']
alphas = parameters1[0]['alpha']
plt.figure(figsize=(10, 6))
plt.plot(alphas, scores)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')
plt.title('Grid Search Results with Polynomial Features')
plt.show()

# Print the original DataFrame before one-hot encoding
print("Original DataFrame before any transformations:")
print(df.head(20))