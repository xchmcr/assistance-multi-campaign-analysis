from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#goal is to predict or optimize the expenditure
# Load the dataset
file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-data.csv'
df = pd.read_csv(file_path)

# Clean and preprocess the data as done before
df['Spend'] = df['Spend'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Spend2'] = df['Spend2'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Cost/Lead'] = df['Cost/Lead'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Cost/Ascend App'] = df['Cost/Ascend App'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Cost/Approved'] = df['Cost/Approved'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Top of Funnel Conversion Rate'] = df['Top of Funnel Conversion Rate'].replace({'%': ''}, regex=True).astype(float)
df['Approval Conversion Rate'] = df['Approval Conversion Rate'].replace({'%': ''}, regex=True).astype(float)

# Convert 'Week' to datetime
df['Week'] = pd.to_datetime(df['Week'])

# Define the target and features
target = 'Spend2'
features = ['Cost/Ascend App', 'Cost/Approved', 'Cost/Lead', 'Leads (from CRM)']

X = df[features]
y = df[target]

# Standardize the features before polynomial transformation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test with and without polynomial features
poly = PolynomialFeatures(degree=1, include_bias=False)  # Start with linear features
X_poly = poly.fit_transform(X_scaled)

# Define the R^2 scorer
r2_scorer = make_scorer(r2_score)

# Cross-validation with polynomial features and Ridge regression
alpha_value = 1  # Test different alpha values
model = Ridge(alpha=alpha_value)
scores = cross_val_score(model, X_poly, y, cv=3, scoring=r2_scorer)
average_r2 = np.mean(scores)
print("Average R^2 Score with Polynomial Features (Degree 1):", average_r2)

# Split the data into training and testing sets with polynomial features
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, shuffle=False, random_state=42)

# Fit the Ridge regression model to the polynomial features
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Convert predictions to a pandas Series to match indices
y_pred_series = pd.Series(y_pred, index=y_test.index)

# Calculate the Mean Squared Error and R^2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score with Polynomial Features (Degree 1):", r2)

# Compare the actual and predicted values
comparison_df = pd.DataFrame({'Actual Spend2': y_test, 'Predicted Spend2': y_pred_series})
print(comparison_df.head(20))

# Generate a distribution plot to compare actual and predicted values
plt.figure(figsize=(10, 6))
ax1 = sns.kdeplot(y_test, color="r", label="Actual Spend2")
sns.kdeplot(y_pred, color="b", label="Fitted Spend2", ax=ax1)
plt.legend()
plt.title('Distribution of Actual vs Predicted Spend2')
plt.show()

# Perform Grid Search for Ridge regression with polynomial features
parameters1 = [{'alpha': [0.0001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
RR = Ridge()
Grid1 = GridSearchCV(RR, parameters1, cv=4)
Grid1.fit(X_poly, y)

# Output the best model
print("Best Estimator with Polynomial Features:", Grid1.best_estimator_)

# View the scores for each alpha
scores = Grid1.cv_results_['mean_test_score']
print("Mean Test Scores with Polynomial Features:", scores)

# Plot the alpha values vs. the mean test scores
alphas = parameters1[0]['alpha']
plt.figure(figsize=(10, 6))
plt.plot(alphas, scores)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')
plt.title('Grid Search Results with Polynomial Features')
plt.show()

# Print the original DataFrame before one-hot encoding
print("Original DataFrame before one-hot encoding:")
print(df.head(20))
