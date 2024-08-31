import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-data.csv'
df = pd.read_csv(file_path)

# Remove dollar signs, commas, and percentage symbols, then convert to numeric
df['Spend'] = df['Spend'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Spend2'] = df['Spend2'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Cost/Lead'] = df['Cost/Lead'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Convert 'Week' to datetime to ensure proper sorting
df['Week'] = pd.to_datetime(df['Week'])

# Print the first 20 rows to check the data
print(df.head(20))

# List of features to plot against 'Ascend Application'
features = ['Spend', 'Spend2', 'Leads (from CRM)', 'Unique Calls', 
            'Approved for Services', 'Cost/Lead', 'Cost/Ascend App', 
            'Cost/Approved', 'Top of Funnel Conversion Rate', 'Approval Conversion Rate']

# Set up the plotting environment
plt.figure(figsize=(15, 10))

# Generate scatter plots for each feature against 'Ascend Application'
for i, feature in enumerate(features):
    plt.subplot(3, 4, i+1)
    sns.scatterplot(x=df[feature], y=df['Ascend Application'])
    plt.title(f'Ascend Application vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Ascend Application')

# Adjust layout
plt.tight_layout()
plt.show()


# # Group by week and calculate the sum for each relevant column
# weekly_data = df.groupby('Week').agg({
#     'Spend': 'sum',
#     'Leads (from CRM)': 'sum',
#     'Ascend Application': 'sum',
#     'Approved for Services': 'sum'
# }).reset_index()

# # Calculate other features such as CPL, CPA Ascend App, CPA Approved
# weekly_data['CPL'] = weekly_data['Spend'] / weekly_data['Leads (from CRM)']
# weekly_data['CPA Ascend App'] = weekly_data['Spend'] / weekly_data['Ascend Application']
# weekly_data['CPA Approved'] = weekly_data['Spend'] / weekly_data['Approved for Services']

# # Replace infinite values and NaNs with 'N/A'
# weekly_data.replace([float('inf'), -float('inf')], 'N/A', inplace=True)
# weekly_data.fillna('N/A', inplace=True)

# # Drop rows with 'N/A' as these cannot be used in modeling
# weekly_data = weekly_data.replace('N/A', np.nan).dropna()

# # Select only the desired features for the polynomial regression model
# features = ["Spend", "Leads (from CRM)", "Ascend Application"]
# X = weekly_data[features]
# y = weekly_data['Approved for Services']  # Target

# # Standardize the features before polynomial transformation
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Add polynomial features
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Define the R^2 scorer
# r2_scorer = make_scorer(r2_score)

# # Cross-validation with polynomial features and Ridge regression
# # Cross-validation with polynomial features and Ridge regression
# alpha_value = 10  # Using a higher alpha value for stronger regularization
# model = Ridge(alpha=alpha_value)

# # Use 2 splits for cross-validation since you only have 2 samples
# scores = cross_val_score(model, X_poly, y, cv=2, scoring=r2_scorer)
# average_r2 = np.mean(scores)
# print("Average R^2 Score with Polynomial Features:", average_r2)


# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, shuffle=False, random_state=42)

# # Fit the Ridge regression model
# model.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = model.predict(X_test)

# # Convert predictions to a pandas Series to match indices
# y_pred_series = pd.Series(y_pred, index=y_test.index)

# # Calculate the Mean Squared Error and R^2 Score
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R^2 Score with Polynomial Features:", r2)

# # Compare the actual and predicted values
# comparison_df = pd.DataFrame({'Actual Approved for Services': y_test, 'Predicted Approved for Services': y_pred_series})
# print(comparison_df.head(20))

# # Generate a distribution plot to compare actual and predicted values
# ax1 = sns.kdeplot(y_test, color="r", label="Actual Value")
# sns.kdeplot(y_pred, color="b", label="Fitted Values", ax=ax1)
# plt.legend()
# plt.show()

# # Perform Grid Search for Ridge regression with polynomial features
# parameters1 = [{'alpha': [0.0001, 0.1, 1, 10, 100, 1000, 10000, 100000]}]
# RR = Ridge()
# Grid1 = GridSearchCV(RR, parameters1, cv=2)
# Grid1.fit(X_poly, y)

# # # # Output the best model
# print("Best Estimator with Polynomial Features:", Grid1.best_estimator_)

# # View the scores for each alpha
# scores = Grid1.cv_results_['mean_test_score']
# print("Mean Test Scores with Polynomial Features:", scores)

# # Plot the alpha values vs. the mean test scores
# alphas = parameters1[0]['alpha']
# plt.plot(alphas, scores)
# plt.xscale('log')
# plt.xlabel('Alpha')
# plt.ylabel('Mean Test Score')
# plt.title('Grid Search Results with Polynomial Features')
# plt.show()
