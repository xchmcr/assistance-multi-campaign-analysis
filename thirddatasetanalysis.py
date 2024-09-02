import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\Migue\assistance-multi-campaign-analysis\sample_data-radiodata.csv'
df = pd.read_csv(file_path)

# Print the initial shape of the DataFrame
print("Initial DataFrame Shape:", df.shape)

# Remove dollar signs, commas, and percentage symbols, then convert to numeric
df['Ord. Buy Rate'] = df['Ord. Buy Rate'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['$ ORD'] = df['$ ORD'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['$ SPENT'] = df['$ SPENT'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['UNQ >= :01'] = pd.to_numeric(df['UNQ >= :01'], errors='coerce')

# Convert 'Week' to datetime to ensure proper sorting (if needed)
df['Week Of'] = pd.to_datetime(df['Week Of'])

# Check data types of all columns
print("\nData Types:")
print(df.dtypes)

# List all relevant numeric columns
numeric_columns = ['Ord. Buy Rate', '$ ORD', '$ SPENT', 'UNQ >= :01', 'Length', 'Ord Spots', 'Spots Ran']

# Convert all relevant columns to float, handling errors
for col in numeric_columns:
    df[col] = df[col].astype(float)

# Optionally, convert 'Days' and 'Times' to categorical if they are not numeric:
df['Days'] = df['Days'].astype('category')
df['Times'] = df['Times'].astype('category')

# Check for any NaN values in numeric columns
print("\nNaN Values in Numeric Columns Before Dropping:")
print(df[numeric_columns].isnull().sum())

# Print the initial number of rows
print("\nInitial Number of Rows:", df.shape[0])

# Drop rows with NaN values only in essential columns (if any)
df_cleaned = df.dropna(subset=numeric_columns)

# Check DataFrame shape after dropping NaN values
print("\nDataFrame Shape After Cleaning:", df_cleaned.shape)

# Select only numeric columns for the correlation matrix
df_numeric = df_cleaned[numeric_columns]

# Calculate the correlation matrix only if the DataFrame is not empty
if not df_numeric.empty:
    correlation_matrix = df_numeric.corr()

    # Print the correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
else:
    print("DataFrame is empty after cleaning; no correlation matrix to display.")

#ridge regression model with 2nd degree polynomials: 
#helps us define which combinations of features (like "Ord Spots," "Spots Ran," "ORD," "Ord. Buy Rate") are leading to the most efficient use of spending.

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

# Create a copy of the DataFrame before one-hot encoding to watch the original dataframe
df_original = df.copy()

# Print the first 10 rows to check the DataFrame
#print(df.head(20))

# Select only the desired features for the polynomial regression model
features = ["Ord Spots", "Spots Ran", "ORD", "Ord. Buy Rate"]
X = df[[col for col in df.columns if any(feature in col for feature in features)]]
y = df['$ SPENT']  # Target

# Standardize the features before polynomial transformation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add second-order polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Define the R^2 scorer
r2_scorer = make_scorer(r2_score)

# Cross-validation with polynomial features and Ridge regression
alpha_value = 10  # Using a higher alpha value for stronger regularization
model = Ridge(alpha=alpha_value)
scores = cross_val_score(model, X_poly, y, cv=3, scoring=r2_scorer)
average_r2 = np.mean(scores)
print("Average R^2 Score with Polynomial Features:", average_r2)

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
print("R^2 Score with Polynomial Features:", r2)

# Compare the actual and predicted values
comparison_df = pd.DataFrame({'Actual SPENT': y_test, 'Predicted SPENT': y_pred_series})
print(comparison_df.head(20))

# Generate a distribution plot to compare actual and predicted values
ax1 = sns.kdeplot(df['$ SPENT'], color="r", label="Actual Value")
sns.kdeplot(y_pred, color="b", label="Fitted Values", ax=ax1)
plt.legend()
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
plt.plot(alphas, scores)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Test Score')
plt.title('Grid Search Results with Polynomial Features')
plt.show()

# Print the original DataFrame before one-hot encoding
print("Original DataFrame before one-hot encoding:")
#print(df_original.head(20))



#report
# Add predicted values to the original DataFrame
df_original['Predicted_$ SPENT'] = np.nan  # Initialize with NaN
df_original.loc[y_pred_series.index, 'Predicted_$ SPENT'] = y_pred_series

# Calculate the efficiency ratio
df_original['Efficiency_Ratio'] = df_original['Predicted_$ SPENT'] / df_original['$ SPENT']

# Remove rows with NaN values in 'Predicted_$ SPENT' or 'Efficiency_Ratio'
df_cleaned = df_original.dropna(subset=['Predicted_$ SPENT', 'Efficiency_Ratio'])

# Sort by Efficiency Ratio (ascending to find the most efficient spending)
efficient_combinations = df_cleaned.sort_values(by='Efficiency_Ratio', ascending=True).reset_index(drop=True)

# Prepare the report
report = []
for station in efficient_combinations['Station'].unique():
    station_df = efficient_combinations[efficient_combinations['Station'] == station].head(10)  # Top 10 for each station
    report.append(f"For the station '{station}', the most efficient combinations are:")
    for index, row in station_df.iterrows():
        report.append(f"Product: {row['Product']}, Length: {row['Length']}, Days: {row['Days']}, Times: {row['Times']}, Actual $ SPENT: ${row['$ SPENT']:.2f}, Predicted $ SPENT: ${row['Predicted_$ SPENT']:.2f}, Efficiency Ratio: {row['Efficiency_Ratio']:.2f}")
    report.append("")  # Add a blank line for readability

# Print the report
print("\n".join(report))





# Report that showcases the top 10 most profitable combinations for each format category!!!

# import pandas as pd
# # Add a new column to df_original for predictions
# df_original['Predicted_Profit'] = np.nan  # Initialize with NaN
# # Add predicted values to the DataFrame
# # Assuming the predictions align with the rows in df_original
# df_original.loc[y_pred_series.index, 'Predicted_Profit'] = y_pred_series

# # Assuming df_original is the DataFrame with actual data and 'predicted_profit' column is added
# df_original['Profit_to_Cost'] = df_original['Profit'] / df_original['Cost']
# # Calculate the ratio for predicted profits (if available)
# df_original['Predicted_Profit_to_Cost'] = df_original['Predicted_Profit'] / df_original['Cost']
# # Identify combinations with the highest Profit-to-Cost ratio
# efficient_combinations = df_original[['Station', 'Format', 'Creative', 'Length', 'Days', 'Times', 'Cost', 'Profit', 'Profit_to_Cost']].sort_values(by='Profit_to_Cost', ascending=False).reset_index(drop=True)
# # Prepare the report
# report = []
# for station in efficient_combinations['Station'].unique():
#     station_df = efficient_combinations[efficient_combinations['Station'] == station]
#     report.append(f"For the station '{station}', the most efficient combinations are:")
#     for index, row in station_df.iterrows():
#         report.append(f"Format: {row['Format']}, Creative: {row['Creative']}, Length: {row['Length']}, Days: {row['Days']}, Times: {row['Times']}, Cost: ${row['Cost']:.2f}, Profit: ${row['Profit']:.2f}, Profit-to-Cost Ratio: {row['Profit_to_Cost']:.2f}")
#     report.append("")  # Add a blank line for readability

# # Print the report
# print("\n".join(report))

#so far we have : 
# regression plots against profit
# third-order polynomial features
# cross validation model(tells us how well our ridge regression model is doing),
# a ridge reg model,
# the average r2 (made by cross validation)
# the mse
# a basic comparison of prices
#  a grid search
# print the original dataset
# report that showcases the top 10 most profitable combinations for each format category
