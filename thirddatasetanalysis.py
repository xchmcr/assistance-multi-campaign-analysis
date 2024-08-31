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

# Check for columns that need conversion from object to float (Days and Times)
# Assuming Days and Times can be interpreted as numeric
df['Days'] = pd.to_numeric(df['Days'], errors='coerce')
df['Times'] = pd.to_numeric(df['Times'], errors='coerce')

# Check for any NaN values introduced
print("\nNaN Values in Numeric Columns Before Dropping:")
print(df[numeric_columns + ['Days', 'Times']].isnull().sum())

# Print the initial number of rows
print("\nInitial Number of Rows:", df.shape[0])

# Option 1: Drop rows with NaN values only in essential columns (Days and Times)
df_cleaned = df.dropna(subset=['Days', 'Times'])

# Check DataFrame shape after dropping NaN values
print("\nDataFrame Shape After Cleaning:", df_cleaned.shape)

# Calculate the correlation matrix only if the DataFrame is not empty
if not df_cleaned.empty:
    correlation_matrix = df_cleaned.corr()

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
