import pandas as pd

# Load the CSV file
file_path = "data/BTC-USDT_5m.csv"  # Adjust the path to your file
df = pd.read_csv(file_path)

# Print column names and data types
print("Column Names:", df.columns)
print("Data Types:\n", df.dtypes)

# Print the first few rows of the data
print("Sample Data:\n", df.head())
