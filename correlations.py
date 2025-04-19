import os
import json
import pandas as pd 
import datetime 
import numpy as np

print('starting')
print(datetime.datetime.now()) 
# Define the path to the local clone of the VCDB repo
vcdb_path = "C:/Users/nels2/Documents/GitHub/VCDB/data/json/validated"  # Update this with the actual path

data = []

# Iterate over all JSON files in the dataset directory
for root, dirs, files in os.walk(vcdb_path):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    json_data = json.load(f)
                    data.append(json_data)
                except json.JSONDecodeError:
                    print(f"Error reading {file_path}")

# Convert the JSON data to a DataFrame
df = pd.json_normalize(data)

df = df.sort_index(axis=1)

# Create a new DataFrame with column names as rows
columns_as_rows = pd.DataFrame(df.columns, columns=['Column Names'])

# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Get the pairs of most correlated features (excluding self-correlations)
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

# Remove self-correlations
filtered_corr = corr_pairs[corr_pairs < 1]


print('ending')
print(datetime.datetime.now()) 