import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime 

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

# Function to safely calculate summary statistics
def safe_summary(df):
    summary = {}
    
    for column in df.columns:
        # Data type
        summary[column] = {
            'Data Type': df[column].dtype,
            'Non-Null Count': df[column].count(),
            'NaN Count': df[column].isnull().sum(),
            'Example Values': df[column].head().tolist()
        }
        
        # For numeric columns, calculate min, max, and mean
        if pd.api.types.is_numeric_dtype(df[column]):
            summary[column]['Min'] = df[column].min()
            summary[column]['Max'] = df[column].max()
            summary[column]['Mean'] = df[column].mean()
        else:
            summary[column]['Min'] = None
            summary[column]['Max'] = None
            summary[column]['Mean'] = None
    
    # Convert to a DataFrame and transpose it to switch columns and rows
    summary_df = pd.DataFrame(summary).T
    return summary_df

# Get the summary DataFrame
summary_df = safe_summary(df)

# Transpose the DataFrame to make columns into rows
#summary_df = summary_df.T
 
# Display basic info and summary
#print("Dataset Overview:\n", df.info())
#print("\nSample Data:\n", df.head())

# Analyze incident types
if 'action.hacking.variety' in df.columns:
    df['action.hacking.variety'].explode().value_counts().head(10).plot(kind='bar', title='Top Hacking Varieties')
    plt.show()

# Analyze victim industries
if 'victim.industry' in df.columns:
    df['victim.industry'].value_counts().head(10).plot(kind='bar', title='Top Victim Industries')
    plt.show()

# Analyze attack patterns over time
if 'timeline.incident.year' in df.columns:
    df['timeline.incident.year'].value_counts().sort_index().plot(kind='line', marker='o', title='Incidents Over Time')
    plt.show()

print('ending')
print(datetime.datetime.now()) 