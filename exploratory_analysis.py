import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
import numpy as np
import html
import random
import plotly.graph_objects as go
import plotly.express as px

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


# Step 2: Load NAICS descriptions from Excel
naics = pd.read_excel('2022_NAICS_Descriptions.xlsx')


# Step 4: Ensure both columns are the same type
df['victim.industry'] = df['victim.industry'].astype(str)
naics['Code'] = naics['Code'].astype(str)

# Step 5: Merge on code
df = df.merge(naics, left_on='victim.industry', right_on='Code', how='left')





df = df.sort_index(axis=1)

# Step 3: Rename NAICS code column for clarity and merging
df = df.rename(columns={
    'Code': 'Victim Industry Code',
    'Description': 'Victim Industry Description',
    'Title': 'Victim Industry Title'
})

non_nan_columns = df.columns[~df.isna().any()]

# Apply function to concatenate NaN columns and include column names
# Function to concatenate non-NaN column values and include column names
# Function to concatenate non-NaN column values and handle lists/dictionaries
def concat_non_nan_columns(row):
    result = []
    for col in row.index:
        # Check if the value is not NaN
        if pd.notna(row[col]):
            # Convert lists or dicts to string representation if needed
            if isinstance(row[col], (list, dict)):
                result.append(f"{col}: {str(row[col])}")
            else:
                result.append(f"{col}: {row[col]}")
    return result

# Function to handle expansion of lists or dictionaries
def expand_column(df, column):
    """
    Expands a column that contains lists or dictionaries into separate columns.
    """
    if df[column].apply(lambda x: isinstance(x, (list, dict))).any():
        # If it's a list, expand it into separate columns
        expanded = df[column].apply(lambda x: pd.Series(x if isinstance(x, list) else [x]))
        expanded.columns = [f"{column}_{i}" for i in range(expanded.shape[1])]
        return expanded
    else:
        return None

# Iterate through the columns of the DataFrame and expand those containing lists or dicts
#for col in df.columns:
#    expanded = expand_column(df, col)
#    if expanded is not None:
#        # Drop the original column and add the expanded columns
#        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)


#df['concatenated_non_nan'] = df.apply(concat_non_nan_columns, axis=1)

# Create a new DataFrame with column names as rows
columns_as_rows = pd.DataFrame(df.columns, columns=['Column Names'])

# Function to safely calculate summary statistics
def safe_summary(df):
    summary = {}

    for column in df.columns:
        non_null_series = df[column].dropna()
        
        summary[column] = {
            'Data Type': df[column].dtype,
            'Non-Null Count': df[column].count(),
            'NaN Count': df[column].isnull().sum(),
            'Example Values': non_null_series.head().tolist()
        }

        if pd.api.types.is_numeric_dtype(df[column]):
            summary[column]['Min'] = df[column].min()
            summary[column]['Max'] = df[column].max()
            summary[column]['Mean'] = df[column].mean()
        else:
            summary[column]['Min'] = None
            summary[column]['Max'] = None
            summary[column]['Mean'] = None

    return pd.DataFrame(summary).T

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

required_cols = ['timeline.incident.year', 'timeline.incident.month', 'timeline.incident.day', 'victim.victim_id']

if all(col in df.columns for col in required_cols):
    # Drop rows with missing values in any of the required columns
    date_df = df.dropna(subset=required_cols)

    # Create datetime column
    date_df['incident_date'] = pd.to_datetime({
        'year': date_df['timeline.incident.year'].astype(int),
        'month': date_df['timeline.incident.month'].astype(int),
        'day': date_df['timeline.incident.day'].astype(int)
    }, errors='coerce')

    # Drop rows where date conversion failed
    date_df = date_df.dropna(subset=['incident_date'])

    # Get latest date and define 365-day window
    latest_date = date_df['incident_date'].max()
    #start_date = date_df['incident_date'].min()
    start_date = latest_date - pd.Timedelta(days=365)

    # Filter to incidents in that window
    recent_df = date_df[(date_df['incident_date'] >= start_date) & (date_df['incident_date'] <= latest_date)]
    recent_df = recent_df.sort_values('incident_date')

    # Daily counts and cumulative total
    daily_counts = recent_df['incident_date'].value_counts().sort_index()
    cumulative_counts = daily_counts.cumsum()

    # Create Plotly figure
    fig = go.Figure()

    # Add cumulative count line
    fig.add_trace(go.Scatter(
        x=cumulative_counts.index,
        y=cumulative_counts.values,
        mode='lines+markers',
        name='Cumulative Incidents',
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=5)
    ))

    # Get the first appearance for each victim
    first_appearances = recent_df.groupby('victim.victim_id')['incident_date'].min()

    # Reset the index of first_appearances so that incident_date becomes a column
    first_appearances = first_appearances.reset_index()

    # Now we can group by the incident_date properly
    date_victims = first_appearances.groupby('incident_date')['victim.victim_id'].apply(list)

    # Add labels for first appearances and hover information
    for date, victim_ids in date_victims.items():
        if date in cumulative_counts.index:
            # Create the text label for the first victim and others if applicable
            if len(victim_ids) > 1:
                label = f'{victim_ids[0]}, +{len(victim_ids) - 1} others'
            else:
                label = f'{victim_ids[0]}'

            # Add a single marker for the first appearance of victims on this date
            fig.add_trace(go.Scatter(
                x=[date],
                y=[cumulative_counts.loc[date]],
                mode='markers+text',
                name=f'First Appearances on {date.date()}',
                marker=dict(symbol='triangle-up', size=10, color='red'),
                text=[label],
                textposition='top center',
                hoverinfo='none'  # Hide hover info since we're showing the text on the chart
            ))

            # Add hover with all victim ids for this date
            fig.add_trace(go.Scatter(
                x=[date],
                y=[cumulative_counts.loc[date]],
                mode='markers',
                name=f'{date.date()}',
                marker=dict(symbol='triangle-up', size=10, color='red'),
                text=[', '.join(map(str, victim_ids))],  # Victim IDs as a comma-separated list
                hovertemplate=' %{text}<extra></extra>',  # Show all victim IDs in the hover box
                hoverinfo='text'  # Use custom hover text
            ))

    # Layout customization
    fig.update_layout(
        title=f'Running Total of Incidents ({start_date.date()} to {latest_date.date()})',
        xaxis_title='Date',
        yaxis_title='Cumulative Incident Count',
        hovermode='closest',  # Show details when hovering close to a point
        template='plotly_dark',
        autosize=True,
        showlegend=False,
        dragmode='zoom' 
        
    )

    # Save plot as HTML file
    fig.write_html('incident_running_total_with_hover_victims.html', config={
    'scrollZoom': True,  # Enable zooming with the mouse wheel
    }
        
        
)
    
    

    print("The plot has been saved as an interactive HTML file.")

else:
    print("One or more required columns are missing.")
 

 


date_df['incident_date'] = pd.to_datetime(date_df['incident_date'])
date_df['year_month'] = date_df['incident_date'].dt.to_period('M').astype(str)

# Get top industry per month
grouped = date_df.groupby(['year_month', 'Victim Industry Title']).size().reset_index(name='count')
top_industries = grouped.loc[grouped.groupby('year_month')['count'].idxmax()].sort_values(by='year_month', ascending=False)
max_count = top_industries['count'].max()

html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Top Industry per Month</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
        }

        .chart-container {
            width: 90%;
            margin: auto;
        }

        .bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }

        .month-label {
            width: 100px;
            font-weight: bold;
            white-space: nowrap;
        }

        .bar {
            height: 15px;
            position: relative;
            background-color: #4682B4;
            border-radius: 5px;
        }

        .bar-text {
            position: absolute;
            left: 5px;
            top: 3px;
            color: black; 
            white-space: nowrap;
        }
    </style>
</head>
<body>

<h2>Top Industry per Month</h2>
<div class="chart-container">
"""

# Add a row per month with bar scaled to count
for _, row in top_industries.iterrows():
    year_month = row['year_month']
    industry = row['Victim Industry Title']
    count = row['count']
    width_pct = (count / max_count) * 100

    html_content += f"""
    <div class="bar-row">
        <div class="month-label">{year_month}</div>
        <div class="bar" style="width: {width_pct}%; background-color: #{hash(industry) % 16777215:06x};">
            <div class="bar-text">{industry} ({count})</div>
        </div>
    </div>
    """

html_content += """
</div>
</body>
</html>
"""

with open('top_industry_per_month_final.html', 'w') as f:
    f.write(html_content)

print("HTML chart saved to 'top_industry_per_month_fullwidth.html'")
print('ending')
print(datetime.datetime.now()) 
 

industry_columns = [col for col in df.columns if 'industry' in col.lower()] 