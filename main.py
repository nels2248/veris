#IMPORT STATEMENTS
import os
import json
import pandas as pd
from sys import platform
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

 
#VARIABLES
#PATH TO LOCAL FILE. 
#ASSUMES THAT NIGHTLY THIS WILL BE PULLED DOWN INTO THIS FOLDER AND RUN FROM THERE.
#THIS WILL BE RUNNING ON GITHUB NIGHTLY TO PICK UP LATEST DATA AS IT GETS ADDED
vcdb_path = "data/json/validated"  

#CREATE EMPTY DICTIONARY FOR DATASET
data = []

#ITERATE OVER ALL THE FILES 
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

#CONVERT JSON TO DATA FRAME
df = pd.json_normalize(data)

#ADD INCIDENT DATE AS YYYY-MM-DD FORMAT
#ONLY WHEN ALL COLUMNS ARE FILLED OUT

required_columns = ['timeline.incident.year', 'timeline.incident.month', 'timeline.incident.day']

#CREATE TEXT TO ADD TO THE HTML LATER 
incident_date_range_html = "" 

#CREATE INCIDENT DATE RANGE 
if all(col in df.columns for col in required_columns):
    df['incident_date'] = pd.to_datetime({
        'year': pd.to_numeric(df['timeline.incident.year'], errors='coerce'),
        'month': pd.to_numeric(df['timeline.incident.month'], errors='coerce'),
        'day': pd.to_numeric(df['timeline.incident.day'], errors='coerce')
    }, errors='coerce')
    
    # Only use non-null incident dates
    valid_dates = df['incident_date'].dropna()
    if not valid_dates.empty:
        #DEFINE THE FORMAT OF THE DATES (NAME OF MONTH, DD, YYYY)
        day_format = '%-d' if platform != 'win32' else '%#d'     

        min_date = valid_dates.min().strftime(f'%B {day_format}, %Y')
        max_date = valid_dates.max().strftime(f'%B {day_format}, %Y')
        incident_date_range_html = f"<p>With incident dates ranging from <strong>{min_date}</strong> to <strong>{max_date}</strong>.</p>"


#HAD ISSUES WITH PLUS.CREATED COLUMN SO TRYING TO CLEAN THIS UP SO I CAN PRINT IT
df['plus.created'] = df['plus.created'].str.strip()
df['plus.created'] = pd.to_datetime(df['plus.created'], errors='coerce', utc=True)
df['plus.created'] = pd.to_datetime(df['plus.created'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce', utc=True)


# Coerce bad values to NaT, parse cleanly
df['plus.created'] = pd.to_datetime(df['plus.created'], errors='coerce', utc=True)

# Drop NaT values and reassign
valid_dates_created = df['plus.created'].dropna()

valid_dates_created = df['plus.created'].dropna()
#DEFINE THE FORMAT OF THE DATES (NAME OF MONTH, DD, YYYY)
day_format = '%-d' if platform != 'win32' else '%#d'  

min_date_created = valid_dates_created.min().strftime(f'%B {day_format}, %Y')
max_date_created = valid_dates_created.max().strftime(f'%B {day_format}, %Y')

created_date_range_html = f"<p>With records created from <strong>{min_date_created}</strong> to <strong>{max_date_created}</strong><span style='color: red; font-size: small;'>(Not Fully Accurate as Issues with Format of Created Colulmn)</span></p>"


#HAD ISSUES WITH PLUS.CREATED COLUMN SO TRYING TO CLEAN THIS UP SO I CAN PRINT IT
df['plus.modified'] = df['plus.modified'].str.strip()
df['plus.modified'] = pd.to_datetime(df['plus.modified'], errors='coerce', utc=True)
df['plus.modified'] = pd.to_datetime(df['plus.modified'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce', utc=True)


# Coerce bad values to NaT, parse cleanly
df['plus.modified'] = pd.to_datetime(df['plus.modified'], errors='coerce', utc=True)

# Drop NaT values and reassign
valid_dates_modified = df['plus.modified'].dropna()

valid_dates_modified = df['plus.modified'].dropna()
#DEFINE THE FORMAT OF THE DATES (NAME OF MONTH, DD, YYYY)
day_format = '%-d' if platform != 'win32' else '%#d'  

min_date_modified = valid_dates_modified.min().strftime(f'%B {day_format}, %Y')
max_date_modified = valid_dates_modified.max().strftime(f'%B {day_format}, %Y')

modified_date_range_html = f"<p>With records modified from <strong>{min_date_modified}</strong> to <strong>{max_date_modified}</strong><span style='color: red; font-size: small;'>(Not Fully Accurate as Issues with Format of Modified Colulmn)</span>.</p>"


# GET SHAPE OF DATAFRAME
num_rows, num_columns = df.shape

# CREATE HTML PAGE
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Exploratory Analysis</title>
</head>
<body>
    <h1>Exploratory Analysis Summary</h1>
    <p>The dataset contains <strong>{num_rows}</strong> rows across <strong>{num_columns}</strong> columns.</p>
    {incident_date_range_html} 
    {created_date_range_html} 
    {modified_date_range_html} 
</body>
</html>
"""

# SAVE HTML PAGE
with open("exploratoryanalysis.html", "w", encoding="utf-8") as f:
    f.write(html_content)

    
#CREATE CORELATION PAIRS
# Create a new DataFrame with column names as rows
columns_as_rows = pd.DataFrame(df.columns, columns=['Column Names'])

# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Get the pairs of most correlated features (excluding self-correlations)
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

# Filter out self-correlations (where the correlation is 1) and duplicates (pairs like A-B and B-A)
corr_pairs = corr_pairs[corr_pairs < 1]

# Convert the multi-index series to a DataFrame and remove duplicates
corr_df = pd.DataFrame(corr_pairs).reset_index()
corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']

# Ensure that pairs like (A, B) and (B, A) are not duplicated
corr_df = corr_df[corr_df.apply(lambda row: row['Feature 1'] < row['Feature 2'], axis=1)]

# Save the correlation pairs to an HTML table
html_content_correlated = corr_df.to_html(index=False, escape=False)

# Add a header to the HTML content
header = """
<html>
<head>
    <title>Correlated Columns</title>
</head>
<body>
    <h1>Correlated Columns</h1>
    <p>Here are the most correlated feature pairs from your data, excluding self-correlations and duplicates.</p>
"""

# Combine the header and the HTML table
html_content_correlated  = header + html_content_correlated + "</body></html>"

# Write the content to an HTML file
with open('correlatedcolumns.html', 'w') as file:
    file.write(html_content_correlated)

#CREATE A DATA COLUMN THAT SHOWS ALL COLUMNS WITH VALUES.  WILL BE USED TO SHOW RECENT RECORDS IN AN EASIER WAY TO DISPLAY DATA
# Robust is_valid function
# Function to format valid values
def format_valid_fields(row):
    result = []
    for col in df.columns:
        val = row[col]
        try:
            if pd.isnull(val):
                continue
            elif isinstance(val, (list, np.ndarray, pd.Series)):
                if len(val) > 0 and any(bool(x) for x in val):
                    result.append(f"{col}: {val}")
            elif val != 0 and bool(val):
                result.append(f"{col}: {val}")
        except Exception:
            continue
    return '<br>'.join(result)

# Create new column
df['FieldsWithValues'] = df.apply(format_valid_fields, axis=1)

# Sort and get last 10
# Exclude rows where incident_date is NaT
df = df[df['incident_date'].notna()]

# Then sort and get last 10
last10 = df[df['incident_date'].notna()].sort_values('incident_date', ascending=False).head(10)


# Build HTML rows (2-column grid)
html_rows = ""
for _, row in last10.iterrows():
    html_rows += f"""
    <div class="record">
        <strong>{row['incident_date'].strftime('%B %d, %Y')}</strong><br>
        {row['FieldsWithValues']}
    </div>
    """

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Last 10 Incidents</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .record {{
            width: 100%;
            box-sizing: border-box;
            padding: 15px;
            border-bottom: 1px solid #ccc;
        }}
        .record:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <h2>Last 10 Incidents</h2>
    {html_rows}
</body>
</html>
"""

# Write to file
with open("last10incidents.html", "w", encoding="utf-8") as f:
    f.write(html_content)
    
    
# Step 1: Drop mostly-empty columns
df_filtered = df.loc[:, df.isnull().mean() < 0.5]

# Step 2: Keep numeric data
df_numeric = df_filtered.select_dtypes(include=[np.number])

# Step 3: Fill missing values
df_numeric = df_numeric.fillna(df_numeric.mean())

# Now you're safe to scale and cluster
# --- Step 2: Standardize the Data ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# --- Step 3: Apply KMeans Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df_numeric['Cluster'] = clusters

# --- Step 4: Reduce to 2D for Visualization ---
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
df_numeric['PC1'] = pca_components[:, 0]
df_numeric['PC2'] = pca_components[:, 1]

# --- Step 5: Plot the Clusters ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_numeric, x='PC1', y='PC2', hue='Cluster', palette='Set2')
plt.title('PCA Cluster Plot')

# Save plot to base64 for embedding
buf = BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')
buf.close()

# --- Step 6: Save HTML Report ---
html_report = f"""
<html>
<head><title>Clustering Report</title></head>
<body>
<h1>Clustering Report</h1>
<h2>Clustered Data (first 10 rows)</h2>
{df_numeric.head(10).to_html(index=False)}

<h2>Cluster Plot (via PCA)</h2>
<img src="data:image/png;base64,{img_base64}" />
</body>
</html>
"""

with open("clustering_report.html", "w") as f:
    f.write(html_report)

print("✅ Clustering report saved as 'clustering_report.html'")