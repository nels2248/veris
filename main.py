#IMPORT STATEMENTS
import os
import json
import pandas as pd
from sys import platform

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
