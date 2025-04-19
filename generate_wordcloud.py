from wordcloud import  WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import json
import pandas as pd 
import datetime 
import numpy as np
from io import BytesIO
import base64
from PIL import Image
from collections import Counter
from io import BytesIO
import base64
import re

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

# Pre-clean function to extract words from summary
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return [word for word in tokens if word not in STOPWORDS and len(word) > 1]

# Filter DataFrame
df_filtered = df.dropna(subset=['summary', 'timeline.incident.year'])
df_filtered['incident_year'] = df_filtered['timeline.incident.year'].astype(int).astype(str)

# Sort years
years = sorted(df_filtered['incident_year'].unique())

# Start HTML
html_output = "<html><head><title>Word Clouds by Year</title></head><body>"
html_output += "<h1>Word Clouds by Year</h1>"

for year in years:
    print(year)
    subset = df_filtered[df_filtered['incident_year'] == year]
    summaries = subset['summary'].astype(str)
    text = ' '.join(summaries)

    # Create word cloud
    wordcloud = WordCloud(
        stopwords=STOPWORDS,
        background_color='white',
        width=800,
        height=400
    ).generate(text)

    # Save image to memory
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    # Tokenize and count words
    all_tokens = []
    for summary in summaries:
        all_tokens.extend(clean_and_tokenize(summary))
    word_counts = Counter(all_tokens).most_common(10)

    # Add to HTML
    html_output += f"<h2>{year}</h2>"
    html_output += f'<img src="data:image/png;base64,{img_str}" style="width:800px;height:auto;"><br>'

    # Add table
    html_output += f"<p>Total records: {len(subset)}</p>"
    html_output += "<table border='1' cellpadding='5' cellspacing='0'><tr><th>Word</th><th>Count</th></tr>"
    for word, count in word_counts:
        html_output += f"<tr><td>{word}</td><td>{count}</td></tr>"
    html_output += "</table><br><br>"

html_output += "</body></html>"

# Save to file
with open("wordclouds_with_tables_by_year.html", "w", encoding='utf-8') as f:
    f.write(html_output)

print('end')
print(datetime.datetime.now()) 