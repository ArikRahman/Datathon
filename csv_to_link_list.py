import pandas as pd
import requests

# Assume 'data.csv' is your CSV file
df = pd.read_csv('csvs/test.csv')

# Extract the 'image' column
image_links = df['image']

# Open a text file in write mode
with open('image_links.txt', 'w') as f:
    # Write each link to the file
    for link in image_links:
        f.write(f"{link}\n")
