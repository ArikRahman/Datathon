import requests
import pandas as pd
import os

# Read the CSV file into a DataFrame
df = pd.read_csv("csvs/test.csv")

# Create a directory to store the downloaded images
os.makedirs("downloaded_images", exist_ok=True)

# Loop through the DataFrame and download each image
for index, row in df.iterrows():
    image_url = row['image']
    image_id = row['id']
    
    # Fetch the image data
    response = requests.get(image_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the image using the 'id' as the filename
        with open(f"downloaded_images/{image_id}.png", "wb") as f:
            f.write(response.content)
    else:
        print(f"Failed to download image at {image_url}, status code: {response.status_code}")
