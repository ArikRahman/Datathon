import requests

# Open the text file in read mode
with open("image_links.txt", "r") as f:
    # Loop through each line (each URL)
    for i, line in enumerate(f):
        image_url = line.strip()  # Remove any leading/trailing whitespace or newline characters
        
        # Fetch the image data
        response = requests.get(image_url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Open a file in binary write mode and save the image
            with open(f"downloaded_image_{i+1}.png", "wb") as img_file:
                img_file.write(response.content)
        else:
            print(f"Failed to download image at {image_url}, status code: {response.status_code}")
