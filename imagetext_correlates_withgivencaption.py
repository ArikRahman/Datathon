import cv2
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
import os
import string

def lemmatizer_func(input):


    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(input)
    stop_words = set(stopwords.words('english'))
    # Lemmatize words that end with 'ing' as verbs
    lemmatized_words = [lemmatizer.lemmatize(word, pos='v') if word.endswith('ing') else word for word in words]
    lemmatized_words = [word for word in lemmatized_words if word.lower() not in stop_words and word not in string.punctuation]
    # Combine the lemmatized words back into a string
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence


# Initialize the WordNetLemmatizer


# Function to extract text from image
def ocr_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image)
    return text

# Function to calculate text similarity
def text_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]


# Read the CSV into a DataFrame
df = pd.read_csv('./csvs/test.csv')  # Replace 'your_file.csv' with the path to your actual CSV file

# Directory where text files are saved
text_dir = 'text_files'  # Update this to match where your text files are stored

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    id_ = row['id']
    caption = row['caption']

    # Create a full path to the text file
    image_path = (f"downloaded_images/{id_}.png")

    # Check if text file exists
    if os.path.exists(image_path):
        # Read the text file into a variable
        

        # Print the id, text content, and caption




        # Perform OCR to extract text
        extracted_text = ocr_image(image_path)








        # Tokenize the sentence into words

        caption = lemmatizer_func(caption)



        extracted_text = lemmatizer_func(extracted_text)

        #####################


        # Calculate similarity
        similarity = text_similarity(extracted_text, caption)

        # Classification based on 60% cutoff
        classification = 1 if similarity >= 0.6 else 0

        print(f"Extracted Text: {extracted_text}")
        print(f"Caption: {caption}")
        print(f"Similarity: {similarity}")
        print(f"Classification: {classification}")
        print(f"ID: {id_}")
        # print(f"Text Content: {text_content}")
        print(f"Caption: {caption}")
        print("=" * 50)
    else:
        print(f"Text file with ID {id_} does not exist.")




# Test the functions
image_path = "downloaded_images/0a0ee14d-3cf0-4022-89f4-7bdc6f35ea78.png"  # Replace with the path to your image
caption = "Discover the key to success: embrace network and outsiders. Tap into connections and fresh perspectives to propel towards goals and open doors to endless opportunities. #SuccessTips #NetworkingPower"


