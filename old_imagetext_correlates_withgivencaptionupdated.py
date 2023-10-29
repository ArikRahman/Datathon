import cv2
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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

# Test the functions
image_path = "downloaded_images/0a0ee14d-3cf0-4022-89f4-7bdc6f35ea78.png"  # Replace with the path to your image
caption = "Discover the key to success: embrace network and outsiders. Tap into connections and fresh perspectives to propel towards goals and open doors to endless opportunities. #SuccessTips #NetworkingPower"






# Perform OCR to extract text
extracted_text = ocr_image(image_path)






lemmatizer = WordNetLemmatizer()


# Tokenize the sentence into words
words = word_tokenize(caption)

# Lemmatize words that end with 'ing' as verbs
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') if word.endswith('ing') else word for word in words]

# Combine the lemmatized words back into a string
lemmatized_sentence = ' '.join(lemmatized_words)



caption = lemmatized_sentence


words = word_tokenize(extracted_text)

# Lemmatize words that end with 'ing' as verbs
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') if word.endswith('ing') else word for word in words]

# Combine the lemmatized words back into a string
lemmatized_sentence = ' '.join(lemmatized_words)



extracted_text = lemmatized_sentence

#####################


# Calculate similarity
similarity = text_similarity(extracted_text, caption)

# Classification based on 60% cutoff
classification = 1 if similarity >= 0.6 else 0

print(f"Extracted Text: {extracted_text}")
print(f"Caption: {caption}")
print(f"Similarity: {similarity}")
print(f"Classification: {classification}")